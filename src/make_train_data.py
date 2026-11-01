"""Synthetic training data generator for QLoRA fine-tuning.

Queries config.TEACHER_MODEL via NIMClient for each chunk in data/processed/chunks.json
to generate up to 4 high-quality question-answer pairs:
- direct: Factual, standard question directly answered by the text.
- paraphrase: Semantically varied phrasing of the core question.
- hinglish: Romanized Hindi + English phrasing as a farmer would type/speak.

Verification & Quality Filters:
1. Grounding check: evidence_quote must appear verbatim (normalized whitespace) in chunk.
2. Contamination prevention: drop question if cosine similarity > 0.85 with ANY question
   in data/test_qa.json (using BAAI/bge-small-en-v1.5).
3. Deduplication: drop exact-duplicate questions.

Caches responses by chunk_id, respects <40 RPM rate limits, handles 429 with backoff,
and outputs data/train_qa.jsonl.
"""

import argparse
import json
import logging
import csv
import random
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Heavy libraries (NIMClient, sentence_transformers, PyTorch) imported lazily inside generate_training_data()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_train_data")


# ── prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert curriculum designer and agricultural data specialist for Indian Government Farmer Schemes (PM-KISAN, PMFBY, Kisan Credit Card).
Your task is to generate high-quality question-answering training pairs strictly based on provided guideline excerpts."""

USER_PROMPT_TEMPLATE = """You are given an excerpt from official Indian agricultural scheme guidelines.
Analyze the text and generate a JSON list of up to 4 question-answer items based ONLY on this text.

Format for each item:
{{
  "question": "<self-contained question>",
  "answer": "<concise 1-3 sentences in English>",
  "evidence_quote": "<verbatim sentence or phrase copied exactly from the chunk>",
        "style": "<direct | paraphrase | hinglish>"
}}

CRITICAL RULES:
1. Answerable ONLY from this chunk: Every question must be fully answerable using ONLY this text.
2. Self-contained: Questions must never say 'this document', 'the chunk', 'the text above', or 'the passage'. Mention the specific scheme (PM-KISAN, PMFBY, KCC) or concept explicitly.
3. Style Diversity:
   - direct: Clear, straightforward factual question.
   - paraphrase: Reworded, scenario-based, or natural variation. Include at least 1 when possible.
   - hinglish: Romanized Hindi mixed with English (e.g. 'PM-KISAN me installment kaise aati hai?', 'KCC loan par kitna subsidy milta hai?'). Include at least 1 when text has sufficient facts.
4. Grounded Quote: evidence_quote MUST be an exact verbatim substring copied from the chunk.
5. Answer length and language: 1 to 3 clear, factual sentences in English only. Even for a Hinglish question, the answer must be plain English, never Hindi or Hinglish.
6. Do not mention PDF page numbers or clause numbers in questions.
7. Non-informative chunks: If the text is purely boilerplate, table of contents, form fields, contact numbers, or headers with no substantive scheme rules, return an empty list: []

Chunk Scheme: {scheme}
Source File: {source_file} (Page {page})
Text:
\"\"\"
{text}
\"\"\"

Return ONLY the raw JSON list (or []). No preamble or reasoning."""


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_ws(text: str) -> str:
    """Normalize internal and external whitespace."""
    return " ".join(text.strip().split())


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Check if normalized evidence quote appears in normalized chunk text."""
    if not quote or not quote.strip():
        return False
    norm_q = normalize_ws(quote).lower()
    norm_c = normalize_ws(chunk_text).lower()
    return norm_q in norm_c


def exceeds_similarity(candidate_embedding, reference_embeddings, threshold: float) -> bool:
    """Cosine filter for already-normalized embeddings (used by train and test checks)."""
    if reference_embeddings is None or len(reference_embeddings) == 0:
        return False
    return float(np.max(np.dot(np.asarray(reference_embeddings), candidate_embedding))) > threshold


def answer_is_english(text: str) -> bool:
    if re.search(r"[\u0900-\u097f]", text):
        return False
    romanized_hindi = {"kisan", "kisaan", "kisanon", "fasal", "katayi", "chukane", "chukata",
                       "wale", "wali", "mein", "me", "kitna", "kitni", "mahine", "mahina",
                       "ke", "liye", "par", "hai", "hain", "ka", "ki", "ko", "se", "tak",
                       "aur", "ya", "kya", "bad", "chhe", "cheh", "saal", "paisa", "milta",
                       "milega", "milenge", "karne", "wala", "wali"}
    return not bool(romanized_hindi & {word.lower() for word in re.findall(r"[A-Za-z]+", text)})


def question_mentions_wrong_scheme(question: str, scheme: str) -> bool:
    question_upper = question.upper()
    schemes = {"PM-KISAN", "PMFBY", "KCC"}
    normalized = scheme.upper()
    return any(other in question_upper for other in schemes - {normalized})


def extract_json_array(text: str) -> Optional[List[Any]]:
    """Find a valid JSON array inside a response with optional extra prose/fences."""
    text = text.strip()
    decoder = json.JSONDecoder()
    candidates = []
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            candidates.append((len(value), start, end, value))
    if not candidates:
        return None
    # Prefer the most complete list; on ties use the last one, which is usually
    # the final answer after explanatory text from a model.
    return max(candidates, key=lambda item: (item[0], item[1]))[3]


def load_test_qa_questions() -> List[str]:
    """Load all question texts from data/test_qa.json."""
    candidates = [
        PROJECT_ROOT / "data" / "test_qa.json",
        PROJECT_ROOT / "Data" / "test_qa.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                items = json.load(f)
                return [normalize_ws(item["question"]) for item in items if "question" in item]
    logger.warning("No test_qa.json found; skipping test-set contamination check.")
    return []


# ── main generation workflow ─────────────────────────────────────────────────

def generate_training_data(
    limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    refusal_examples: int = 0,
) -> None:
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        chunks_path = PROJECT_ROOT / "Data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if limit is not None and limit > 0:
        chunks = chunks[:limit]

    out_file = output_path or (PROJECT_ROOT / "data" / "train_qa.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path or (PROJECT_ROOT / "cache" / "train_gen_cache.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Load generation cache: chunk_id -> raw_response_text
    gen_cache: Dict[str, str] = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                gen_cache = json.load(f)
            logger.info("Loaded %d cached chunk responses from %s", len(gen_cache), cache_file)
        except Exception as e:
            logger.warning("Could not read cache file %s: %s", cache_file, e)

    # Initialize NIM client
    from src.nim_client import NIMClient
    nim_client = NIMClient()
    teacher_model = config.TEACHER_MODEL

    # Load test-set embeddings for contamination filter (cosine sim > 0.85)
    test_questions = load_test_qa_questions()
    from src.rag import get_embedding_model
    embedder = get_embedding_model()

    if test_questions:
        logger.info("Encoding %d test-set questions for similarity thresholding...", len(test_questions))
        test_embeddings = embedder.encode(test_questions, normalize_embeddings=True, show_progress_bar=False)
    else:
        test_embeddings = None

    # Tracking metrics
    stats = {
        "chunks_processed": 0,
        "chunks_returned_empty": 0,
        "items_generated": 0,
        "dropped_missing_fields": 0,
        "dropped_quote_not_in_chunk": 0,
        "dropped_test_similarity": 0,
        "dropped_duplicate": 0,
        "dropped_bad_answer": 0,
        "dropped_forbidden_phrase": 0,
        "dropped_page_reference": 0,
        "dropped_non_english_answer": 0,
        "dropped_wrong_scheme": 0,
        "dropped_near_duplicate": 0,
        "dropped_bad_style": 0,
        "dropped_json_parse_error": 0,
        "final_valid_items": 0,
        "by_scheme": {},
        "by_style": {},
    }

    seen_questions: Set[str] = set()
    verified_records: List[Dict[str, Any]] = []
    kept_embeddings = []

    logger.info("Starting synthesis on %d chunks with model %s...", len(chunks), teacher_model)
    teacher_cache_version = hashlib.sha256(
        (teacher_model + USER_PROMPT_TEMPLATE + "reasoning_effort=none").encode("utf-8")
    ).hexdigest()[:16]

    for idx, chunk in enumerate(chunks, 1):
        cid = chunk.get("id", f"chunk_{idx}")
        cache_key = f"{cid}|{teacher_cache_version}"
        scheme = chunk.get("scheme", "UNKNOWN")
        source_file = chunk.get("source_file", "")
        page = chunk.get("page", 0)
        chunk_text = chunk.get("text", "")

        stats["chunks_processed"] += 1

        # Check local disk cache first
        if cache_key in gen_cache:
            raw_reply = gen_cache[cache_key]
        else:
            prompt = USER_PROMPT_TEMPLATE.format(
                scheme=scheme,
                source_file=source_file,
                page=page,
                text=chunk_text,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                raw_reply, _lat, _cached = nim_client.generate(
                    messages=messages,
                    model=teacher_model,
                    temperature=0.3,
                    max_tokens=2048,
                    use_cache=True,
                    request_options={"reasoning_effort": "none"},
                )
            except Exception as e:
                logger.error("Error calling NIM for chunk %s: %s", cid, e)
                continue

            gen_cache[cache_key] = raw_reply
            # Persist cache periodically
            if len(gen_cache) % 10 == 0:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(gen_cache, f, ensure_ascii=False)

        parsed_items = extract_json_array(raw_reply)
        if parsed_items is None:
            stats["dropped_json_parse_error"] += 1
            continue

        if len(parsed_items) == 0:
            stats["chunks_returned_empty"] += 1
            continue

        for item in parsed_items[:4]:
            stats["items_generated"] += 1
            if not isinstance(item, dict):
                stats["dropped_missing_fields"] += 1
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            quote = str(item.get("evidence_quote", "")).strip()
            style = str(item.get("style", "direct")).strip().lower()

            if not q or not a or not quote:
                stats["dropped_missing_fields"] += 1
                continue

            if not a or len(a.split()) > 80:
                stats["dropped_bad_answer"] += 1
                continue
            if not answer_is_english(a):
                stats["dropped_non_english_answer"] += 1
                continue
            if question_mentions_wrong_scheme(q, scheme):
                stats["dropped_wrong_scheme"] += 1
                continue
            if style not in {"direct", "paraphrase", "hinglish"}:
                stats["dropped_bad_style"] += 1
                continue
            if re.search(r"this document|the chunk|the text above|according to the passage", q, re.I):
                stats["dropped_forbidden_phrase"] += 1
                continue
            if re.search(r"\b(?:page|pages|clause|paragraph|para)\s+(?:no\.?\s*)?\d+", q, re.I):
                stats["dropped_page_reference"] += 1
                continue

            # Check 1: Evidence quote in chunk
            if not quote_in_chunk(quote, chunk_text):
                stats["dropped_quote_not_in_chunk"] += 1
                continue

            # Check 2: Exact duplicate question
            norm_q = normalize_ws(q).lower()
            if norm_q in seen_questions:
                stats["dropped_duplicate"] += 1
                continue

            # Check 3: Semantic duplicates and test-set contamination.
            q_emb = embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
            if exceeds_similarity(q_emb, kept_embeddings, 0.95):
                stats["dropped_near_duplicate"] += 1
                continue
            if test_embeddings is not None and len(test_embeddings) > 0:
                if exceeds_similarity(q_emb, test_embeddings, 0.85):
                    stats["dropped_test_similarity"] += 1
                    continue

            # Valid item
            seen_questions.add(norm_q)
            kept_embeddings.append(q_emb)
            stats["final_valid_items"] += 1
            stats["by_scheme"][scheme] = stats["by_scheme"].get(scheme, 0) + 1
            stats["by_style"][style] = stats["by_style"].get(style, 0) + 1

            record = {
                "question": q,
                "answer": a,
                "chunk_id": cid,
                "source_file": source_file,
                "page": page,
                "scheme": scheme,
                "style": style,
                "evidence_quote": quote,
            }
            verified_records.append(record)

        if idx % 20 == 0 or idx == len(chunks):
            logger.info(
                "Progress: %d/%d chunks processed | %d valid items generated so far",
                idx, len(chunks), stats["final_valid_items"]
            )

    # Optional, explicit refusal examples are generated separately and included
    # in all counts as a training treatment.
    if refusal_examples:
        if refusal_examples < 0:
            raise ValueError("--refusal-examples must be non-negative")
        chunk_embeddings = embedder.encode([c.get("text", "") for c in chunks], normalize_embeddings=True,
                                           show_progress_bar=False)
        refusal_prompt = (
            f"Create exactly {refusal_examples} distinct, plausible farmer questions that are outside the "
            "provided Indian farmer scheme documents (other states, years, or schemes). Return a JSON array "
            "of objects with question and answer only. Every answer must be exactly "
            "'Not found in the documents'. Do not use any fact in the supplied corpus."
        )
        raw, _, _ = nim_client.generate(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": refusal_prompt}],
            model=teacher_model, temperature=0.3, max_tokens=2048, use_cache=True,
            request_options={"reasoning_effort": "none"},
        )
        raw_items = extract_json_array(raw) or []
        for item in raw_items[:refusal_examples]:
            q = str(item.get("question", "")).strip()
            if not q or str(item.get("answer", "")).strip() != "Not found in the documents":
                continue
            q_emb = embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
            if exceeds_similarity(q_emb, test_embeddings, 0.85) or exceeds_similarity(q_emb, chunk_embeddings, 0.80):
                continue
            if exceeds_similarity(q_emb, kept_embeddings, 0.95):
                continue
            verified_records.append({"question": q, "answer": "Not found in the documents", "chunk_id": "refusal",
                                     "source_file": "", "page": 0, "scheme": "out_of_scope",
                                     "style": "refusal", "evidence_quote": ""})
            kept_embeddings.append(q_emb)
        stats["refusal_examples_kept"] = sum(r["style"] == "refusal" for r in verified_records)
        stats["by_style"]["refusal"] = stats["refusal_examples_kept"]
        stats["by_scheme"]["out_of_scope"] = stats["refusal_examples_kept"]
        stats["final_valid_items"] = len(verified_records)

    # Save final cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(gen_cache, f, ensure_ascii=False)

    # Write output JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in verified_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print summary report
    print("\n" + "=" * 60)
    print("      SYNTHETIC TRAINING DATA GENERATION REPORT")
    print("=" * 60)
    print(f"Chunks processed            : {stats['chunks_processed']}")
    print(f"Chunks returned [] (empty)  : {stats['chunks_returned_empty']}")
    print(f"Total raw items generated   : {stats['items_generated']}")
    print("-" * 60)
    print("Filter Drops:")
    print(f"  Missing fields            : {stats['dropped_missing_fields']}")
    print(f"  JSON parse errors         : {stats['dropped_json_parse_error']}")
    print(f"  Quote not verbatim in chunk: {stats['dropped_quote_not_in_chunk']}")
    print(f"  Test similarity > 0.85    : {stats['dropped_test_similarity']}")
    print(f"  Duplicate question        : {stats['dropped_duplicate']}")
    print(f"  Near-duplicate question   : {stats['dropped_near_duplicate']}")
    print(f"  Bad/long answer           : {stats['dropped_bad_answer']}")
    print(f"  Forbidden phrase          : {stats['dropped_forbidden_phrase']}")
    print(f"  Page/clause reference     : {stats['dropped_page_reference']}")
    print(f"  Invalid style             : {stats['dropped_bad_style']}")
    print(f"  Non-English answer        : {stats['dropped_non_english_answer']}")
    print(f"  Wrong scheme in question  : {stats['dropped_wrong_scheme']}")
    print("-" * 60)
    print(f"Final valid training items  : {stats['final_valid_items']}")
    print(f"Output saved to             : {out_file}")
    print("\nCounts by Scheme:")
    for sch, cnt in sorted(stats["by_scheme"].items()):
        print(f"  {sch:<20}: {cnt}")
    print("\nCounts by Style:")
    for sty, cnt in sorted(stats["by_style"].items()):
        print(f"  {sty:<20}: {cnt}")
    print("\nExamples (up to 3 per style):")
    for sty in ("direct", "paraphrase", "hinglish", "refusal"):
        examples = [r for r in verified_records if r["style"] == sty][:3]
        for ex in examples:
            print(f"  [{sty}] Q: {ex['question']}\n         A: {ex['answer']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QLoRA training data from guideline chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process (for testing)")
    parser.add_argument("--out", "--output", dest="output", type=Path, default=None, help="Output path for train_qa.jsonl")
    parser.add_argument("--cache", type=Path, default=None, help="Path for chunk response cache JSON")
    parser.add_argument("--refusal-examples", type=int, default=0)
    args = parser.parse_args()

    generate_training_data(
        limit=args.limit,
        output_path=args.output,
        cache_path=args.cache,
        refusal_examples=args.refusal_examples,
    )


if __name__ == "__main__":
    main()
