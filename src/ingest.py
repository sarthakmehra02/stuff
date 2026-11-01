"""PDF Ingestion and Chunking Pipeline for Indian Farmer Schemes.

This module processes official farmer scheme guidelines (PM-KISAN, PMFBY, KCC)
located in data/raw_pdfs/ using PyMuPDF (fitz) and tiktoken:
- Applies unicodedata.normalize('NFKC', ...) to eliminate ligatures (e.g. 'fi', 'fl')
  and collapses repeated whitespace.
- Filters PDF pages according to PAGE_RANGES defined in config.py, printing the
  number of pages kept and skipped per file.
- Tracks active numbered clause markers ('clause_at_start') across sections,
  supporting multi-level clauses (e.g. '10.4', '5.3.2') and short numbered headings
  (e.g. '11. Validity of the list of beneficiaries'), while ignoring roman numerals
  and list letters.
- Enforces sentence-aligned chunk boundaries: every chunk ends at the last full
  sentence before the 400-token hard cap, and the next chunk starts with the last
  1-2 full sentences of the previous chunk (~50 tokens), never mid-sentence.
- Validates that total chunk tokens / document tokens is strictly between 1.0 and 1.2
  (failing loudly if higher).
- Asserts that exactly 0 chunks exceed the 400 token hard cap.
- Saves structured records to data/processed/chunks.json with id, text, scheme,
  source_file, page, clause_at_start, and tokens.

CLI Example:
    python src/ingest.py
    python src/ingest.py --chunk-tokens 400 --overlap 50 --output data/processed/chunks.json
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import fitz  # PyMuPDF
import tiktoken

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PAGE_RANGES, PATHS, ensure_directories


# Real Tokenizer setup (cl100k_base used by GPT-4 / modern LLM pipelines)
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken cl100k_base encoding."""
    return len(TOKENIZER.encode(text))


# Regex patterns for clause extraction
ROMAN_NUMERAL_PATTERN = re.compile(r"^(?:[ivxlcdm]+|[IVXLCDM]+)[\.\)]\s*", re.IGNORECASE)
LIST_LETTER_PATTERN = re.compile(r"^(?:\([a-zA-Z0-9]+\)|[a-zA-Z][\.\)])\s*")
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:Page\s+)?(?:[-–—]?\s*\d+\s*[-–—]?|\d+\s+of\s+\d+)\s*$",
    re.IGNORECASE,
)


def extract_clause_marker(text: str) -> Optional[str]:
    """Extract a numbered clause marker from a line or heading.

    Supports:
      - Multi-level clause markers (e.g., '10.4', '5.3.2', '5.1.6 Fixation of Sub-limits')
      - Short numbered headings (e.g., '11. Validity of the list of beneficiaries', '1. Scheme')
    Excludes:
      - Roman numerals ('I.', 'ii.', 'iv.')
      - List letters ('(a)', '(b)', 'A.', 'B.')
      - Dates formatted with dots ('01.12.2018', '31.01.2019')

    Args:
        text: Input string to inspect.

    Returns:
        Clause marker string if detected, or None.
    """
    text_clean = text.strip()
    if not text_clean:
        return None

    # Exclude roman numerals and list letter bullets
    if ROMAN_NUMERAL_PATTERN.match(text_clean) or LIST_LETTER_PATTERN.match(text_clean):
        return None

    # Exclude date strings (e.g. 01.12.2018 or 31.01.2019)
    if re.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}", text_clean):
        return None

    # 1. Decimal / multi-level clause pattern: "10.4", "5.3.2", "5.1.6 Fixation of Sub-limits"
    m_dec = re.match(r"^(\d+\.\d+(?:\.\d+)*)(?:\s+(.*))?$", text_clean)
    if m_dec:
        num = m_dec.group(1)
        rest = (m_dec.group(2) or "").strip()
        # If rest is a short heading title (<= 45 chars, <= 7 words, title/uppercase, doesn't end in punctuation)
        if (
            rest
            and len(rest) <= 45
            and len(rest.split()) <= 7
            and not rest.endswith((".", ",", ";", ":", "?", "!"))
            and rest[0].isupper()
        ):
            return f"{num} {rest}"
        return num

    # 2. Short numbered heading: "11. Validity of the list of beneficiaries", "1. Scheme"
    m_heading = re.match(r"^(\d+)\.\s+([A-Z][A-Za-z0-9\s,\-\(\)\/\'\"]{2,60})$", text_clean)
    if m_heading:
        title = m_heading.group(2).strip()
        if not title.endswith((".", ",", ";", ":", "?", "!")) and len(title.split()) <= 10:
            return f"{m_heading.group(1)}. {title}"

    return None


def detect_scheme_from_filename(filename: str) -> str:
    """Infer farmer scheme name from filename prefix."""
    lower = filename.lower()
    if lower.startswith("pmkisan") or "pm_kisan" in lower or "pm-kisan" in lower:
        return "PM-KISAN"
    elif lower.startswith("pmfby") or "pm_fby" in lower or "fasal_bima" in lower:
        return "PMFBY"
    elif lower.startswith("kcc") or "kisan_credit" in lower or "interest_subvention" in lower:
        return "KCC"
    return "OTHER"


def clean_line(text: str) -> str:
    """Normalize unicode (NFKC to resolve ligatures like 'fi' -> 'fi'), collapse whitespace, and strip artifacts."""
    # 1. Normalize unicode (NFKC replaces ligatures like 'fi'/'fl' with 'fi'/'fl')
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[\uf000-\uf8ff]", "", text)
    # 2. Collapse internal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pages_with_metadata(
    pdf_path: Path,
    allowed_pages: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Extract raw lines and layout positions from PDF pages, filtered by allowed_pages."""
    doc = fitz.open(str(pdf_path))
    pages_data = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        if allowed_pages is not None and page_num not in allowed_pages:
            continue

        page_height = page.rect.height
        blocks = page.get_text("blocks")
        lines_with_pos = []

        for b in blocks:
            if b[6] == 0:  # text block
                block_lines = b[4].split("\n")
                y0, y1 = b[1], b[3]
                for line in block_lines:
                    cleaned = clean_line(line)
                    if cleaned:
                        lines_with_pos.append({
                            "text": cleaned,
                            "y0": y0,
                            "y1": y1,
                            "is_top": y0 < (page_height * 0.12),
                            "is_bottom": y1 > (page_height * 0.88),
                        })

        pages_data.append({
            "page_num": page_num,
            "lines": lines_with_pos,
        })

    doc.close()
    return pages_data


def detect_repeated_headers_footers(pages_data: List[Dict[str, Any]]) -> Tuple[set, set]:
    """Identify recurring header and footer strings across pages."""
    total_pages = len(pages_data)
    if total_pages < 3:
        return set(), set()

    top_counts: Dict[str, int] = {}
    bottom_counts: Dict[str, int] = {}

    for page in pages_data:
        seen_top = set()
        seen_bottom = set()
        for item in page["lines"]:
            text = item["text"]
            if item["is_top"] and not PAGE_NUMBER_PATTERN.match(text):
                if text not in seen_top:
                    top_counts[text] = top_counts.get(text, 0) + 1
                    seen_top.add(text)
            if item["is_bottom"] and not PAGE_NUMBER_PATTERN.match(text):
                if text not in seen_bottom:
                    bottom_counts[text] = bottom_counts.get(text, 0) + 1
                    seen_bottom.add(text)

    threshold = max(2, int(total_pages * 0.25))
    repeated_headers = {t for t, cnt in top_counts.items() if cnt >= threshold}
    repeated_footers = {t for t, cnt in bottom_counts.items() if cnt >= threshold}
    return repeated_headers, repeated_footers


def split_text_into_sentences(text: str) -> List[str]:
    """Segment text into whole grammatical sentences protecting common abbreviations."""
    text = unicodedata.normalize("NFKC", text)
    abbr_map = {
        "Rs.": "Rs__DOT__",
        "i.e.": "ie__DOT__",
        "e.g.": "eg__DOT__",
        "viz.": "viz__DOT__",
        "No.": "No__DOT__",
        "para.": "para__DOT__",
        "Dr.": "Dr__DOT__",
        "Mr.": "Mr__DOT__",
        "Smt.": "Smt__DOT__",
        "Govt.": "Govt__DOT__",
        "Ltd.": "Ltd__DOT__",
        "Pvt.": "Pvt__DOT__",
    }
    temp = text
    for k, v in abbr_map.items():
        temp = temp.replace(k, v)

    raw_segments = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", temp) if s.strip()]
    restored = []
    for s in raw_segments:
        for k, v in abbr_map.items():
            s = s.replace(v, k)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            restored.append(s)
    return restored


def chunk_document(
    pdf_path: Path,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a single PDF into structured, sentence-bounded chunks.

    Args:
        pdf_path: Path to the raw PDF file.
        target_tokens: Maximum hard-cap token size per chunk (400).
        overlap_tokens: Desired token overlap between consecutive chunks (~50).

    Returns:
        Tuple of (list_of_chunks, diagnostic_summary).
    """
    filename = pdf_path.name
    scheme = detect_scheme_from_filename(filename)

    # Determine allowed page range from PAGE_RANGES
    doc = fitz.open(str(pdf_path))
    doc_total_pages = len(doc)
    doc.close()

    ranges = PAGE_RANGES.get(filename)
    if ranges:
        allowed_pages = set()
        for start, end in ranges:
            for p in range(start, end + 1):
                allowed_pages.add(p)
        skipped_pages = doc_total_pages - len(allowed_pages)
    else:
        allowed_pages = set(range(1, doc_total_pages + 1))
        skipped_pages = 0

    print(
        f"  {filename}: Kept {len(allowed_pages)} pages, Skipped {skipped_pages} pages "
        f"(Total: {doc_total_pages})"
    )

    pages_data = extract_pages_with_metadata(pdf_path, allowed_pages=allowed_pages)
    repeated_headers, repeated_footers = detect_repeated_headers_footers(pages_data)

    # Collect clean page lines
    pages_clean_lines: List[Tuple[int, List[str]]] = []
    total_raw_characters = 0

    for page in pages_data:
        p_num = page["page_num"]
        clines = []
        for line_item in page["lines"]:
            text = line_item["text"]
            if PAGE_NUMBER_PATTERN.match(text):
                continue
            if line_item["is_top"] and text in repeated_headers:
                continue
            if line_item["is_bottom"] and text in repeated_footers:
                continue
            clines.append(text)
            total_raw_characters += len(text)
        pages_clean_lines.append((p_num, clines))

    # Calculate total document tokens from kept pages
    full_document_text = "\n".join(["\n".join(lines) for _, lines in pages_clean_lines])
    full_document_text = re.sub(r"\s+", " ", full_document_text).strip()
    total_doc_tokens = count_tokens(full_document_text)

    # Segment document into whole sentences while tracking the active clause
    sentences: List[Dict[str, Any]] = []
    current_clause: Optional[str] = None

    for page_num, lines in pages_clean_lines:
        idx = 0
        while idx < len(lines):
            line = lines[idx]

            # Detect two-line numbered headings e.g. "11." followed by "Validity of the list..."
            if re.match(r"^\d+\.\s*$", line) and idx + 1 < len(lines):
                candidate_heading = f"{line.strip()} {lines[idx + 1].strip()}"
                marker = extract_clause_marker(candidate_heading)
                if marker:
                    current_clause = marker
                    sentences.append({
                        "text": candidate_heading,
                        "page": page_num,
                        "tokens": count_tokens(candidate_heading),
                        "clause": current_clause,
                    })
                    idx += 2
                    continue

            # Check single line for clause marker
            line_marker = extract_clause_marker(line)
            if line_marker:
                current_clause = line_marker

            # Split line into whole sentences (never slicing mid-sentence)
            sents = split_text_into_sentences(line)
            for s in sents:
                s_marker = extract_clause_marker(s)
                if s_marker:
                    current_clause = s_marker

                s_tok_len = count_tokens(s)
                sentences.append({
                    "text": s,
                    "page": page_num,
                    "tokens": s_tok_len,
                    "clause": current_clause,
                })
            idx += 1

    # Chunking:
    # 1. End each chunk at the last full sentence before the 400-token cap.
    # 2. Start the next chunk with the last 1-2 full sentences of the previous one (~50 tokens).
    chunks: List[Dict[str, Any]] = []
    start_idx = 0
    n_sents = len(sentences)
    chunk_index = 1
    prefix_clean = re.sub(r"[^a-zA-Z0-9]", "_", pdf_path.stem.lower())[:24]

    while start_idx < n_sents:
        curr_sents = [sentences[start_idx]["text"]]
        end_idx = start_idx + 1

        while end_idx < n_sents:
            candidate = curr_sents + [sentences[end_idx]["text"]]
            candidate_text = " ".join(candidate)
            if count_tokens(candidate_text) > target_tokens:
                break
            curr_sents = candidate
            end_idx += 1

        chunk_text = " ".join(curr_sents)
        chunk_tok_count = count_tokens(chunk_text)
        chunk_pages = [sentences[i]["page"] for i in range(start_idx, end_idx)]
        chunk_clause = sentences[start_idx]["clause"]
        min_page = min(chunk_pages) if chunk_pages else 1

        chunk_id = f"{prefix_clean}_p{min_page:03d}_c{chunk_index:03d}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "scheme": scheme,
            "source_file": filename,
            "page": min_page,
            "clause_at_start": chunk_clause,
            "tokens": chunk_tok_count,
        })
        chunk_index += 1

        if end_idx >= n_sents:
            break

        # Start the next chunk with the last 1-2 full sentences of the previous chunk (~50 tokens)
        cands = [end_idx - 1]
        if end_idx - 2 > start_idx:
            cands.append(end_idx - 2)

        best_cand = end_idx - 1
        best_diff = 999999
        for c in cands:
            ov_tokens = sum(sentences[k]["tokens"] for k in range(c, end_idx))
            diff = abs(ov_tokens - overlap_tokens)
            if diff < best_diff:
                best_diff = diff
                best_cand = c

        start_idx = best_cand

    # Metrics and validation
    total_chunk_tokens = sum(c["tokens"] for c in chunks)
    dup_ratio = total_chunk_tokens / total_doc_tokens if total_doc_tokens > 0 else 1.0
    chunks_over_400 = sum(1 for c in chunks if c["tokens"] > target_tokens)

    summary = {
        "filename": filename,
        "scheme": scheme,
        "total_pages": doc_total_pages,
        "kept_pages": len(allowed_pages),
        "skipped_pages": skipped_pages,
        "total_doc_tokens": total_doc_tokens,
        "total_chunk_tokens": total_chunk_tokens,
        "total_chunks": len(chunks),
        "dup_ratio": round(dup_ratio, 4),
        "chunks_over_400": chunks_over_400,
        "is_probably_scanned": total_raw_characters < 500,
        "average_tokens": (
            round(total_chunk_tokens / len(chunks), 1) if chunks else 0
        ),
    }

    return chunks, summary


def ingest_all_pdfs(
    raw_dir: Optional[Path] = None,
    output_json: Optional[Path] = None,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """Ingest every PDF in raw directory, perform validation, and save chunks to JSON.

    Args:
        raw_dir: Directory containing raw scheme PDFs.
        output_json: Path to save output chunks.json.
        chunk_tokens: Hard cap tokens per chunk (default: 400).
        overlap_tokens: Desired token overlap (~50).

    Returns:
        Complete list of chunk dictionaries.
    """
    ensure_directories()
    raw_dir = raw_dir or PATHS.raw_pdfs_dir
    output_json = output_json or PATHS.chunks_file

    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"[Warning] No PDF files found in {raw_dir}")
        return []

    all_chunks: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    print("=" * 95)
    print(f"Ingesting {len(pdf_files)} PDF documents from: {raw_dir}")
    print(f"Target Chunk Tokens: {chunk_tokens} (Hard Cap) | Overlap: ~{overlap_tokens}")
    print("=" * 95)

    for pdf_path in pdf_files:
        chunks, summary = chunk_document(
            pdf_path,
            target_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(chunks)
        summaries.append(summary)

    # Save to chunks.json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 105)
    print("INGESTION & DUPLICATION SUMMARY")
    print("=" * 105)
    header = (
        f"{'Filename':<38} | {'Pages (Kept/Skip)':<17} | {'Chunks':<6} | "
        f"{'Doc Tok':<7} | {'Chunk Tok':<9} | {'Dup Ratio':<9} | {'Over 400':<8}"
    )
    print(header)
    print("-" * 105)

    failed_files = []
    for s in summaries:
        page_stat = f"{s['kept_pages']}/{s['skipped_pages']} (of {s['total_pages']})"
        print(
            f"{s['filename']:<38} | {page_stat:<17} | "
            f"{s['total_chunks']:<6} | {s['total_doc_tokens']:<7} | "
            f"{s['total_chunk_tokens']:<9} | {s['dup_ratio']:<9.4f} | {s['chunks_over_400']:<8}"
        )
        # Requirement: Dup ratio between 1.0 and 1.2, 0 chunks over 400 tokens
        if s["dup_ratio"] > 1.20 or s["dup_ratio"] < 1.00:
            failed_files.append(
                f"{s['filename']}: Duplication ratio {s['dup_ratio']:.4f} outside [1.0, 1.2]"
            )
        if s["chunks_over_400"] > 0:
            failed_files.append(
                f"{s['filename']}: {s['chunks_over_400']} chunks exceed {chunk_tokens} token cap"
            )

    print("-" * 105)
    total_tokens_all = sum(c["tokens"] for c in all_chunks)
    avg_len_overall = round(total_tokens_all / len(all_chunks), 1) if all_chunks else 0
    total_over_400 = sum(s["chunks_over_400"] for s in summaries)

    print(f"Total Chunks Generated:      {len(all_chunks)}")
    print(f"Average Chunk Length:        {avg_len_overall} tokens")
    print(f"Total Chunks Over 400 Tok:   {total_over_400} (Target: 0)")
    print(f"Saved chunks to:             {output_json}")

    # Fail loudly if any duplication ratio is higher than 1.2 or chunks over 400
    if failed_files:
        error_msg = "\n[ERROR] Ingestion Quality Check Failed:\n" + "\n".join(
            f"  - {err}" for err in failed_files
        )
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

    print("\nAll quality checks passed: all duplication ratios between 1.0 and 1.2, and 0 chunks over 400 tokens.")
    print("=" * 105)
    return all_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Indian farmer scheme PDFs into structured chunks."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PATHS.raw_pdfs_dir),
        help="Directory with raw PDF files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PATHS.chunks_file),
        help="Output path for chunks.json",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=400,
        help="Target tokens per chunk (default: 400)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between chunks (default: 50)",
    )
    args = parser.parse_args()

    ingest_all_pdfs(
        raw_dir=Path(args.raw_dir),
        output_json=Path(args.output),
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap,
    )


if __name__ == "__main__":
    main()
