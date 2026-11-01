# Report facts and experiment record

Use this file as the source of factual numbers in the report. Unknown experiment outputs stay marked `TODO(user)` until the corresponding local or Colab run is completed. Do not infer missing scores.

## Research setup

- Research question: compare prompting, retrieval-augmented generation, fine-tuning, and fine-tuning plus RAG on a small unseen-document QA task.
- Sole evaluated generator: `Qwen/Qwen2.5-3B-Instruct`.
- Generator precision: NF4 4-bit, double quantization, fp16 compute (T4); greedy decoding (`do_sample=False`); maximum 300 new tokens; batch size 1; random seed 42.
- Conditions: `zero_shot`, `prompted`, `rag`, `finetuned`, `finetuned_rag`.
- Teacher: `nvidia/nemotron-3.5-super-vl-120b-a12b` (NIM; synthetic QA only; no-thinking mode is requested for JSON-only output).
- Judge: `nvidia/nemotron-3-ultra-550b-a55b` (NIM; blind answer grading and RAG faithfulness only).
- Prompt fingerprint for the current implementation: `da2331af8b29cd7eb7355586a30dd46ad6a16a5e72764c1b7335f815a38db9d5`.
- Git commit hash for the baseline run: `32359aa9177be7fafd6439047e263fce83d6684e`.

## Source corpus

| Source file | Scheme | Pages in PDF | Pages retained | Document date / scope |
| --- | --- | ---: | ---: | --- |
| `pmkisan_operational_guidelines_2020.pdf` | PM-KISAN | 12 | 12 | 2020-03-29 revised operational guidelines |
| `kcc_rbi_master_circular_2018.pdf` | KCC | 15 | 15 | RBI master circular 2018-19 |
| `kcc_interest_subvention_2025_26.pdf` | KCC | 5 | 5 | Interest subvention for FY 2025-26; circular dated 2026-01-13 per `sources.md` |
| `pmfby_operational_guidelines_2023.pdf` | PMFBY | 213 | 86 | Revised guidelines effective from Kharif 2023 |

PMFBY retained pages use printed/PDF page ranges 11, 14–23, 47–69, 88–126, 134–138, 168–173, and 211–212. Total retained: 86; skipped: 127. The 2019 PM-KISAN scheme overview is an old conflicting document and is excluded from the indexed corpus.

## Chunking and retrieval

- Chunk count: 180 total — KCC 24, PMFBY 141, PM-KISAN 15.
- Chunk token lengths from `data/processed/chunks.json`: minimum 87, mean 390.16, maximum 400; no listed chunk exceeds 400.
- Target chunking uses sentence boundaries and about 50-token overlap. Actual duplication ratio: `TODO(user): recompute from source PDFs`.
- Embeddings: `BAAI/bge-small-en-v1.5`; normalized vectors and FAISS inner product for cosine search; top-k = 4.
- Manifest SHA-256 equals current `chunks.json`: `96799e9c79d6820d7f4d574449b9e13142946583f6f34896a7be320fd70d94a7`.
- Retrieval hit criterion: expected source file and a retrieved page within ±1 of any gold page.
- Overall: hit@1 28/51 (54.90%); hit@4 48/51 (94.12%). Three hit@4 misses: q042, q044, q049.
- By scheme (answerable questions): KCC 7/12 hit@1, 12/12 hit@4; PM-KISAN 7/13, 12/13; PMFBY 14/26, 24/26.
- By question type: cross-scheme 1/3 hit@1 and 1/3 hit@4; direct 17/30 and 30/30; Hinglish 2/5 and 5/5; multi-step 6/8 and 8/8; paraphrase 2/5 and 4/5.
- Retrieval CSV: `results/retrieval_eval.csv`; computation rerun locally with `python src/eval_retrieval.py`.

## Fixed evaluation set

- File: `data/test_qa.json` (read-only).
- Total 60: 51 answerable, 9 unanswerable.
- By type: direct 30; multi-step 8; cross-scheme 3; paraphrase 5; Hinglish 5; unanswerable 9.
- By scheme: KCC 15; PM-KISAN 15; PMFBY 30.
- Validator result: pass; no missing required fields, duplicate IDs/questions, invalid types, or invalid answerable/source-page combinations.
- Gold answer review against source PDFs: `TODO(user): complete manual audit when PDFs are available in the local corpus`.
- Deliberate conflict question: q043 compares a 2018 KCC circular's mandatory-insurance wording with the 2023 PMFBY opt-out rule; the newer 2023 text governs the experiment.

## Curated few-shot examples

- Four examples in `data/fewshot.json`; facts checked in `chunks.json`.
- Maximum BGE cosine similarity to any test question: NPCI card design 0.7860; PMU CEO 0.7261; PMFBY loss-assessor qualification 0.7307; monthly-rainfall refusal 0.7422. All are below 0.85.
- Verification command: `python src/prompts.py` (requires BGE model weights in the local Hugging Face cache or network access).

## Training data and fine-tuning

- Five-chunk teacher preview: 5 chunks, 2 empty, 11 raw items; 2 retained after filters (one direct, one Hinglish), 2 evidence-quote drops, 4 test-similarity drops, 1 exact duplicate, 2 non-English-answer drops.
- Full teacher generation: 180/180 chunks processed; 2 returned empty; 616 parsed raw items; 507 retained. The 109 candidate-item drops were: 2 missing fields, 10 non-verbatim evidence quotes, 45 test-similarity >0.85, 3 exact duplicates, 12 near duplicates >0.95, 28 non-English answers, and 9 wrong-scheme questions. Five chunks had JSON parse errors (separately counted at chunk level). Other item-level drop categories were zero.
- Training pairs by scheme: KCC 51, PM-KISAN 24, PMFBY 432. By style: direct 213, Hinglish 149, paraphrase 145. No refusal examples were requested.
- Manual review of the 50-pair sample: 1 bad item (2%). The bad item asked for a deadline in days while its evidence stated 48 hours. The automated candidate rejection share (109/616 = 17.7%) is not the human bad-example rate.
- Chunk-disjoint split (seed 42): 482 train examples from 162 chunks; 25 validation examples from 8 chunks; no chunk overlap.
- Human bad-example rate from the 50-pair review: `TODO(user)`.
- Train/validation counts after chunk-level 95/5 split: `TODO(user)`.
- QLoRA parameters: rank 16, alpha 32, dropout 0.05; target attention and MLP projections; max length 384; batch 4; gradient accumulation 4; learning rate 2e-4; cosine schedule; warmup 0.03; 3 epochs; fp16 true and bf16 false; seed 42.
- Adapter, steps, training time, final train/validation loss, peak memory, overfitting signal, and Colab library versions: `TODO(user)` from `training_summary.json`.

## Answer evaluation and analysis

- Baseline raw rows: 60 each for `zero_shot`, `prompted`, and `rag` (180 total), all IDs unique. Raw files and `run_metadata.json` are in `results/`. Fine-tuned raw rows and all judged rows remain pending.
- Strict/lenient accuracy and 95% bootstrap intervals: `TODO(user)` in `results/summary_overall.csv`, `summary_by_type.csv`, `summary_by_scheme.csv`.
- Baseline mean answer latency from the raw rows: zero-shot 13.92 s, prompted 1.35 s, RAG 4.78 s. Peak GPU memory allocated: 2942 MiB on Tesla T4. P95 latency, hallucination/false-refusal rates, RAG faithfulness, and mean generated tokens: pending blind judging/analysis.
- Pairwise exact two-sided McNemar tests and discordant counts: `TODO(user)` in `results/paired_tests.csv` (n=51 is small).
- Retrieval-hit × correctness tables: `TODO(user)` in `results/retrieval_vs_correctness.csv`.
- Judge manual agreement and Cohen's kappa: `TODO(user)` in `results/judge_agreement.json`.
- Error cases and wrong-scheme citations: `TODO(user)` in `results/error_analysis.md`.
- Plots: `TODO(user)` in `results/plots/` after judged outputs are present.

## Environment versions

- Python available for local checks: 3.12.2.
- Isolated temporary verification environment (not a Colab record): pytest 9.2.0, torch 2.15.0, transformers 5.22.1, sentence-transformers 6.2.0, faiss-cpu 1.15.1, NumPy 2.5.4, pandas 3.1.0, matplotlib 3.11.2, OpenAI SDK 3.42.0.
- Actual Colab baseline versions: torch 2.11.0+cu130, transformers 4.57.6, peft 0.21.2, trl 0.29.1, bitsandbytes 0.50.2, sentence-transformers 3.4.1, faiss-cpu 1.15.1.
- NVIDIA NIM smoke: **PASS** — teacher and judge each returned clean one-word and JSON responses. The original teacher model ID returned 404, so the teacher endpoint was updated to the verified Nemotron 3.5 Super VL endpoint. The five-chunk preview then completed with JSON parsing enabled and no-thinking request option.

## Limitations

- Documents date from 2018–2026, so gold answers mean “what these documents state,” not current law.
- The test set has only 51 answerable and 9 unanswerable questions; results are indicative.
- Retrieval grading tolerates a one-page offset; some gold pages were corrected after reviewing retrieval output.
- Cross-scheme items record one primary source only, understating retrieval quality.
- Teacher and judge are Nemotron-family models; style bias is possible and should be checked with the human 20-row sample.
- Synthetic training data receives evidence-quote and automatic filter checks plus a 50-pair manual review; it is not fully human verified.
- The fine-tuned model learns document facts by design; leakage is controlled at the question level.
- Fine-tuning examples have no retrieved context; `finetuned_rag` therefore sees a prompt distribution that differs from training.
- PMFBY is 141/180 (78.3%) of indexed chunks.
- Chunk edges may contain sentence fragments.
- No refusal training is used unless the optional Phase 2 flag is enabled.
