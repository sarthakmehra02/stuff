# Phase 0 Audit

Date: 2026-11-01 (Asia/Calcutta)

## Initial repository state

- Python 3.12.2 was available locally; the Windows Store Python alias was unusable, so an isolated temporary environment was used for checks.
- The working tree initially had no local commits; project files were untracked. The `origin` remote points to the user's `stuff` repository.
- The project data directory was stored with uppercase `Data`. It has been normalized to lowercase `data` so the Linux Colab notebook paths work after cloning. The data contents were preserved.
- `.env` is ignored; its key value was never displayed. Local NIM smoke passed for the updated teacher and judge endpoints.

## Phase 0 acceptance checks

- `python src/validate_test_set.py`: **PASS** — 60 records, 51 answerable and 9 unanswerable; no validation errors.
- `python src/eval_retrieval.py` in the isolated dependency environment: **PASS** — hit@1 28/51 (54.90%), hit@4 48/51 (94.12%). The three top-4 misses are q042, q044, and q049.
- `data/processed/chunks.json`: **180 chunks** — KCC 24, PMFBY 141, PM-KISAN 15.
- `data/processed/faiss_manifest.json`: **MATCH** — stored SHA-256 `96799e9c79d6820d7f4d574449b9e13142946583f6f34896a7be320fd70d94a7` equals the current chunks file hash.
- `.gitignore` excludes `.env`, raw PDF/old-version folders, cache and SQLite files, model weight formats, FAISS `.bin` index, virtual environments, and the local `human.md` work log.
- The four source PDFs listed in the prompt are present in `data/raw_pdfs/`; old/conflicting documents are in `data/old_versions/`.
- `data/fewshot.json` initially contained examples that did not match the prescribed fact set. It was replaced in Phase 1 and validated against `chunks.json` and the fixed test set.
- Phase 0's required scripts, ingestion, RAG prompt, NIM client, and test set exist. The later-phase training, condition runner, judging, demo, notebooks, and report components were initially missing or incomplete and have since been implemented in this work session.

## Follow-up verification

- The test set remains byte-for-byte unchanged.
- BGE cosine similarity from each new few-shot question to its nearest test question: 0.7860, 0.7261, 0.7307, and 0.7422; all below 0.85.
- Initial unit suite after implementation: 12 passed; additional tests were subsequently added, so rerun `pytest` for the current total.
- Retrieval computation used cached `BAAI/bge-small-en-v1.5` weights in offline mode after network access to the Hugging Face Hub was denied. The indexed vectors were already present and their manifest matched the current chunks.
- Package versions used for unit checks are recorded in `docs/REPORT_FACTS.md`. These are not Colab runtime versions.
- The configured original teacher endpoint returned 404. It was replaced by `nvidia/nemotron-3.5-super-vl-120b-a12b`; the official endpoint passed the smoke check. Teacher calls set `reasoning_effort=none` for JSON-only output while still reading `message.content` only.
- The five-chunk preview retained two examples after filtering. Full generation is pending review of the preview.

## Remaining Phase 0/experiment items

- PDF-by-PDF gold-answer review is still needed; the test set is read-only.
- There is no pre-existing commit history to compare against. Create the first local project commit after reviewing the complete implementation; do not push.

## Phase 0 result

The numeric corpus, test-set, retrieval, and index-manifest acceptance checks pass. Work continued through the remaining phases at the user's direction. External Colab runs and human-review gates are still pending; their outputs are marked `TODO(user)` in `docs/REPORT_FACTS.md`.

## Final local checks

- `pytest -q tests src`: **PASS**, 16 tests.
- `compileall` for config, source, and scripts: **PASS**.
- All five notebooks parse as valid JSON; test-set validation and few-shot prompt validation pass.
- Rebuilt update-test FAISS index: 180 vectors. Synthetic set has 3 labeled changes, 10 questions marked synthetic, and a notice file.
- Retrieval recomputed: hit@1 28/51 (54.90%), hit@4 48/51 (94.12%).
- The initial commit could not be created: Git could not create `.git/index.lock` because `.git` is read-only in this workspace. Nothing was staged or committed; no push was attempted.

## Phase 2 — approved full teacher run

The user approved full generation after reviewing the preview. The teacher processed 180/180 chunks; 2 returned empty, and 5 had JSON parse errors. Of 616 parsed candidates, 507 passed. Candidate item drops (109; 17.7%) were 2 missing fields, 10 unsupported evidence quotes, 45 test-similarity failures, 3 exact duplicates, 12 near duplicates, 28 non-English answers, and 9 wrong-scheme questions. The candidate rejection share is distinct from the human bad-example rate.

Final dataset: KCC 51, PM-KISAN 24, PMFBY 432; direct 213, Hinglish 149, paraphrase 145; no refusal examples. Added the required `scheme` JSONL field and regenerated from cache. The 50-pair review CSV is ready at `results/train_sample_for_manual_check.csv`; the split is waiting for the user's grades and bad-rate report, as required by the master prompt.

## Phase 2 review and split

Reviewed 50 pairs against their evidence and source chunks. One item (2%) was bad: its question asked for a deadline in days while the evidence said 48 hours. This is below the prompt's ~15% threshold. The sample CSV now contains good/bad grades and a note. Created the chunk-level seed-42 split: 482 training examples (162 chunks), 25 validation examples (8 chunks), with zero chunk overlap.

Current checkpoint: the full local suite passes (16 tests). Phase 2 is complete. The next required handoff is Phase 3/User Action 1: run the Colab baseline preview, review five answers per condition, then confirm before the full 60-question baseline run.

## Public GitHub handoff

- The user requested a public repository. Commit `a00c78f` (`Complete farmer schemes RAG project`) was pushed to `main` at `https://github.com/sarthakmehra02/stuff`.
- Repository visibility was changed to public and confirmed with an unauthenticated GitHub API read. The public notebook file is readable without authentication.
- `.env`, raw PDFs, old versions, model weights, caches, FAISS artifacts, test backups, and local `human.md` were excluded. Notebook clone cells use the public URL and no GitHub token.

## Phase 3 baseline run received

- Imported the user's Colab archive into local `results/`: 60 unique rows each for zero-shot, prompted, and RAG, plus `run_metadata.json`. Metadata fingerprint matches the frozen local prompt fingerprint and commit `32359aa`.
- Colab model/runtime record: Qwen2.5-3B-Instruct, NF4 4-bit/double quantization/fp16, seed 42, Tesla T4, 2942 MiB peak allocated. Mean latencies: zero-shot 13.92 s, prompted 1.35 s, RAG 4.78 s. The notebook cell reports 20m12s for the resumed full-set run.
- Preliminary answer inspection found an off-topic RAG answer (q030) and a RAG citation containing a clause number (q025), which conflicts with the project citation rule. Preserve this run unchanged; include these in later error analysis.
- Baseline generation is complete. Blind judging waits until the fine-tuned condition outputs are also available.
