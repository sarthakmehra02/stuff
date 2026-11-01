# Fine-tuning vs RAG vs Prompting for Indian Farmer Schemes

A reproducible term-project benchmark comparing five ways to answer questions about official PM-KISAN, PMFBY, and Kisan Credit Card (KCC) documents. The evaluated model is always `Qwen/Qwen2.5-3B-Instruct`; the NIM teacher creates synthetic training examples and a separate, larger NIM judge grades answers.

Public repository: [github.com/sarthakmehra02/stuff](https://github.com/sarthakmehra02/stuff). The Colab notebooks clone it without a GitHub token.

## Research conditions

| Condition | LoRA adapter | Retrieved context | Prompt |
| --- | --- | --- | --- |
| `zero_shot` | No | No | Qwen default chat template |
| `prompted` | No | No | `SYSTEM_BASE` and four curated examples |
| `rag` | No | Top 4 chunks | Existing grounded prompt in `src/rag.py` |
| `finetuned` | Yes | No | `SYSTEM_BASE` |
| `finetuned_rag` | Yes | Top 4 chunks | Existing grounded prompt in `src/rag.py` |

All conditions use the same 4-bit NF4, double-quantized base model, fp16 compute, greedy decoding, seed 42, and 300-token output limit. The adapter is loaded once for a run; base conditions disable it.

## Repository map

- `config.py` — shared paths, model IDs, prompt constants, seed, and experiment settings.
- `data/processed/chunks.json` — 180 document chunks; FAISS files are rebuildable and ignored.
- `data/test_qa.json` — fixed 60-question test set; treat it as read-only.
- `data/fewshot.json` — four examples checked against source chunks and test-question similarity.
- `src/` — ingestion, RAG, teacher-data generation, QLoRA, condition runner, blind judge, analysis, and demo.
- `notebooks/01_baselines.ipynb` through `04_demo.ipynb` — Colab GPU workflows.
- `docs/COLAB_RUNBOOK.md` — click-by-click setup and hand-off instructions.
- `docs/REPORT_FACTS.md`, `docs/VIVA_NOTES.md` — report and viva preparation.
- `human.md` — local phase log. It is intentionally excluded from Git by `.gitignore`.

## Local Windows setup

Use Python 3.12 and PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put `NVIDIA_API_KEY` in `.env` for local teacher and judge calls. Never put this key in Colab. `.env`, raw PDFs, caches, model weights, and FAISS index files are ignored.

Validate the fixed test data, curated examples, and current retrieval baseline:

```powershell
python src/validate_test_set.py
python src/prompts.py
python src/eval_retrieval.py
```

Expected fixed-data checks: 60 questions (51 answerable and 9 unanswerable), 180 chunks, hit@1 28/51 and hit@4 48/51. Retrieval is frozen; do not tune it against test results.

## Reproduce the experiment

### 1. Generate training examples locally

The NIM teacher runs locally. Start with five chunks and inspect its output:

```powershell
python src/make_train_data.py --limit 5 --out data/train_qa_preview.jsonl
```

After approving the sample, generate all 180 chunks, create a 50-pair review CSV, and split by source chunk:

```powershell
python src/make_train_data.py --out data/train_qa.jsonl
python src/sample_train_pairs.py
python src/split_train.py
```

Training examples are filtered by evidence quotes, BGE cosine similarity to all test questions (`<= 0.85`), duplicate similarity (`<= 0.95`), answer length, and forbidden question wording. Refusal examples are off unless `--refusal-examples N` is supplied.

Review `results/train_sample_for_manual_check.csv` before fine-tuning. Record the human bad-example rate in `human.md` and `docs/REPORT_FACTS.md`.

### 2. Run baseline conditions in Colab

Follow `docs/COLAB_RUNBOOK.md`, then open `notebooks/01_baselines.ipynb`. The notebook runs five examples first; inspect them before running all 60. Raw outputs and `run_metadata.json` are copied to Drive.

### 3. Fine-tune in Colab

After generating and splitting training data, run `notebooks/02_finetune.ipynb`. It first performs a 20-step smoke run, then trains for three epochs. The adapter, checkpoints, training summary, and loss plot are written to Drive.

### 4. Evaluate the adapter in Colab

Run `notebooks/03_eval_finetuned.ipynb`. Inspect the five-item preview, then evaluate all 60 questions. Download all five `raw_*.jsonl` files and `run_metadata.json` from Drive into local `results/`.

### 5. Judge and analyze locally

The judge sees only the question, gold answer, and candidate answer. It never receives the condition label. Run a five-item smoke evaluation first, inspect the verdicts, then run the complete evaluation:

```powershell
python src/judge.py --conditions zero_shot,prompted,rag,finetuned,finetuned_rag --limit 5
python src/judge.py --conditions zero_shot,prompted,rag,finetuned,finetuned_rag
python src/manual_check.py
```

Fill `my_verdict` in `results/manual_check_sample.csv`, then score agreement and generate summaries:

```powershell
python src/manual_check.py --score
python src/analyze_results.py
```

### 6. Optional synthetic update experiment

`python src/update_test.py` creates a separate index and ten questions. Three facts are edited in copied chunks and clearly marked synthetic; no original data is changed. Run inference with the adapter in Colab using `src/run_update_test.py`, then judge locally using `src/judge_update.py`. These are fabricated edits, not real policy changes.

### 7. Demo and adapter upload

Run `notebooks/04_demo.ipynb` on a Colab T4 to launch the five-panel Gradio demo. To publish an adapter, set `HF_TOKEN` in the local environment and run `scripts/upload_adapter.py` with your repository ID. Upload is opt-in; the script defaults to a private Hugging Face repository.

## Current verified facts

- Corpus index: 180 chunks (KCC 24, PMFBY 141, PM-KISAN 15).
- Test set: 60 questions; 51 answerable and 9 unanswerable.
- Retrieval: hit@1 28/51 (54.90%); hit@4 48/51 (94.12%). The three hit@4 misses are q042, q044, and q049.
- The retrieval manifest SHA-256 matches `data/processed/chunks.json`.
- These are retrieval and data-validation facts only. No model answer comparison or judge score is claimed until the Colab and NIM outputs exist.

## Limitations

- Source documents date from 2018–2026; gold answers describe those documents and may not reflect current law.
- The test set is small (51 answerable and 9 unanswerable questions), so results are indicative.
- Retrieval grading allows a one-page tolerance, and some gold pages were corrected after reviewing retrieval output.
- Cross-scheme questions list only one primary source, which understates retrieval performance.
- Teacher and judge are both Nemotron-family models, so style bias is possible; the 20-row manual judge check helps assess it.
- Synthetic training data relies on evidence-quote, similarity, and automatic form checks; a 50-pair human review is required.
- The fine-tuned model sees document facts by design; leakage control is at the question level.
- Fine-tuning has no retrieved context, so `finetuned_rag` uses a prompt distribution not seen during training.
- PMFBY accounts for 141/180 chunks (78.3%). Chunking can leave sentence fragments at boundaries.
- There is no refusal training unless the optional Phase 2 refusal-example flag is used.
