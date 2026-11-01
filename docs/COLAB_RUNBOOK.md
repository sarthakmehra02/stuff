# Colab runbook

This runbook keeps all NVIDIA NIM work on the local Windows machine. Colab receives only code, chunks, synthetic train data, the fixed test set, and model outputs. The code repository is public, so the notebooks clone it without a GitHub token. **Never add `NVIDIA_API_KEY` to Colab Secrets.**

## Before opening Colab

1. Install Python 3.12 locally, create a `.venv`, then install `requirements.txt` as described in `README.md`.
2. Put the NVIDIA key in local `.env`. Verify teacher/judge connectivity with `python src/test_nim.py`.
3. Run `python src/validate_test_set.py`, `python src/prompts.py`, and `python src/eval_retrieval.py`. The test set must remain unchanged; retrieval should reproduce hit@1 28/51 and hit@4 48/51.
4. Generate five training chunks with `python src/make_train_data.py --limit 5 --out data/train_qa_preview.jsonl`. Inspect the examples before generating all chunks.
5. Generate the full synthetic dataset, create `results/train_sample_for_manual_check.csv`, complete your human review, then create the chunk-level train/validation split with `python src/split_train.py`.
6. The notebooks clone `https://github.com/sarthakmehra02/stuff.git` directly. Keep raw PDFs, `.env`, NIM cache, model checkpoints, adapters, and FAISS index files out of Git.
7. Create the Google Drive folder `MyDrive/farmer-llm/`. Training checkpoints and outputs are saved there so a Colab disconnect does not erase them.

## Set up Colab

1. Open Google Colab and choose **Runtime → Change runtime type → T4 GPU** (if available).
2. Open one of `notebooks/01_baselines.ipynb`, `02_finetune.ipynb`, `03_eval_finetuned.ipynb`, or `04_demo.ipynb` from the public repository.
3. The setup cell clones the project from its public URL; no GitHub secret is required.
4. Run the first GPU-check cell. It prints `nvidia-smi` and stops if CUDA is unavailable.
5. Run the Drive and repository setup cell to clone or update the public project.
6. Run the dependency cell. `requirements-colab.txt` deliberately omits torch so Colab's CUDA build is not replaced. The cell prints installed library versions for the report.

## Run in order

### Notebook 01 — baseline conditions

1. Run the index cell to rebuild FAISS from `data/processed/chunks.json`.
2. Run the five-question cell for `zero_shot`, `prompted`, and `rag`.
3. Review five answers per condition and the recorded metadata. The prompt fingerprint is written to `results/run_metadata.json`.
4. If the sample looks valid, run the full-set cell. It writes one line per question to `results/raw_<condition>.jsonl` and copies the results plus metadata to Drive.

### Notebook 02 — fine-tuning

1. Confirm the train/validation JSONL files exist in the repository.
2. Run the 20-step `--smoke` cell. It saves to `MyDrive/farmer-llm/run1_smoke/`, separate from full-training checkpoints. Review the printed decoded masking preview: system and question tokens should be masked, while the assistant answer should be supervised.
3. If the smoke run finishes, run full training. The adapter, checkpoints, `training_summary.json`, and `loss_curve.png` are written under `MyDrive/farmer-llm/run1/`; the smoke checkpoint will not be resumed by the full run.
4. After a disconnect, reconnect Drive, re-run setup, and re-run the training command. The script resumes from the newest checkpoint it finds.

### Notebook 03 — fine-tuned conditions

1. Check that `MyDrive/farmer-llm/run1/adapter/` exists.
2. Run the five-question preview for `finetuned` and `finetuned_rag`; review the answers.
3. Run all 60 questions. Copy the two `raw_finetuned*.jsonl` files and the updated `run_metadata.json` from Drive to local `results/`.
4. Confirm the five local `raw_*.jsonl` files each have 60 rows before judging.

### Notebook 04 — demo

1. Run the GPU and repository setup cells.
2. Confirm the adapter path under Drive.
3. Run the final cell. Gradio prints the temporary public URL. Keep the runtime open while demonstrating; the URL stops working when the runtime disconnects.

## Local judge and analysis

1. Download all five raw output files and `run_metadata.json` from Drive into the local `results/` directory.
2. Run the judge with `--limit 5` first and inspect the verdicts. A full run makes approximately 420 NIM calls (300 answer grades plus up to 120 RAG-faithfulness grades). At 38 requests/minute, the rate-limit floor is about 11 minutes; actual time depends on reasoning-model latency.
3. Run the full judge only after the five-item sample is acceptable.
4. Create `results/manual_check_sample.csv`, fill `my_verdict`, then run `python src/manual_check.py --score`.
5. Run `python src/analyze_results.py` to produce summary CSV files, paired McNemar tests, retrieval/correctness tables, error analysis, and PNG plots.

## Free-tier and recovery notes

- GPU availability and session length vary. Save training checkpoints and all outputs on Drive.
- If a session disconnects, reconnect Drive, rerun the setup cells, and use the same output paths with `--resume` for evaluation or the same training `--out` directory for checkpoint recovery.
- Do not change prompts or retrieval after seeing test answers. `prompt_fingerprint` must match throughout the full benchmark.
- If a run fails, keep its JSONL file and rerun with `--resume`; completed IDs are skipped.
- The demo and adapter upload are separate from evaluation. Adapter upload is opt-in and requires a local `HF_TOKEN`; the script defaults to a private model repository.

## What to bring back

- After Notebook 01: five answers per condition, `run_metadata.json`, peak GPU memory, and seconds per answer.
- After Notebook 02: `training_summary.json`, `loss_curve.png`, peak memory, and total training time.
- After Notebook 03: all five `raw_*.jsonl` files copied into local `results/`.
- After the manual judge check: the completed `manual_check_sample.csv` and `judge_agreement.json`.
