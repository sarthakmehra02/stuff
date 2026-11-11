# Qwen 7B reranker results

These files were extracted from the Colab run archive received on 2026-11-11.
All five conditions contain 60 unique question IDs and use the same run
metadata, Qwen2.5-7B-Instruct 4-bit model, seed 42, and reranked retrieval
configuration. The adapter checkpoints are intentionally not copied here;
they remain in the original archive/Google Drive because they are large.

| File | Condition |
|---|---|
| `raw_zero_shot.jsonl` | Zero-shot |
| `raw_prompted.jsonl` | Prompted |
| `raw_rag.jsonl` | RAG + reranker |
| `raw_finetuned.jsonl` | Fine-tuned |
| `raw_finetuned_rag.jsonl` | Fine-tuned + RAG + reranker |

`run_metadata.json`, `training_summary.json`, and `loss_curve.png` document the
run and training provenance.
