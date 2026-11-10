# Reranked retrieval and retraining experiment

This phase adds a second-stage cross-encoder reranker without replacing the
original embedding-only baseline.

## Retrieval

1. BAAI/bge-small-en-v1.5 retrieves a candidate pool (20 by default).
2. `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each question/chunk pair.
3. The highest-scoring four chunks are passed to the Qwen generator.

Run the new RAG conditions in Colab after installing `requirements-colab.txt`:

```bash
python src/run_conditions.py --conditions rag,finetuned_rag \
  --adapter models/qlora_adapter --out results/reranked \
  --rerank --candidate-k 20
```

The original `results/raw_rag.jsonl` remains the embedding-only comparison.
Reranked records include `embedding_score` and `rerank_score` for auditing.

## Fine-tuning

The existing generator already creates examples from bounded document chunks,
filters semantic duplicates and test-set similarity, and splits by `chunk_id`.
The training script now defaults to a 1024-token sequence limit, lower learning
rate, gradient accumulation, weight decay, and epoch-level validation. These
settings prevent answer truncation and make the validation curve meaningful.

Rebuild the chunk-disjoint split before training:

```bash
python src/split_train.py --input data/train_qa.jsonl \
  --out-dir data/train_split --val-fraction 0.15
python src/finetune.py --train data/train_split/train.jsonl \
  --val data/train_split/val.jsonl --out models/qlora_reranked_run \
  --epochs 3 --max-seq-length 1024 --learning-rate 1e-4
```

The 60-question test set is not used to create, tune, or select training
examples. Compare the new run with the old run using the same blind judge and
report answerable accuracy, unanswerable hallucination rate, and false-refusal
rate separately.
