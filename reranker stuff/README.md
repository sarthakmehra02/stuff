# Complete reranker notebooks

Run one notebook at a time on a Colab T4:

- `run_all_3b_reranked.ipynb` uses Qwen2.5-3B-Instruct.
- `run_all_7b_reranked.ipynb` uses Qwen2.5-7B-Instruct.

Each notebook runs zero-shot, prompted, RAG, fine-tuned, and fine-tuned+RAG.
Outputs are isolated in `results/reranker_3b_all` or `results/reranker_7b_all`
and copied to matching Google Drive folders. Review the five-item previews when
they appear before allowing the following full-evaluation cell to run.
