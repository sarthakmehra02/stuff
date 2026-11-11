# Independent Codex review — Qwen 7B preview

This is a separate assistant review of q001–q005 for every condition. I
checked each candidate against the verified gold answer in `data/test_qa.json`.
It is not the same as the blind NIM judge and is not independent human
annotation. The remaining 55 questions per condition still use the blind-judge
labels in `judged_*.jsonl`.

| Condition | q001 | q002 | q003 | q004 | q005 | Correct / Partial / Wrong |
|---|---|---|---|---|---|---|
| Zero-shot | wrong | wrong | wrong | partial | wrong | 0 / 1 / 4 |
| Prompted | partial | partial | wrong | wrong | wrong | 0 / 2 / 3 |
| RAG + reranker | correct | correct | correct | wrong | correct | 4 / 0 / 1 |
| Fine-tuned | wrong | wrong | wrong | partial | wrong | 0 / 1 / 4 |
| Fine-tuned + RAG | correct | correct | partial | partial | correct | 2 / 2 / 1 |

Notes:

- RAG correctly retrieved and used the source for q001, q002, q003, and q005.
- q004 is answerable: a refusal is wrong even though the response is cautious.
- Fine-tuned+RAG gives the strongest preview overall, but q003 omits the
  succession exception and q004 omits the former Group D/MTS exception.
- These labels are Codex's independent review of only 25 preview answers; use
  the blind-judge table for the complete 300-answer comparison.
