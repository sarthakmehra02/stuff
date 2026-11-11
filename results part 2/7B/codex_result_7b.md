# Codex 7B grading result

This file is the checked-result handoff for the Qwen2.5-7B-Instruct reranker
run. The raw answers are complete (60 rows per condition). Blind judging uses
`nvidia/nemotron-3-ultra-550b-a55b` against the verified 60-question gold set.

## Final blind-judge result

All five judged files contain 60 unique question IDs. The judge model was
`nvidia/nemotron-3-ultra-550b-a55b`. There are 51 answerable and 9 unanswerable
questions. “Exact overall” counts a correct answer plus a correct refusal.

| Condition | Judged rows | Correct | Partial | Wrong | Refused | Hallucinated | Exact overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | 60 | 4 | 20 | 27 | 1 | 8 | 5/60 (8.3%) |
| prompted | 60 | 3 | 13 | 35 | 8 | 1 | 11/60 (18.3%) |
| RAG + reranker | 60 | 25 | 4 | 22 | 9 | 0 | 34/60 (56.7%) |
| fine-tuned | 60 | 5 | 19 | 27 | 0 | 9 | 5/60 (8.3%) |
| fine-tuned + RAG + reranker | 60 | 31 | 12 | 8 | 1 | 8 | 32/60 (53.3%) |

On answerable questions, exact accuracy was 4/51 (7.8%) zero-shot, 3/51
(5.9%) prompted, 25/51 (49.0%) RAG, 5/51 (9.8%) fine-tuned, and 31/51
(60.8%) fine-tuned + RAG. Correct-or-partial answerable rates were 47.1%,
31.4%, 56.9%, 47.1%, and 84.3%, respectively.

The reranked retriever hit the expected source/page for 43/51 answerable
questions (84.3%) in both RAG conditions. This is lower than the earlier
embedding-only top-4 retrieval result, so reranking should be treated as an
experimental result rather than an automatic improvement.

The labels are blind-judge labels, not independent human grades. Answer
grading was completed without faithfulness calls because the NIM service
intermittently returned empty faithfulness messages. The `judged_*.jsonl`
files contain the per-question verdicts; `track_judging.py` remains available
for progress checks on future reruns.
