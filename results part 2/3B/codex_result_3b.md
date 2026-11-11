# Qwen 3B reranker result

The five raw files contain 60 unique questions each. Blind answer grading used
`nvidia/nemotron-3-ultra-550b-a55b` against the verified gold set. All five
judged files contain 60 rows.

| Condition | Correct | Partial | Wrong | Refused | Hallucinated | Exact overall |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 1 | 13 | 37 | 7 | 2 | 8/60 (13.3%) |
| Prompted | 4 | 11 | 36 | 9 | 0 | 13/60 (21.7%) |
| RAG + reranker | 12 | 2 | 37 | 9 | 0 | 21/60 (35.0%) |
| Fine-tuned | 6 | 15 | 30 | 0 | 9 | 6/60 (10.0%) |
| Fine-tuned + RAG | 31 | 11 | 9 | 2 | 7 | 33/60 (55.0%) |

On the 51 answerable questions, exact accuracy was 2.0%, 7.8%, 23.5%, 11.8%,
and 60.8%, respectively. Correct-or-partial answerable rates were 27.5%,
29.4%, 27.5%, 41.2%, and 82.4%.

The reranked retriever hit the expected source/page for 43/51 answerable
questions (84.3%) in both RAG conditions. The hybrid had the best answerable
accuracy but hallucinated on 7/9 unanswerable questions; plain RAG refused all
9. Faithfulness was not run because the judge API intermittently returned empty
faithfulness messages.

Use `track_judging.py` to verify the 60/60 files. These are blind AI-judge
labels, not independent human annotation.

See [codex_review_3b.md](codex_review_3b.md) for the separate independent
Codex review of the 25 preview answers.
