# Qwen 3B reranker results

The five raw condition files contain 60 unique questions each and use
`Qwen/Qwen2.5-3B-Instruct` with the reranked retriever. The answer judge began
writing resumable files but the NVIDIA judge API returned HTTP 429 rate-limit
errors before all 300 rows were graded. Run `track_judging.py` to inspect the
partial progress and rerun the judge later; do not report partial counts as
final accuracy.
