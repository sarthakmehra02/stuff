# Qwen 3B reranker result

The five raw files contain 60 unique questions each. Blind answer grading is
being run with `nvidia/nemotron-3-ultra-550b-a55b` against the verified gold
set. Run `track_judging.py` to monitor progress. Final Correct/Partial/Wrong
counts will be added after all five judged files reach 60 rows.
