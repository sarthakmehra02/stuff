# Independent Codex review — Qwen 3B preview

I independently checked q001–q005 for each 3B condition against the verified
gold answers. This is separate from the blind NIM judge and is not human
annotation; the remaining 55 questions use the blind-judge files.

| Condition | q001 | q002 | q003 | q004 | q005 | Correct / Partial / Wrong |
|---|---|---|---|---|---|---|
| Zero-shot | wrong | wrong | wrong | partial | wrong | 0 / 1 / 4 |
| Prompted | wrong | wrong | wrong | wrong | wrong | 0 / 0 / 5 |
| RAG + reranker | correct | correct | correct | wrong | wrong | 3 / 0 / 2 |
| Fine-tuned | wrong | wrong | wrong | partial | wrong | 0 / 1 / 4 |
| Fine-tuned + RAG | correct | correct | partial | partial | correct | 3 / 2 / 0 |

The hybrid preview is strongest, but its q003 and q004 answers omit important
exceptions. Plain RAG refuses q004 and q005 even though those preview questions
are answerable, so those refusals are wrong under the verified key.
