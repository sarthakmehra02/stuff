# Viva notes — 25 likely questions

Keep the answers tied to the code and measured outputs. Do not claim model performance until the five full raw and judged files exist.

1. **What question does the project answer?** It compares prompting, RAG, LoRA fine-tuning, and their combination for a small model answering questions grounded in three farmer-scheme documents. The condition matrix is in `README.md`.
2. **Why Qwen2.5-3B-Instruct?** It is the sole model under test, so differences come from the adaptation method rather than changing the base model. See `config.MODEL_ID`.
3. **Why use LoRA?** LoRA trains a small set of low-rank adapter weights while freezing the base model, reducing memory and training cost. See `src/finetune.py` and `LoraConfig`.
4. **What does the LoRA rank `r=16` control?** It sets the rank of the trainable update matrices; higher rank allows a larger update but uses more adapter parameters. See the `LoraConfig` in `src/finetune.py`.
5. **What does alpha 32 do?** It scales the LoRA update relative to its rank; the experiment fixes alpha at 32. See `src/finetune.py`.
6. **Why 4-bit NF4?** It lowers GPU memory use for a 3B model on a free Colab T4 while preserving a quantized base for QLoRA. See `src/models.py` and `src/finetune.py`.
7. **Why double quantization?** It quantizes the quantization constants too, saving additional memory. See `bnb_4bit_use_double_quant=True` in both model loaders.
8. **Why fp16 instead of bf16?** The target T4 has no native bf16 support; both inference and training use fp16 compute. See `src/models.py` and `src/finetune.py`.
9. **Why mask the question and system tokens during training?** Completion-only loss teaches the model to produce the answer, not reproduce the instruction. `format_example()` in `src/finetune.py` sets prompt labels to `-100`.
10. **How is train/validation leakage reduced?** All generated examples from the same source chunk go to the same side of the 95/5 split. See `split_by_chunk()` in `src/split_train.py`.
11. **How is test-question leakage checked?** Training and few-shot questions are embedded with BGE small and compared with all test questions; examples above 0.85 similarity are rejected. See `src/make_train_data.py` and `validate_fewshot()` in `src/prompts.py`.
12. **Does fine-tuning see the test set or judge output?** No. It trains only on synthetic train JSONL. The test set is used only by leakage filters and evaluation code; no gold answers are used for training.
13. **Why use the same base model weights for every condition?** This isolates the adaptation method. A run loads one model with its adapter and uses `disable_adapter()` for base conditions. See `generate()` in `src/models.py`.
14. **Why greedy decoding?** `do_sample=False` removes sampling randomness so repeated conditions use deterministic decoding settings. See `generate()` in `src/models.py`.
15. **What is the difference between zero-shot and prompted?** Zero-shot has only the user question and Qwen's chat template. Prompted adds `SYSTEM_BASE` and four question/answer examples. See `build_messages()` in `src/prompts.py`.
16. **What does RAG add?** It retrieves the top four chunks and provides them to the existing grounded prompt, which asks for file/page citations or a refusal. See `retrieve()` and `build_rag_prompt()` in `src/rag.py`.
17. **What does hit@k mean?** It is the fraction of answerable questions with an expected source file and a page within ±1 in the top k retrieved chunks. The current measured hit@1 is 28/51 and hit@4 is 48/51 (`results/retrieval_eval.csv`).
18. **Why distinguish retrieval and answer correctness?** RAG can fail because relevant evidence was not retrieved or because the model misused retrieved evidence. `retrieval_vs_correctness.csv` separates these cases.
19. **Why is the judge a different model?** A larger separate model reduces self-grading bias. The judge is a Nemotron Ultra NIM model; the tested model is Qwen 3B. IDs are in `config.py`.
20. **How is the judge blinded?** The judge prompt contains only the question, reference answer, and candidate answer. Processing order is shuffled across conditions, and the condition is kept out of the prompt. See `run_judging()` in `src/judge.py`.
21. **Why measure faithfulness separately?** A RAG answer can be factually right but unsupported by the retrieved excerpts. The judge grades support in a separate call. See `FAITHFUL_PROMPT` in `src/judge.py`.
22. **Why use bootstrap confidence intervals?** With only 51 answerable items, a resampling interval communicates sampling uncertainty without assuming a normal accuracy distribution. See `bootstrap_ci()` in `src/analyze_results.py`.
23. **Why use McNemar's test?** The same 51 questions are answered by each condition, so the comparison is paired; McNemar uses the discordant outcomes. See `mcnemar_exact()` in `src/analyze_results.py`.
24. **What does the small sample imply?** The 51 answerable and 9 unanswerable items give indicative evidence only. Small differences may be unstable, and the exact p-values do not remove that limitation.
25. **What conclusions can be made today?** Only the fixed data and retrieval baseline are verified: 180 chunks, 60 test questions, hit@1 28/51, and hit@4 48/51. Comparative accuracy, cost, latency, judge agreement, and fine-tuning outcomes remain `TODO(user)` until the Colab and NIM runs are completed.
