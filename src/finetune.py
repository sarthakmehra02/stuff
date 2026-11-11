"""QLoRA training with labels masked on system and user tokens."""

import argparse
import json
import logging
import time
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_ID, SEED, SYSTEM_BASE

logger = logging.getLogger("finetune")


def format_example(example: dict, tokenizer, max_seq_length: int = 384) -> dict:
    """Render a chat sample and mask everything before assistant completion."""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": example["question"]},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": example["answer"]}]
    prefix = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
    prefix_ids = list(prefix)
    answer_ids = list(full[len(prefix_ids):])
    # Preserve the supervised answer whenever a model's chat template makes
    # the prompt longer than the sequence budget. Keep the most recent prompt
    # tokens (the question and assistant marker) and truncate only excess
    # context from the left. If an answer itself is unusually long, truncate
    # its tail rather than dropping the example entirely.
    answer_ids = answer_ids[:max_seq_length]
    available_prefix = max_seq_length - len(answer_ids)
    prefix_ids = prefix_ids[-max(0, available_prefix):] if available_prefix else []
    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    if not answer_ids:
        raise ValueError("Sample answer is empty after tokenization")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(train_path: Path, val_path: Path, out_path: Path, epochs: int = 3,
          smoke: bool = False, max_seq_length: int = 1024,
          learning_rate: float = 1e-4):
    import torch
    import matplotlib.pyplot as plt
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU. Run this script in Colab with a T4 runtime.")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    if smoke:
        train_rows = train_rows[:40]
    train_data = Dataset.from_list([format_example(row, tokenizer, max_seq_length) for row in train_rows])
    val_data = Dataset.from_list([format_example(row, tokenizer, max_seq_length) for row in val_rows])
    preview = format_example(train_rows[0], tokenizer, max_seq_length)
    print("MASKING CHECK — decoded prompt tokens (masked) followed by supervised answer tokens:")
    print(tokenizer.decode([token for token, label in zip(preview["input_ids"], preview["labels"]) if label == -100]))
    print("SUPERVISED ANSWER:", tokenizer.decode([label for label in preview["labels"] if label != -100]))
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args_dict = dict(
        output_dir=str(checkpoint_dir), num_train_epochs=epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.05, weight_decay=0.01, fp16=True, bf16=False, optim="paged_adamw_8bit",
        seed=SEED, eval_strategy="epoch", logging_steps=10,
        save_strategy="epoch", save_total_limit=3,
        gradient_checkpointing=True, report_to="none",
        max_steps=20 if smoke else -1,
    )
    # Transformers releases differ in the evaluation strategy parameter name.
    import inspect
    if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")
    training_args = TrainingArguments(**args_dict)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as exc:
        if args_dict["optim"] != "paged_adamw_8bit" or not any(term in str(exc).lower() for term in ("bitsandbytes", "paged", "8-bit")):
            raise
        logger.warning("paged_adamw_8bit failed; retrying with adamw_torch: %s", exc)
        args_dict["optim"] = "adamw_torch"
        if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
            args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy", "steps")
        training_args = TrainingArguments(**args_dict)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                          data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
        checkpoint = get_last_checkpoint(str(checkpoint_dir))
        result = trainer.train(resume_from_checkpoint=checkpoint)
    train_metrics = trainer.evaluate(train_data, metric_key_prefix="train_final")
    val_metrics = trainer.evaluate(val_data, metric_key_prefix="val")
    adapter_dir = out_path / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logs = trainer.state.log_history
    curve = out_path / "loss_curve.png"
    train_points = [(row["step"], row["loss"]) for row in logs if "loss" in row and "eval_loss" not in row]
    val_points = [(row["step"], row["eval_loss"]) for row in logs if "eval_loss" in row]
    plt.figure(figsize=(7, 4))
    if train_points:
        plt.plot(*zip(*train_points), label="train loss")
    if val_points:
        plt.plot(*zip(*val_points), label="validation loss")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(curve, dpi=160); plt.close()
    overfit = bool(train_points and val_points and val_points[-1][1] > val_points[0][1] and train_points[-1][1] < train_points[0][1])
    summary = {
        "model_id": MODEL_ID, "train_examples": len(train_rows), "validation_examples": len(val_rows),
        "steps": trainer.state.global_step, "epochs": epochs, "smoke": smoke,
        "wall_clock_seconds": time.perf_counter() - start,
        "final_train_loss": train_metrics.get("train_final_loss", result.training_loss),
        "final_validation_loss": val_metrics.get("val_loss"), "train_metrics": train_metrics,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / (1024 ** 2), "overfitting_signal": overfit,
        "max_seq_length": max_seq_length, "hyperparameters": args_dict,
        "library_versions": {p: metadata.version(p) for p in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets")},
    }
    (out_path / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train_split/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/train_split/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/run1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args.train, args.val, args.out, args.epochs, args.smoke,
          args.max_seq_length, args.learning_rate)


if __name__ == "__main__":
    main()
