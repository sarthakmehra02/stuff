"""Qwen inference helpers shared by Colab benchmark and demo."""

import logging
import random
import time
from typing import Any

from config import MAX_NEW_TOKENS, MODEL_ID, SEED

logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def peak_gpu_mb() -> float:
    try:
        import torch
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    except ImportError:
        return 0.0


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    except ImportError:
        return "PyTorch not installed"


def load_model(adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    set_seed()
    if not torch.cuda.is_available():
        logger.warning("No CUDA device found. This project requires a Colab T4; 4-bit model loading may fail on CPU.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quantization, device_map="auto", low_cpu_mem_usage=True
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    if torch.cuda.is_available():
        logger.info("GPU: %s; peak allocated %.1f MiB", torch.cuda.get_device_name(), peak_gpu_mb())
    return model, tokenizer


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]] | None = None,
             use_adapter: bool = False, max_new_tokens: int = MAX_NEW_TOKENS,
             prompt: str | None = None, system: str | None = None) -> dict[str, Any]:
    import torch

    if messages is None:
        messages = ([{"role": "system", "content": system}] if system else [])
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_tokens = int(input_ids.shape[-1])
    start = time.perf_counter()
    adapter_context = getattr(model, "disable_adapter", None)
    if not use_adapter and callable(adapter_context):
        context = adapter_context()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with torch.inference_mode(), context:
        outputs = model.generate(
            input_ids, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    new_ids = outputs[0, prompt_tokens:]
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "latency_s": latency,
        "prompt_tokens": prompt_tokens,
        "new_tokens": int(new_ids.shape[-1]),
    }
