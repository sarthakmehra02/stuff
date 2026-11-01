"""Central Configuration for Indian Farmer Schemes LLM Comparison Project.

This module centralizes all model identifiers, paths, API endpoints, rate
limiting parameters, and QLoRA training hyperparameters for running on
a free Google Colab T4 GPU (fp16, no bf16).

Never hardcode API keys here; keys are loaded dynamically from environment
variables or .env files.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent

# Shared experiment settings. Keep prompt/model settings here so every stage
# consumes one source of truth.
SYSTEM_BASE = (
    "You answer questions about three Indian farmer schemes: PM-KISAN (income support), "
    "PMFBY (crop insurance) and the Kisan Credit Card (crop loans and interest subvention). "
    "Answer in 1-3 short sentences. If the answer is not in the official documents or you "
    "are not sure, reply exactly: Not found in the documents"
)
REFUSAL_STRING = "Not found in the documents"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 300
SEED = 42
TOP_K = 4
FEWSHOT_FILE = PROJECT_ROOT / "data" / "fewshot.json"
TEST_QA_FILE = PROJECT_ROOT / "data" / "test_qa.json"
TRAIN_QA_FILE = PROJECT_ROOT / "data" / "train_qa.jsonl"
TRAIN_SPLIT_DIR = PROJECT_ROOT / "data" / "train_split"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


def get_raw_pdfs_dir() -> Path:
    """Find raw_pdfs directory handling case sensitivity between Windows/Linux."""
    candidates = [
        PROJECT_ROOT / "data" / "raw_pdfs",
        PROJECT_ROOT / "Data" / "raw_pdfs",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    # Default fallback
    return PROJECT_ROOT / "data" / "raw_pdfs"


def get_nvidia_api_key() -> str:
    """Retrieve and sanitize NVIDIA API key from environment variable.

    Returns:
        Clean API key string without 'Bearer ' prefix.

    Raises:
        ValueError: If NVIDIA_API_KEY is not set or empty.
    """
    raw_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not raw_key:
        raise ValueError(
            "NVIDIA_API_KEY environment variable is not set. "
            "Please create a .env file or export NVIDIA_API_KEY with your key."
        )
    # Strip any accidental 'Bearer ' prefix
    if raw_key.lower().startswith("bearer "):
        raw_key = raw_key[7:].strip()
    return raw_key


@dataclass(frozen=True)
class ModelConfig:
    """Model names, temperatures, and endpoints."""
    # Base model targeted for free Colab T4
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # NVIDIA NIM endpoints & models
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    teacher_model_name: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    judge_model_name: str = "nvidia/nemotron-3-ultra-550b-a55b"
    # Model used by RAG answer() when no generate_fn is provided (dev default)
    # Development-only NIM default for the standalone RAG CLI; benchmark answers
    # are always generated locally by Qwen through run_conditions.py.
    generation_model_name: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"

    # Sampling temperatures
    teacher_temperature: float = 0.3
    judge_temperature: float = 0.0

    # Embedding model for RAG (runs locally on CPU or T4)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting and retry configurations for NVIDIA NIM API.

    Must stay strictly under 40 requests per minute.
    """
    max_requests_per_minute: int = 38
    min_request_interval_seconds: float = 1.6  # 60 / 38 ~= 1.58s
    max_retries: int = 5
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0


@dataclass(frozen=True)
class PathConfig:
    """File paths for data, cache, model checkpoints, and evaluation results."""
    project_root: Path = PROJECT_ROOT
    raw_pdfs_dir: Path = field(default_factory=get_raw_pdfs_dir)

    # Processed data
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    extracted_pages_file: Path = processed_dir / "extracted_pages.jsonl"
    chunks_file: Path = processed_dir / "chunks.json"
    train_qa_file: Path = processed_dir / "train_qa.jsonl"
    test_qa_file: Path = processed_dir / "test_qa.jsonl"

    # FAISS Vector Index & Metadata
    faiss_index_file: Path = processed_dir / "faiss_index.bin"
    faiss_metadata_file: Path = processed_dir / "faiss_metadata.json"
    faiss_manifest_file: Path = processed_dir / "faiss_manifest.json"

    # ChromaDB Vector Store
    chroma_db_dir: Path = processed_dir / "chroma_db"

    # SQLite Cache for NIM API calls
    cache_dir: Path = PROJECT_ROOT / "cache"
    nim_cache_db: Path = cache_dir / "nim_cache.sqlite"

    # Model adapters & checkpoints
    models_dir: Path = PROJECT_ROOT / "models"
    qlora_adapter_dir: Path = models_dir / "qlora_adapter"

    # Results & Reports
    results_dir: Path = PROJECT_ROOT / "results"
    report_dir: Path = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class QLoRAConfig:
    """Hyperparameters optimized for free Colab T4 GPU (16GB VRAM, sm_75).

    T4 compute capability (7.5) does NOT support native bfloat16.
    fp16 must be True, and bf16 must be False.
    """
    fp16: bool = True
    bf16: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "epoch"


# Instantiate singleton configurations
MODELS = ModelConfig()
RATE_LIMITS = RateLimitConfig()
PATHS = PathConfig()
QLORA = QLoRAConfig()

# Direct module-level settings
TEACHER_MODEL = MODELS.teacher_model_name
JUDGE_MODEL = MODELS.judge_model_name
GENERATION_MODEL = MODELS.generation_model_name
TEACHER_TEMPERATURE = MODELS.teacher_temperature
JUDGE_TEMPERATURE = MODELS.judge_temperature

# Page filtering ranges: filename -> list of (start_page, end_page) 1-indexed, inclusive
# Files not present in this dict are fully included.
PAGE_RANGES = {
    "pmfby_operational_guidelines_2023.pdf": [
        (11, 11), (14, 23), (47, 69), (88, 126),
        (134, 138), (168, 173), (211, 212),
    ],
}



def ensure_directories() -> None:
    """Ensure all required runtime directories exist."""
    for directory in [
        PATHS.processed_dir,
        PATHS.cache_dir,
        PATHS.chroma_db_dir,
        PATHS.models_dir,
        PATHS.qlora_adapter_dir,
        PATHS.results_dir,
        PATHS.report_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project Root: {PATHS.project_root}")
    print(f"Raw PDFs Dir: {PATHS.raw_pdfs_dir}")
    print(f"Base Model: {MODELS.base_model_name}")
    print(f"Teacher Model: {TEACHER_MODEL} (temp={TEACHER_TEMPERATURE})")
    print(f"Judge Model:   {JUDGE_MODEL} (temp={JUDGE_TEMPERATURE})")
    print(f"Generation Model: {GENERATION_MODEL}")
    print(f"Rate limit: <= {RATE_LIMITS.max_requests_per_minute} req/min")
    print(f"Colab T4 Dtypes: fp16={QLORA.fp16}, bf16={QLORA.bf16}")
    ensure_directories()
    print("Runtime directories verified.")
