"""NVIDIA NIM OpenAI-Compatible Client with SQLite Caching and Rate Limiting.

Features:
- Enforces strict rate limiting (< 40 requests per minute, default 38 req/min).
- Caches all API responses to disk via SQLite (cache/nim_cache.sqlite) to prevent duplicate spend.
- Implements exponential backoff with jitter on HTTP 429 (Rate Limit) and transient 5xx errors.
- Never logs or exposes raw API keys.

CLI Example:
    # Test NIM client connectivity and response caching
    python src/nim_client.py --prompt "Explain PM-KISAN in one sentence." --model "nvidia/llama-3.1-nemotron-70b-instruct"
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS,
    RATE_LIMITS,
    PATHS,
    get_nvidia_api_key,
    ensure_directories,
)


class NIMResponseCache:
    """Thread-safe SQLite-based disk cache for NVIDIA NIM completions."""

    def __init__(self, db_path: Optional[str] = None):
        ensure_directories()
        self.db_path = str(db_path or PATHS.nim_cache_db)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completion_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    messages_json TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_cache_key(model: str, messages: List[Dict[str, str]], temperature: float,
                          request_options: Optional[Dict[str, Any]] = None) -> str:
        """Generate deterministic SHA256 hash for request parameters."""
        payload = {"model": model, "messages": messages, "temperature": round(temperature, 4)}
        if request_options:
            payload["request_options"] = request_options
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, model: str, messages: List[Dict[str, str]], temperature: float,
            request_options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Fetch cached response text if exists."""
        key = self.compute_cache_key(model, messages, temperature, request_options)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_text FROM completion_cache WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def set(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        response_text: str,
        total_tokens: int = 0,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store completion in SQLite cache."""
        key = self.compute_cache_key(model, messages, temperature, request_options)
        messages_json = json.dumps(messages, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO completion_cache
                (cache_key, model, temperature, messages_json, response_text, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, model, temperature, messages_json, response_text, total_tokens),
            )
            conn.commit()

    def count(self) -> int:
        """Return total cached responses."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM completion_cache")
            return cursor.fetchone()[0]


class NIMRateLimiter:
    """Sliding-window rate limiter ensuring requests stay below limits (< 40 RPM)."""

    def __init__(
        self,
        max_per_minute: int = RATE_LIMITS.max_requests_per_minute,
        min_interval: float = RATE_LIMITS.min_request_interval_seconds,
    ):
        self.max_per_minute = max_per_minute
        self.min_interval = min_interval
        self.timestamps: List[float] = []
        self.last_call_time: float = 0.0

    def wait(self) -> None:
        """Wait if needed to conform to the rate limit before proceeding."""
        now = time.time()

        # Enforce minimum inter-request spacing
        elapsed_since_last = now - self.last_call_time
        if elapsed_since_last < self.min_interval:
            time.sleep(self.min_interval - elapsed_since_last)

        now = time.time()
        # Clean timestamps older than 60 seconds
        self.timestamps = [t for t in self.timestamps if now - t < 60.0]

        # If at max RPM capacity, wait until the oldest timestamp clears 60s
        if len(self.timestamps) >= self.max_per_minute:
            sleep_duration = 60.0 - (now - self.timestamps[0]) + 0.1
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        current_time = time.time()
        self.timestamps.append(current_time)
        self.last_call_time = current_time


class NIMClient:
    """High-level client for NVIDIA NIM with caching, rate-limiting, and retries."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_nvidia_api_key()
        self.client = OpenAI(
            base_url=MODELS.nvidia_base_url,
            api_key=self.api_key,
        )
        self.cache = NIMResponseCache()
        self.limiter = NIMRateLimiter()

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        use_cache: bool = True,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, bool]:
        """Generate a chat completion with caching and backoff.

        Only `message.content` is returned; reasoning fields are never used.
        Optional request options are included in the cache key and can control
        provider-specific behavior such as disabling teacher reasoning.

        Args:
            messages: List of chat message dicts (role, content).
            model: NIM model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in completion (2048 default gives
                reasoning models room to finish their chain-of-thought).
            use_cache: Whether to check and write to disk cache.
            request_options: Extra provider options forwarded in the request body.

        Returns:
            Tuple of (response_text, latency_seconds, is_from_cache).
        """
        # 1. Check disk cache
        if use_cache:
            cached_text = self.cache.get(model, messages, temperature, request_options)
            if cached_text is not None:
                return cached_text, 0.001, True

        # 2. Rate limiting & Exponential Backoff Retries
        retries = 0
        backoff = RATE_LIMITS.initial_backoff_seconds

        while True:
            self.limiter.wait()
            start_time = time.time()
            try:
                call_options = {"extra_body": request_options} if request_options else {}
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **call_options,
                )
                latency = time.time() - start_time
                choice_msg = response.choices[0].message
                reply_text = choice_msg.content or ""
                if not reply_text:
                    # Empty content is a hard error; reasoning fields are never inspected.
                    raise RuntimeError("NIM returned empty message.content; reasoning fields are not used.")
                total_tokens = response.usage.total_tokens if response.usage else 0

                # 3. Save to disk cache
                if use_cache:
                    self.cache.set(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_text=reply_text,
                        total_tokens=total_tokens,
                        request_options=request_options,
                    )

                return reply_text, latency, False

            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                retries += 1
                if retries > RATE_LIMITS.max_retries:
                    raise RuntimeError(
                        f"NIM API request failed after {RATE_LIMITS.max_retries} retries for model {model}: {e}"
                    ) from e
                sleep_time = backoff * (RATE_LIMITS.backoff_multiplier ** (retries - 1))
                print(f"[RateLimit/Error] Caught {type(e).__name__}. Retrying in {sleep_time:.2f}s (Attempt {retries}/{RATE_LIMITS.max_retries})...")
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Test NVIDIA NIM Client with caching and rate limiting.")
    parser.add_argument("--prompt", type=str, default="Summarize the core benefit of PM-KISAN in one bullet.", help="Prompt text")
    parser.add_argument("--model", type=str, default=MODELS.teacher_model_name, help="NIM Model ID")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--no-cache", action="store_true", help="Bypass disk cache")
    args = parser.parse_args()

    client = NIMClient()
    messages = [{"role": "user", "content": args.prompt}]

    print(f"Connecting to NVIDIA NIM: {MODELS.nvidia_base_url}")
    print(f"Model: {args.model} | Temperature: {args.temp}")
    print(f"Prompt: {args.prompt}")

    reply, latency, from_cache = client.generate(
        messages=messages,
        model=args.model,
        temperature=args.temp,
        use_cache=not args.no_cache,
    )

    print("\n--- Response ---")
    print(reply.strip())
    print("----------------")
    print(f"Latency: {latency:.3f}s | From Disk Cache: {from_cache}")
    print(f"Total Cached Items in SQLite: {client.cache.count()}")


if __name__ == "__main__":
    main()
