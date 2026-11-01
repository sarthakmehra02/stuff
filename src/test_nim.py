"""Smoke-test for NVIDIA NIM connectivity across all configured models.

Checks that TEACHER_MODEL and JUDGE_MODEL each:
1. Respond to a one-word prompt with clean content (no reasoning prefix).
2. Return valid JSON when asked for it.

Reasoning models (TEACHER, JUDGE) separate their chain-of-thought into
`reasoning_content`; the final answer is always in `message.content`.
The NIM client reads `content` only, so no stripping is needed.

Exits with code 1 if any model fails so CI can catch it.

Usage:
    python src/test_nim.py
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import TEACHER_MODEL, JUDGE_MODEL, GENERATION_MODEL
from src.nim_client import NIMClient

ONE_WORD_PROMPT = "Reply with exactly one word: OK"
JSON_PROMPT = 'Return only this JSON: {"ok": true}'

# Reasoning models need more tokens so content isn't truncated.
# Non-reasoning models are fine with fewer.
REASONING_MODELS = {TEACHER_MODEL, JUDGE_MODEL}


def _max_tokens_for(model: str) -> int:
    return 2048 if model in REASONING_MODELS else 256


def _extract_final(text: str) -> str:
    """Return the text as-is — for NIM reasoning models, content is already
    separated from reasoning_content by the API.  No stripping needed."""
    return text.strip()


def check_one_word(client: NIMClient, model: str, role: str) -> bool:
    """Send a one-word prompt; pass if reply (after extraction) is a short word."""
    try:
        reply, latency, from_cache = client.generate(
            messages=[{"role": "user", "content": ONE_WORD_PROMPT}],
            model=model,
            temperature=0.0,
            max_tokens=_max_tokens_for(model),
            use_cache=False,
        )
        answer = _extract_final(reply)
        # Accept "OK" (case-insensitive) or any single short word
        clean = len(answer.split()) <= 3 and len(answer) < 40
        status = "PASS" if clean else "FAIL"
        print(f"    [{status}] one-word  ({latency:.2f}s) -> {answer!r}")
        return clean
    except Exception as exc:
        print(f"    [FAIL] one-word  -> Exception: {exc}")
        return False


def check_json(client: NIMClient, model: str, role: str) -> bool:
    """Ask for JSON; pass if json.loads succeeds after extraction."""
    try:
        reply, latency, from_cache = client.generate(
            messages=[{"role": "user", "content": JSON_PROMPT}],
            model=model,
            temperature=0.0,
            max_tokens=_max_tokens_for(model),
            use_cache=False,
        )
        answer = _extract_final(reply)
        # Strip markdown code fences if present
        stripped = answer.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(stripped)
        status = "PASS"
        print(f"    [{status}] json      ({latency:.2f}s) -> parsed={parsed!r}")
        return True
    except json.JSONDecodeError as e:
        print(f"    [FAIL] json      -> JSONDecodeError: {e}  (raw={_extract_final(reply)[:120]!r})")
        return False
    except Exception as exc:
        print(f"    [FAIL] json      -> Exception: {exc}")
        return False


def check_model(client: NIMClient, model: str, role: str) -> bool:
    print(f"\n{'─' * 62}")
    print(f"  Role:  {role}")
    print(f"  Model: {model}")
    ok_word = check_one_word(client, model, role)
    ok_json = check_json(client, model, role)
    return ok_word and ok_json


def main():
    """Test all three configured NIM models and exit non-zero if any fail."""
    print("=" * 62)
    print("  NIM MODEL SMOKE TEST  (2 tests per model)")
    print("=" * 62)
    print(f"  TEACHER_MODEL    = {TEACHER_MODEL}")
    print(f"  JUDGE_MODEL      = {JUDGE_MODEL}")
    print(f"  GENERATION_MODEL = {GENERATION_MODEL}")
    print()
    print("  NOTE: Reasoning models (TEACHER, JUDGE) keep chain-of-thought")
    print("  in reasoning_content; message.content is the clean final answer.")
    print("  max_tokens is raised to 2048 for reasoning models.")

    client = NIMClient()

    results = {
        "TEACHER": check_model(client, TEACHER_MODEL, "TEACHER"),
        "JUDGE": check_model(client, JUDGE_MODEL, "JUDGE"),
    }

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    all_passed = True
    model_map = {
        "TEACHER":    TEACHER_MODEL,
        "JUDGE":      JUDGE_MODEL,
    }
    for role, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {role:12s} ({model_map[role]})")
        if not passed:
            all_passed = False

    print("=" * 62)
    if all_passed:
        print("  All models returned clean OK and valid JSON.  NIM OK.")
        sys.exit(0)
    else:
        print("  One or more models FAILED. Check API key and model availability.")
        sys.exit(1)


if __name__ == "__main__":
    main()
