"""Debug probe: print the full raw response object for TEACHER and JUDGE models.

Shows:
- Full response dict (serialized)
- Which field the text arrives in (content vs reasoning_content vs reasoning)
- Token counts
- Whether content was cut off (finish_reason)

Usage:
    python src/debug_nim_raw.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import TEACHER_MODEL, JUDGE_MODEL
from src.nim_client import NIMClient, get_nvidia_api_key
from openai import OpenAI
import config

PROBE_PROMPT = "Reply with exactly one word: OK"

def probe_raw(model: str, role: str):
    print(f"\n{'=' * 70}")
    print(f"  RAW PROBE: {role} = {model}")
    print(f"{'=' * 70}")

    api_key = get_nvidia_api_key()
    client = OpenAI(
        base_url=config.MODELS.nvidia_base_url,
        api_key=api_key,
    )

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            temperature=0.0,
            max_tokens=512,  # enough to see reasoning preamble
        )
        latency = time.time() - t0

        print(f"\nLatency: {latency:.2f}s")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Usage: {response.usage}")

        msg = response.choices[0].message
        print(f"\nmsg.content = {msg.content!r}")
        print(f"msg.reasoning_content = {getattr(msg, 'reasoning_content', 'NOT PRESENT')!r}")
        print(f"msg.reasoning = {getattr(msg, 'reasoning', 'NOT PRESENT')!r}")

        # Dump full model_dump if available
        try:
            raw = msg.model_dump()
            print(f"\nFull message dict:\n{json.dumps(raw, indent=2, default=str)}")
        except Exception as e:
            print(f"\n[Could not model_dump: {e}]")

        # Also try full response
        try:
            resp_raw = response.model_dump()
            print(f"\nFull response choices[0]:\n{json.dumps(resp_raw['choices'][0], indent=2, default=str)}")
        except Exception as e:
            print(f"[Could not dump full response: {e}]")

    except Exception as exc:
        print(f"[EXCEPTION] {type(exc).__name__}: {exc}")


def main():
    probe_raw(TEACHER_MODEL, "TEACHER")
    probe_raw(JUDGE_MODEL, "JUDGE")


if __name__ == "__main__":
    main()
