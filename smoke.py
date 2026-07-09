"""Smoke test: 3 IFEval prompts against the running server. Proves the pipe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_client import list_models, make_client  # noqa: E402
from tasks import ifeval  # noqa: E402


def main() -> int:
    client = make_client()
    models = list_models(client)
    print(f"[smoke] server models: {models}")
    if not models:
        print("ERROR: no models on server. Is llama-server up on :8080?", file=sys.stderr)
        return 1
    model = models[0]
    summary, results = ifeval.run(client, model, n_samples=3, max_tokens=8192)


    print("\n=== smoke summary ===")
    print(json.dumps(summary, indent=2))
    for r in results:
        print(f"\n--- key={r['key']} strict={r['strict_pass']} loose={r['loose_pass']} ---")
        print(f"prompt: {r['prompt'][:200]}...")
        print(f"response ({r['response_len']} chars): {r['response'][:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
