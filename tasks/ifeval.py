"""IFEval: instruction following with verifiable constraints (541 prompts).

Uses the vendored google-research `instruction_following_eval` verifier.
Reports both `strict` and `loose` pass rates — strict treats leading/trailing
whitespace as instruction failure, loose is more forgiving.

Compatibility note: the HF `google/IFEval` dataset stores all-possible kwargs
(with nulls) for every instruction in a row. The vendored google-research
verifier calls `instruction.build_description(**kwargs)` which TypeErrors
when extras are passed (e.g. `CommaChecker.build_description(self)` takes
no args but gets `num_highlights=None, relation=None, ...`). We filter each
kwargs dict to the parameters its instruction's `build_description` actually
accepts before constructing the InputExample.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Make the vendored verifier importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ifeval_lib"))
from instruction_following_eval import instructions_registry  # noqa: E402
from instruction_following_eval.evaluation_lib import (  # noqa: E402
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from datasets import load_dataset  # noqa: E402
from bench_client import chat as _bench_chat  # noqa: E402

NAME = "ifeval"
MAX_TOKENS_DEFAULT = 8192
DATASET_ID = "google/IFEval"


def _filter_kwargs_for_instruction(instruction_id: str, kwargs: dict) -> dict:
    """Strip kwargs that the instruction's `build_description` doesn't accept.

    Keeps only the keys that the instruction class declares as parameters,
    and drops any that are None (so an instruction that doesn't need a
    particular param doesn't get a null passed in).
    """
    cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    sig = inspect.signature(cls.build_description)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid and v is not None}


def run(client, model: str, n_samples: int | None = None, max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, progress_every: int = 1, seed: int | None = 42, **kwargs):
    ds = load_dataset(DATASET_ID, split="train", cache_dir="./.cache")
    if seed:
        ds = ds.shuffle(seed=seed)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    print(f"[ifeval] {len(ds)} samples, model={model}, max_tokens={max_tokens}, seed={seed}")

    results: list[dict] = []
    for i, row in enumerate(ds):
        prompt = row["prompt"]
        instr_ids = row["instruction_id_list"]
        kwarg_list = row["kwargs"]

        # Filter kwargs to what each instruction's build_description accepts.
        # See module docstring for why this is needed.
        filtered_kwargs = [
            _filter_kwargs_for_instruction(instr_id, kw)
            for instr_id, kw in zip(instr_ids, kwarg_list)
        ]

        try:
            resp = _bench_chat(
                client,
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                think_off=bool(kwargs.get("think_off", False)),
                no_think=bool(kwargs.get("no_think", False)),
            )
        except Exception as exc:  # noqa: BLE001
            # Network/timeout/server-error: log and continue. The previous run
            # died here when one sample exceeded the openai client's 600s timeout
            # and the unhandled APITimeoutError killed the whole process.
            print(f"  [{i + 1}/{len(ds)}] ERROR on key={row['key']}: {type(exc).__name__}: {exc}")
            results.append({
                "key": row["key"],
                "prompt": prompt,
                "response_len": 0,
                "instruction_ids": instr_ids,
                "strict_pass": False,
                "loose_pass": False,
                "per_instruction_strict": {"_error": repr(exc)},
                "error": repr(exc),
            })
            continue
        # The Ornith/Qwen models in this setup may emit a `reasoning_content`
        # first; we only grade the final `content` field. If content is empty
        # (length cap hit during reasoning), record as failed.
        msg = resp.choices[0].message
        response_text = (msg.content or "").strip()

        example = InputExample(
            key=row["key"],
            instruction_id_list=instr_ids,
            kwargs=filtered_kwargs,
            prompt=prompt,
        )
        prompt_to_response = {prompt: response_text}
        try:
            strict_out = test_instruction_following_strict(example, prompt_to_response)
            loose_out = test_instruction_following_loose(example, prompt_to_response)
            strict_pass = bool(strict_out.follow_all_instructions)
            loose_pass = bool(loose_out.follow_all_instructions)
            per_strict = dict(zip(instr_ids, strict_out.follow_instruction_list))
        except Exception as exc:  # noqa: BLE001
            # A bad instruction definition shouldn't kill the whole run
            strict_pass = loose_pass = False
            per_strict = {"_error": repr(exc)}

        results.append({
            "key": row["key"],
            "prompt": prompt,
            "response_len": len(response_text),
            "instruction_ids": instr_ids,
            "strict_pass": strict_pass,
            "loose_pass": loose_pass,
            "per_instruction_strict": per_strict,
        })

        if (i + 1) % progress_every == 0 or i == 0:
            sp = sum(1 for r in results if r["strict_pass"])
            lp = sum(1 for r in results if r["loose_pass"])
            n = len(results)
            avg_len = sum(r["response_len"] for r in results) / n
            truncated = sum(1 for r in results if r["response_len"] < 100)
            print(
                f"  [{i + 1}/{len(ds)}] "
                f"strict={sp}/{n} ({sp / n:.2%})  "
                f"loose={lp}/{n} ({lp / n:.2%})  "
                f"avg_len={avg_len:.0f}  "
                f"truncated={truncated}/{n}"
            )

    strict_pass = sum(1 for r in results if r["strict_pass"])
    loose_pass = sum(1 for r in results if r["loose_pass"])
    summary = {
        "benchmark": NAME,
        "model": model,
        "n_samples": len(results),
        "strict_pass": strict_pass,
        "loose_pass": loose_pass,
        "strict_pass_rate": strict_pass / len(results) if results else 0.0,
        "loose_pass_rate": loose_pass / len(results) if results else 0.0,
        "seed": seed,
    }
    return summary, results
