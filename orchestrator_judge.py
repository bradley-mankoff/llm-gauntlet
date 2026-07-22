"""Code-only orchestrator judge with leak-safe template feedback."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from scout_metrics import score_capsule

# Priority order when rejecting (first matching semantics; output sorted by this list).
TAG_PRIORITY: list[str] = [
    "no_handoff",
    "wrong_file",
    "wrong_span",
    "too_thin",
    "weak_match",
]

FEEDBACK_TEMPLATES: dict[str, str] = {
    "no_handoff": (
        "Handoff incomplete: status was not accepted, or FILE/LINES were missing. "
        "Return an accepted capsule with a concrete file path and line range."
    ),
    "wrong_file": (
        "Wrong file: the selected path is not the target implementation file. "
        "Re-locate the symbol or behavior in the correct module before peeks."
    ),
    "wrong_span": (
        "Wrong span: file is plausible but the line range misses the relevant region. "
        "Widen or shift LINES toward the definition or call site, not unrelated helpers."
    ),
    "too_thin": (
        "Too thin: range overlaps the right area but the capsule code is incomplete. "
        "Expand LINES to cover the full definition body needed by the orchestrator."
    ),
    "weak_match": (
        "Weak match: file is right but the capsule is not yet usable "
        "(insufficient overlap with the needed region or text). Refine LINES and code."
    ),
}


@dataclass
class JudgeResult:
    accept: bool
    tags: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    feedback: str = ""


def feedback_for_tags(tags: list[str]) -> str:
    """Join fixed feedback templates for tags; never includes gold content."""
    parts: list[str] = []
    for tag in tags:
        msg = FEEDBACK_TEMPLATES.get(tag)
        if msg:
            parts.append(msg)
    return "\n".join(parts)


def code_judge(
    *,
    file_ok: bool,
    status: str | None,
    pred_file: str | None,
    pred_lines: str | None,
    pred_code: str,
    pred_symbol: str | None,
    gold_files: list[str],
    gold_code: str,
    gold_range: tuple[int, int] | None,
    gold_start: int | None,
    gold_symbol: str | None,
) -> JudgeResult:
    """Accept iff file_ok and (bleu>=0.8 or line_iou>=0.5 or gold_start_hit)."""
    scores = score_capsule(
        file_ok=bool(file_ok),
        pred_code=pred_code or "",
        pred_lines=pred_lines,
        pred_symbol=pred_symbol,
        gold_code=gold_code or "",
        gold_range=gold_range,
        gold_start=gold_start,
        gold_symbol=gold_symbol,
    )
    bleu = float(scores.get("bleu") or 0.0)
    iou = float(scores.get("line_iou") or 0.0)
    start_hit = bool(scores.get("gold_start_hit"))
    code_ok = bool(scores.get("code_ok"))

    st = (status or "").strip().lower()
    status_accepted = st in {"", "accepted", "accept", "ok", "success"}
    has_handoff = bool(pred_file) and bool(pred_lines)

    # Accept path: same composite as score_capsule code_ok.
    if code_ok:
        return JudgeResult(accept=True, tags=[], scores=scores, feedback="")

    tags: list[str] = []
    if (not status_accepted) or (not has_handoff):
        tags.append("no_handoff")
    if not file_ok:
        tags.append("wrong_file")
    else:
        # file_ok but failed accept composite
        if (not start_hit) and iou < 0.15:
            tags.append("wrong_span")
        elif (start_hit or iou >= 0.15) and bleu < 0.15 and iou < 0.35:
            tags.append("too_thin")
        else:
            tags.append("weak_match")

    rank = {t: i for i, t in enumerate(TAG_PRIORITY)}
    tags = sorted(set(tags), key=lambda t: rank.get(t, 999))

    feedback = feedback_for_tags(tags)
    assert_no_gold_leak(feedback, list(gold_files or []), gold_symbol, gold_range)
    return JudgeResult(accept=False, tags=tags, scores=scores, feedback=feedback)


def assert_no_gold_leak(
    feedback: str,
    gold_files: list[str],
    gold_symbol: str | None,
    gold_range: tuple[int, int] | None,
) -> None:
    """Raise if feedback contains gold paths, symbol leaf, or gold line numbers."""
    if not feedback:
        return
    fb = feedback
    fb_lower = fb.lower()

    for gf in gold_files or []:
        if not gf:
            continue
        p = Path(gf)
        candidates = {gf, str(p), p.as_posix(), p.name}
        for c in candidates:
            if c and len(c) >= 2 and c.lower() in fb_lower:
                raise AssertionError(f"gold leak: path/basename {c!r} in feedback")

    if gold_symbol:
        leaf = gold_symbol.split("::")[-1].split(".")[-1].strip()
        if leaf and len(leaf) >= 2 and leaf.lower() in fb_lower:
            raise AssertionError(f"gold leak: symbol leaf {leaf!r} in feedback")

    if gold_range:
        a, b = int(gold_range[0]), int(gold_range[1])
        for n in {a, b}:
            if re.search(rf"\b{n}\b", fb):
                raise AssertionError(f"gold leak: line number {n} in feedback")


def estimate_cost_usd(
    prompt_tokens: int | float,
    completion_tokens: int | float,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """dollars = tokens/1e6 * prices (per 1M tokens)."""
    pin = float(prompt_tokens or 0) / 1e6 * float(price_in_per_mtok or 0)
    pout = float(completion_tokens or 0) / 1e6 * float(price_out_per_mtok or 0)
    return pin + pout
