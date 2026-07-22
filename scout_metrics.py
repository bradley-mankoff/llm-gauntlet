"""Scoring helpers for tiered scout capsules."""
from __future__ import annotations

import re
from pathlib import Path


def clean_pred_code(text: str) -> str:
    s = text or ""
    m = re.search(r"```(?:\w+)?\n([\s\S]*?)```", s)
    if m:
        s = m.group(1)
    s = re.sub(r"^VERDICT:.*$", "", s, flags=re.M)
    s = re.sub(r"^(FILE|SYMBOL|LINES|REASON|CENTER_LINE):.*$", "", s, flags=re.M)
    return s.strip()


def parse_line_range(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", str(s))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        a, b = b, a
    return a, b


def line_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if not a or not b:
        return 0.0
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return float(inter) / float(union) if union else 0.0


def gold_start_in_range(gold_start: int | None, pred: tuple[int, int] | None) -> bool:
    if not gold_start or not pred:
        return False
    return pred[0] <= int(gold_start) <= pred[1]


def find_symbol_lines(path: Path, symbol: str, *, limit: int = 8) -> list[int]:
    if not symbol or not path.is_file():
        return []
    # strip qualifiers: Foo::bar -> bar, foo.bar -> bar
    leaf = symbol.split("::")[-1].split(".")[-1].strip()
    if not leaf or leaf in {"?", "unknown", "null"}:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    pat_def = re.compile(rf"\b{re.escape(leaf)}\s*\(")
    pat_any = re.compile(rf"\b{re.escape(leaf)}\b")
    hits: list[int] = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s or s.startswith(("//", "/*", "*", "#", "import ", "from ", "using ")):
            continue
        if pat_def.search(ln) or pat_any.search(ln):
            hits.append(i)
            if len(hits) >= limit:
                break
    return hits


def extract_tight_block(
    path: Path,
    start_line: int,
    *,
    max_lines: int = 40,
    symbol: str | None = None,
) -> tuple[str, int, int]:
    """Return (code, start_1based, end_1based) tight definition slice."""
    if not path.is_file():
        return "", 0, 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "", 0, 0
    n = len(lines)
    i = max(0, min(n - 1, int(start_line or 1) - 1))

    # If symbol provided and not near start, snap to nearest symbol hit.
    if symbol:
        hits = find_symbol_lines(path, symbol, limit=12)
        if hits:
            # prefer hit at/after nominal start, else nearest
            after = [h for h in hits if h >= (start_line or 1) - 5]
            pick = after[0] if after else min(hits, key=lambda h: abs(h - (start_line or 1)))
            i = pick - 1

    # walk up through short comment/specifiers only
    lo = i
    for _ in range(10):
        if lo <= 0:
            break
        prev = lines[lo - 1].strip()
        if not prev:
            break
        if prev.startswith(("#include", "#if", "#endif", "#define", "#pragma", "using ", "namespace ")):
            break
        if prev.startswith(("//", "/*", "*", "*/", "@")) or prev.endswith((",", "\\", ")")):
            lo -= 1
            continue
        if prev in {"public:", "private:", "protected:"}:
            lo -= 1
            continue
        break

    hi_cap = min(n, lo + max_lines)
    chunk = lines[lo:hi_cap]
    # brace-balanced cut, capped
    out = []
    bal = 0
    started = False
    for ln in chunk:
        out.append(ln)
        for ch in ln:
            if ch == "{":
                bal += 1
                started = True
            elif ch == "}":
                bal -= 1
        if started and bal <= 0:
            break
        if not started and len(out) >= max_lines:
            break
    if not out:
        out = lines[i : min(n, i + min(20, max_lines))]
        lo = i
    text = "\n".join(out).rstrip()
    start_1 = lo + 1
    end_1 = lo + len(out)
    return (text + ("\n" if text else "")), start_1, end_1


def slice_file_lines(path: Path, start: int, end: int, *, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ""
    a = max(1, int(start))
    b = max(a, int(end))
    if b - a + 1 > max_lines:
        b = a + max_lines - 1
    a0 = min(len(lines), a) - 1
    b0 = min(len(lines), b)
    text = "\n".join(lines[a0:b0]).rstrip()
    return text + ("\n" if text else "")


def code_bleu(gt_code: str, hyp: str) -> float:
    gt = (gt_code or "").strip()
    hy = clean_pred_code(hyp or "")
    if not gt or not hy:
        return 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return float(
            sentence_bleu(
                [gt.split()],
                hy.split(),
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 1.0 if gt == hy else 0.0


def score_capsule(
    *,
    file_ok: bool,
    pred_code: str,
    pred_lines: str | None,
    pred_symbol: str | None,
    gold_code: str,
    gold_range: tuple[int, int] | None,
    gold_start: int | None,
    gold_symbol: str | None,
) -> dict:
    pred_range = parse_line_range(pred_lines)
    pred = clean_pred_code(pred_code)
    bleu = code_bleu(gold_code, pred) if file_ok else 0.0
    iou = line_iou(pred_range, gold_range) if file_ok else 0.0
    start_hit = gold_start_in_range(gold_start, pred_range) if file_ok else False
    sym_hit = False
    if file_ok and gold_symbol and pred_symbol:
        g = gold_symbol.split("::")[-1].split(".")[-1].lower()
        p = pred_symbol.split("::")[-1].split(".")[-1].lower()
        sym_hit = bool(g and (g == p or g in p or p in g))
    if file_ok and gold_symbol and pred and not sym_hit:
        g = gold_symbol.split("::")[-1].split(".")[-1]
        if g and re.search(rf"\b{re.escape(g)}\b", pred):
            sym_hit = True

    # Primary "code ok": useful to an orchestrator — right region or strong text match.
    code_ok = bool(file_ok and (bleu >= 0.8 or iou >= 0.5 or start_hit))
    code_ok_strict = bool(file_ok and bleu >= 0.8)
    return {
        "bleu": round(float(bleu), 4),
        "line_iou": round(float(iou), 4),
        "gold_start_hit": bool(start_hit),
        "symbol_hit": bool(sym_hit),
        "code_ok": code_ok,
        "code_ok_strict_bleu": code_ok_strict,
        "pred_range": list(pred_range) if pred_range else None,
        "gold_range": list(gold_range) if gold_range else None,
    }


def rebuild_query_gold(repo: str | Path, q: dict, *, max_lines: int = 40) -> dict:
    """Attach tight gold code + range onto query dict (copy)."""
    out = dict(q)
    repo = Path(repo)
    primary = q.get("file") or ""
    golds = list(q.get("gold_files") or ([primary] if primary else []))
    symbol = (q.get("symbol") or "").strip() or None
    start = int(q.get("code_start_line") or q.get("start_line") or 1)

    best_code, best_a, best_b, best_rel = "", 0, 0, ""
    best_score = -1

    ordered: list[str] = []
    if primary:
        ordered.append(primary)
    for g in golds:
        if g and g not in ordered:
            ordered.append(g)

    leaf = (symbol or "").split("::")[-1].split(".")[-1]
    for rel in ordered:
        code, a, b = extract_tight_block(
            repo / rel,
            start if rel == primary else 1,
            max_lines=max_lines,
            symbol=symbol,
        )
        if not code.strip() and symbol:
            hits = find_symbol_lines(repo / rel, symbol)
            if hits:
                code, a, b = extract_tight_block(
                    repo / rel, hits[0], max_lines=max_lines, symbol=symbol
                )
        score = len(code)
        if leaf and leaf in code:
            score += 10000
        if rel.endswith((".cpp", ".c", ".cc", ".cxx", ".py", ".ts", ".js")):
            score += 500
        if score > best_score:
            best_score = score
            best_code, best_a, best_b, best_rel = code, a, b, rel

    out["code"] = best_code
    out["code_start_line"] = best_a or start
    out["code_end_line"] = best_b or best_a or start
    if best_rel and best_rel != primary:
        out["code_file"] = best_rel
    return out
