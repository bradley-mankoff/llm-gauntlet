#!/usr/bin/env python3
"""Generate harder file-only scout queries for prism-llama.cpp with hygiene filters.

Non-overfit rules (applied before any model scores):
- Drop TODO/FIXME/banner-only/low-signal comments
- Require a behavioral claim (verb + enough content tokens)
- Prefer unique-ish definition anchors (symbol body substance)
- Strip symbol names from queries (hardness)
- Dedupe by file+symbol; diversify directories
- Optional: mark C/C++ header/impl dual gold via gold_files
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SKIP_DIR = {
    ".git",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    ".venv",
    "vendor",
    "third_party",
    "deps",
    "ggml-cuda",
    "ggml-metal",
    "ggml-vulkan",
    "ggml-sycl",
    "ggml-opencl",
    "ggml-hip",
    "ggml-musa",
    "ggml-cann",
    "ggml-zdnn",
    "ggml-rpc",
    "ggml-hexagon",
    "ggml-blas",
    "ggml-cpu",
    "ggml-webgpu",
}
GENERIC_NAMES = {
    "main",
    "init",
    "free",
    "new",
    "get",
    "set",
    "read",
    "write",
    "open",
    "close",
    "run",
    "start",
    "stop",
    "test",
    "setup",
    "teardown",
    "encode",
    "decode",
    "create",
    "destroy",
    "update",
    "process",
    "handle",
    "callback",
    "error",
    "log",
    "print",
    "debug",
    "info",
    "build",
    "parse",
    "format",
    "clear",
    "reset",
    "copy",
    "move",
    "size",
    "empty",
    "data",
    "type",
    "value",
    "name",
    "load",
    "save",
}

NOISE_PAT = re.compile(
    r"^(TODO|FIXME|XXX|HACK|NOTE)\b|"
    r"^/+|"  # banner ////
    r"^=+|"
    r"^-{3,}|"
    r"^copyright\b|"
    r"^license\b|"
    r"^spdx\b|"
    r"^#include\b|"
    r"^using\b|"
    r"^namespace\b",
    re.I,
)

CONTENT_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
VERBISH = re.compile(
    r"\b(find|return|compute|convert|parse|load|save|build|create|destroy|allocate|"
    r"free|check|validate|normalize|serialize|deserialize|encode|decode|render|"
    r"schedule|dispatch|apply|update|initialize|configure|resolve|match|filter|"
    r"sort|merge|split|append|remove|insert|lookup|search|map|bind|open|close|"
    r"write|read|send|recv|handle|process|generate|collect|format|print|log|"
    r"implements?|responsible|used to|ensures?|stores?|keeps?|avoids?)\b",
    re.I,
)


def rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def too_generic(name: str) -> bool:
    n = name.lower()
    if n in GENERIC_NAMES:
        return True
    if len(name) < 5:
        return True
    if name.startswith("_") and len(name) < 12:
        return True
    return False


def query_signal_ok(q: str) -> bool:
    q = q.strip()
    if len(q) < 40:
        return False
    if NOISE_PAT.search(q.strip()):
        # allow if there's substantial content after TODO-like prefix? still reject pure noise
        body = re.sub(r"^(TODO|FIXME|XXX|HACK|NOTE)\s*:?\s*", "", q, flags=re.I)
        if len(CONTENT_TOKEN.findall(body)) < 6:
            return False
    # banner-only
    if re.fullmatch(r"[/\-=*_#\s]+", q):
        return False
    tokens = CONTENT_TOKEN.findall(q)
    if len(tokens) < 6:
        return False
    # must look like a behavioral ask
    if not VERBISH.search(q) and not q.lower().startswith("find"):
        return False
    # reject if mostly punctuation / path noise
    alpha = sum(c.isalpha() for c in q)
    if alpha / max(len(q), 1) < 0.55:
        return False
    return True


def scrub_symbol(q: str, name: str) -> str:
    q = re.sub(rf"\b{re.escape(name)}\b", "this component", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" :,-")
    return q


def normalize_query(q: str, name: str) -> str | None:
    q = scrub_symbol(q, name)
    q = re.sub(r"\s+", " ", q).strip()
    if not q.lower().startswith("find"):
        if q:
            q = f"Find the code that {q[0].lower() + q[1:]}" if q[0].isupper() else f"Find the code that {q}"
    q = scrub_symbol(q, name)
    q = q[:220].strip()
    if not query_signal_ok(q):
        return None
    if re.search(rf"\b{re.escape(name)}\b", q, re.I):
        return None
    return q


def extract_py(root: Path) -> list[dict]:
    out: list[dict] = []
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIR or part.startswith(".") for part in p.parts):
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            if too_generic(name):
                continue
            try:
                seg = ast.get_source_segment(src, node) or ""
            except Exception:
                seg = ""
            if len(seg) < 120:
                continue
            doc = (ast.get_docstring(node) or "").strip()
            if doc:
                first = doc.split("\n")[0].strip()
                q = normalize_query(first, name)
                hardness = "doc"
            else:
                body_lines = [
                    ln.strip()
                    for ln in seg.splitlines()[1:10]
                    if ln.strip() and not ln.strip().startswith("#")
                ]
                hint = " ".join(body_lines)[:160]
                hint = re.sub(r"[^A-Za-z0-9_.,;:()\[\]\{\} \-+/*=<>]", " ", hint)
                hint = re.sub(r"\s+", " ", hint).strip()
                q = normalize_query(f"implements: {hint}", name)
                hardness = "body"
            if not q:
                continue
            out.append(
                {
                    "query": q,
                    "file": rel(root, p),
                    "symbol": name,
                    "lang": "py",
                    "start_line": int(getattr(node, "lineno", 1) or 1),
                    "hardness": hardness,
                    "body_len": len(seg),
                }
            )
    return out


def extract_c_family(root: Path) -> list[dict]:
    out: list[dict] = []
    func_re = re.compile(
        r"^(?P<head>[\w\s\*:&<>,~]+?)\b(?P<name>[A-Za-z_][A-Za-z0-9_]{4,})\s*\((?P<args>[^;]*)\)\s*(?:const\s*)?\{",
        re.M,
    )
    skip = {"if", "for", "while", "switch", "return", "sizeof", "catch", "else", "do", "try"}
    for pat in ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp"):
        for p in root.rglob(pat):
            if any(part in SKIP_DIR or part.startswith(".") for part in p.parts):
                continue
            rp = rel(root, p)
            if not any(rp.startswith(pref) for pref in ("src/", "common/", "include/", "tools/", "examples/")):
                continue
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(src) > 350_000:
                continue
            for m in func_re.finditer(src):
                name = m.group("name")
                if name in skip or too_generic(name):
                    continue
                if name.startswith("ggml_"):
                    continue
                start = src[: m.start()].count("\n") + 1
                window = src[m.start() : m.start() + 700]
                if window.count(";") < 3:
                    continue
                # comments immediately above
                pre = src[max(0, m.start() - 400) : m.start()]
                comments = []
                for line in reversed(pre.splitlines()):
                    s = line.strip()
                    if s.startswith("//"):
                        comments.append(s[2:].strip())
                    elif s.startswith("/*") or s.startswith("*"):
                        comments.append(re.sub(r"^/\*+|\*+/|^\*", "", s).strip())
                    elif s == "" or s.startswith("#"):
                        continue
                    else:
                        break
                ctext = " ".join(reversed(comments)).strip()
                ctext = re.sub(r"\s+", " ", ctext)
                if ctext and not NOISE_PAT.search(ctext) and len(CONTENT_TOKEN.findall(ctext)) >= 5:
                    q = normalize_query(ctext[:180], name)
                    hardness = "comment"
                else:
                    # body-derived but require verbs/identifiers density
                    body = re.sub(rf"\b{re.escape(name)}\b", "this function", window, flags=re.I)
                    body = re.sub(r"[^A-Za-z0-9_.,;:() \-+/*=<>]", " ", body)
                    body = re.sub(r"\s+", " ", body).strip()
                    q = normalize_query(f"implements: {body[:150]}", name)
                    hardness = "body"
                if not q:
                    continue
                out.append(
                    {
                        "query": q,
                        "file": rp,
                        "symbol": name,
                        "lang": p.suffix.lstrip("."),
                        "start_line": start,
                        "hardness": hardness,
                        "body_len": len(window),
                    }
                )
    return out


def add_dual_gold(items: list[dict], root: Path) -> None:
    """If foo.cpp is gold and foo.h exists (or reverse), record gold_files pair."""
    for it in items:
        f = Path(it["file"])
        stem = f.stem
        parent = root / f.parent
        pair = None
        if f.suffix in {".c", ".cc", ".cpp"}:
            for ext in (".h", ".hpp"):
                cand = parent / f"{stem}{ext}"
                if cand.exists():
                    pair = rel(root, cand)
                    break
        elif f.suffix in {".h", ".hpp"}:
            for ext in (".c", ".cc", ".cpp"):
                cand = parent / f"{stem}{ext}"
                if cand.exists():
                    pair = rel(root, cand)
                    break
        gold = [it["file"]]
        if pair and pair not in gold:
            gold.append(pair)
        it["gold_files"] = gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / ".." / "prism-llama.cpp")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "queries" / "scout_prism_hard_file.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.repo.resolve()
    rng = random.Random(args.seed)

    cands = extract_py(root) + extract_c_family(root)
    # dedupe file+symbol keep longest body
    uniq: dict[tuple[str, str], dict] = {}
    for c in cands:
        k = (c["file"], c["symbol"])
        if k not in uniq or c.get("body_len", 0) > uniq[k].get("body_len", 0):
            uniq[k] = c
    cands = list(uniq.values())
    print(f"[gen] raw after hygiene: {len(cands)}")

    # score for sampling: prefer comment/doc + sibling pressure + non-py
    by_dir: dict[str, list] = defaultdict(list)
    for c in cands:
        by_dir[str(Path(c["file"]).parent)].append(c)
    for c in cands:
        sibs = len(by_dir[str(Path(c["file"]).parent)])
        hard_bonus = {"doc": 3.0, "comment": 2.5, "body": 1.0}.get(c["hardness"], 1.0)
        lang_bonus = 0.0 if c["lang"] == "py" else 1.2
        c["score"] = hard_bonus + min(sibs, 10) * 0.25 + lang_bonus + min(c.get("body_len", 0), 800) / 800.0

    cands.sort(key=lambda c: (-c["score"], c["file"], c["symbol"]))

    picked: list[dict] = []
    seen_files: set[str] = set()
    lang_count: dict[str, int] = defaultdict(int)
    for c in cands:
        if len(picked) >= args.n:
            break
        if c["file"] in seen_files and rng.random() > 0.15:
            continue
        # keep py minority on this repo
        if c["lang"] == "py" and lang_count["py"] >= max(4, args.n // 8):
            continue
        item = {
            "query": c["query"],
            "file": c["file"],
            "symbol": c["symbol"],
            "lang": c["lang"],
            "start_line": c["start_line"],
            "hardness": c["hardness"],
        }
        picked.append(item)
        seen_files.add(c["file"])
        lang_count[c["lang"]] += 1

    if len(picked) < args.n:
        have = {(x["file"], x["symbol"]) for x in picked}
        for c in cands:
            if (c["file"], c["symbol"]) in have:
                continue
            picked.append(
                {
                    "query": c["query"],
                    "file": c["file"],
                    "symbol": c["symbol"],
                    "lang": c["lang"],
                    "start_line": c["start_line"],
                    "hardness": c["hardness"],
                }
            )
            if len(picked) >= args.n:
                break

    rng.shuffle(picked)
    picked = picked[: args.n]
    add_dual_gold(picked, root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(picked, indent=2) + "\n", encoding="utf-8")
    print(f"[gen] wrote {args.out} n={len(picked)}")
    print("[gen] lang", dict(Counter := __import__("collections").Counter(x["lang"] for x in picked)))
    print("[gen] hardness", dict(__import__("collections").Counter(x["hardness"] for x in picked)))
    print("[gen] dual-gold", sum(1 for x in picked if len(x.get("gold_files") or []) > 1))
    for x in picked[:5]:
        print("-", x["lang"], x["file"], x["symbol"], x["hardness"])
        print(" ", x["query"][:120])
        if len(x.get("gold_files") or []) > 1:
            print("  gold_files", x["gold_files"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
