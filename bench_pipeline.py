"""End-to-end scout pipeline benchmark.

Runs the full embedder -> reranker -> scout LLM pipeline on a set of queries.
Writes a checkpoint after every query so a server crash mid-run does not wipe
progress (this bit us on the MTPLX 27B 29/30 runs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_client import chat as _bench_chat
from bench_client import list_models, make_client
from scout_pipeline import ScoutPipeline, _extract_code_block, _parse_file_ref


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pipeline] warning: could not read checkpoint {path}: {exc}")
        return []
    results = data.get("results")
    return results if isinstance(results, list) else []


def _write_output(path: Path, summary: dict, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _running_summary(
    *,
    model: str,
    top_n: int,
    base_url: str,
    results: list[dict],
    total_time: float,
    wall: float,
    n_target: int,
    status: str,
    file_only: bool = False,
) -> dict:
    n = len(results)
    file_matches = sum(1 for r in results if r.get("file_ok"))
    code_matches = sum(1 for r in results if r.get("code_ok"))
    total_bleu = sum(float(r.get("bleu") or 0.0) for r in results)
    return {
        "benchmark": "scout_pipeline",
        "model": model,
        "n_samples": n,
        "n_target": n_target,
        "top_n": top_n,
        "file_match_rate": round(file_matches / n, 4) if n else 0.0,
        "code_match_rate": round(code_matches / n, 4) if n else 0.0,
        "avg_bleu": round(total_bleu / n, 4) if n else 0.0,
        "avg_scout_time_s": round(total_time / n, 1) if n else 0.0,
        "wall_time_sec": wall,
        "base_url": base_url,
        "status": status,
        "file_only": file_only,
    }


def _chat_with_retries(client, model, messages, *, max_tokens, temperature, think_off, no_think,
                       retries: int = 3, backoff_s: float = 5.0, base_url: str | None = None):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if base_url and not _server_healthy(base_url):
                print(f"  [pipeline] server unhealthy before attempt {attempt}; waiting...")
                if not _wait_for_server(base_url, timeout_s=180.0):
                    raise RuntimeError(f"server not healthy at {base_url}")
            return _bench_chat(
                client,
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                think_off=think_off,
                no_think=no_think,
            )
        except Exception as exc:
            last_exc = exc
            print(f"  [pipeline] chat error attempt {attempt}/{retries}: {type(exc).__name__}: {exc}")
            if attempt < retries:
                # Give MTPLX time to recover / be restarted externally.
                time.sleep(backoff_s * attempt)
                if base_url:
                    _wait_for_server(base_url, timeout_s=120.0)
    assert last_exc is not None
    raise last_exc

def _server_healthy(base_url: str, timeout_s: float = 5.0) -> bool:
    """Best-effort OpenAI-compat health probe (models list)."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _wait_for_server(base_url: str, *, timeout_s: float = 180.0, poll_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _server_healthy(base_url):
            return True
        time.sleep(poll_s)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=5, help="Files to pass to scout LLM (5 recovers near-miss siblings like sdpa_2pass_paged)")
    ap.add_argument("--think-off", action="store_true",
                    help="Inject <|think_off|> in user messages (froggeric hard switch)")
    ap.add_argument("--no-think", action="store_true",
                    help="Append /no_think to user messages (Qwen soft switch, template-agnostic)")
    ap.add_argument("--no-graphify", action="store_true",
                    help="Skip graphify pre-check, use embedder+reranker only")
    ap.add_argument("--file-only", action="store_true",
                    help="Score file match only (no code/BLEU); shorter prompt and max_tokens")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from --out if it already contains partial results")
    ap.add_argument("--sleep", type=float, default=10.0,
                    help="Seconds to sleep between queries (MTPLX stability drain)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Chat retries per query on transport/server errors")
    args = ap.parse_args()

    client = make_client(base_url=args.base_url)
    models = list_models(client)
    if args.model == "auto":
        if len(models) != 1:
            print(f"ERROR: --model auto requires exactly one served model, got {models}", file=sys.stderr)
            return 2
        args.model = models[0]

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)
    if args.n_samples:
        queries = queries[: args.n_samples]

    out_path = Path(args.out)
    results: list[dict] = _load_checkpoint(out_path) if args.resume else []
    start_i = len(results)
    if start_i:
        print(f"[pipeline] resuming from {out_path} with {start_i}/{len(queries)} done")
    if start_i > len(queries):
        print(f"ERROR: checkpoint has {start_i} results but only {len(queries)} queries", file=sys.stderr)
        return 2

    print(f"[pipeline] {len(queries)} queries, scout={args.model}, top_n={args.top_n}, "
          f"graphify={not args.no_graphify}, file_only={args.file_only}, sleep={args.sleep}s")

    pipeline = ScoutPipeline(args.repo)
    file_matches = sum(1 for r in results if r.get("file_ok"))
    code_matches = sum(1 for r in results if r.get("code_ok"))
    total_time = sum(float(r.get("elapsed") or 0.0) for r in results)
    total_bleu = sum(float(r.get("bleu") or 0.0) for r in results)

    t0_total = time.time()
    fatal: str | None = None

    for i in range(start_i, len(queries)):
        q = queries[i]
        try:
            chunks = pipeline.retrieve(
                q["query"],
                top_k=20 if args.file_only else 10,
                rerank_top_n=args.top_n,
                use_graphify=not args.no_graphify,
            )
            if args.file_only:
                prompt = pipeline.build_file_only_prompt(q["query"], chunks)
                max_toks = min(args.max_tokens, 64)
            else:
                prompt = pipeline.build_scout_prompt(q["query"], chunks)
                max_toks = args.max_tokens

            t0 = time.time()
            resp = _chat_with_retries(
                client,
                args.model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_toks,
                temperature=0.0,
                think_off=args.think_off,
                no_think=args.no_think,
                retries=args.retries,
                base_url=args.base_url,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            response_text = ""
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                response_text = resp.choices[0].message.content

            candidate_files = []
            seen = set()
            for c in chunks:
                f = c.get("file")
                if f and f not in seen:
                    seen.add(f)
                    candidate_files.append(f)
            file_ref = _parse_file_ref(response_text, candidates=candidate_files)

            gt_file = q["file"]
            # Accept exact rel path or basename match (multi-lang repos).
            file_ok = False
            if file_ref is not None:
                fr = Path(file_ref)
                gt = Path(gt_file)
                file_ok = (
                    fr.name == gt.name
                    or str(fr).replace("\\","/") == str(gt).replace("\\","/")
                    or str(fr).replace("\\","/").endswith("/" + str(gt).replace("\\","/"))
                    or str(gt).replace("\\","/").endswith("/" + str(fr).replace("\\","/"))
                )

            if args.file_only:
                extracted = ""
                bleu = 0.0
                code_ok = False
            else:
                extracted = _extract_code_block(response_text)
                gt_code = q.get("code", "")
                try:
                    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

                    bleu = sentence_bleu(
                        [gt_code.split()],
                        extracted.split(),
                        smoothing_function=SmoothingFunction().method1,
                    )
                except Exception:
                    bleu = 1.0 if extracted.strip() == gt_code.strip() else 0.0
                code_ok = bleu >= 0.8
            if file_ok:
                file_matches += 1
            if code_ok:
                code_matches += 1
            total_bleu += bleu

            results.append(
                {
                    "query": q["query"][:100],
                    "gt_file": gt_file,
                    "file_ref": file_ref,
                    "file_ok": file_ok,
                    "bleu": round(float(bleu), 4),
                    "code_ok": code_ok,
                    "elapsed": round(elapsed, 1),
                    "response": response_text[:300],
                }
            )
        except Exception as exc:
            fatal = f"query {i + 1}/{len(queries)} failed: {type(exc).__name__}: {exc}"
            print(f"  [pipeline] FATAL {fatal}")
            traceback.print_exc()
            results.append(
                {
                    "query": q["query"][:100],
                    "gt_file": q.get("file"),
                    "file_ref": None,
                    "file_ok": False,
                    "bleu": 0.0,
                    "code_ok": False,
                    "elapsed": 0.0,
                    "response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        wall = round(time.time() - t0_total, 1)
        status = "partial" if (fatal or len(results) < len(queries)) else "complete"
        summary = _running_summary(
            model=args.model,
            top_n=args.top_n,
            base_url=args.base_url,
            results=results,
            total_time=total_time,
            wall=wall,
            n_target=len(queries),
            status=status,
            file_only=args.file_only,
        )
        # Always checkpoint after each query so a later crash keeps prior work.
        _write_output(out_path, summary, results)

        done = len(results)
        fm = file_matches / done if done else 0.0
        cm = code_matches / done if done else 0.0
        last = results[-1]
        if args.file_only:
            print(
                f"  [{done}/{len(queries)}] file={fm:.1%} "
                f"({last.get('elapsed', 0):.0f}s) ref={last.get('file_ref')} ok={last.get('file_ok')} -> {out_path}"
            )
        else:
            print(
                f"  [{done}/{len(queries)}] file={fm:.1%} code={cm:.1%} "
                f"({last.get('elapsed', 0):.0f}s) -> {out_path}"
            )

        if fatal:
            print(f"\n[pipeline] stopped early after checkpointing {done} results")
            print(json.dumps(summary, indent=2))
            return 1

        if i + 1 < len(queries) and args.sleep > 0:
            # Drain server post-commit work between queries. MTPLX is especially
            # prone to dying under back-to-back long generations without this.
            time.sleep(args.sleep)

    wall = round(time.time() - t0_total, 1)
    summary = _running_summary(
        model=args.model,
        top_n=args.top_n,
        base_url=args.base_url,
        results=results,
        total_time=total_time,
        wall=wall,
        n_target=len(queries),
        status="complete",
        file_only=args.file_only,
    )
    _write_output(out_path, summary, results)
    print(f"\n[pipeline] done in {wall}s")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
