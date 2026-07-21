"""Scout pipeline: Qwen3-Embedding-0.6B -> BGE-Reranker-v2-m3 -> scout LLM (with graphify pre-check)."""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path


class ScoutPipeline:
    def __init__(self, repo_path: str, collection_name: str = "scout_code"):
        self.repo = Path(repo_path).expanduser().resolve()
        self.collection_name = collection_name
        self._embedder = None
        self._reranker = None
        self._collection = None
        self._graphify_available = None  # tri-state: None=unchecked, True/False
        self._graphify_graph = None  # cached networkx graph
        self._graphify_mtime = None

    def _check_graphify(self) -> bool:
        """Check if graphify has been run on this repo."""
        if self._graphify_available is None:
            graph_path = self.repo / "graphify-out" / "graph.json"
            self._graphify_available = graph_path.exists()
        return self._graphify_available

    def _load_graphify_graph(self):
        """Load graphify-out/graph.json once; reload only if mtime changes."""
        import json
        import networkx as nx
        from networkx.readwrite import json_graph

        graph_path = self.repo / "graphify-out" / "graph.json"
        try:
            mtime = graph_path.stat().st_mtime
        except OSError:
            self._graphify_available = False
            self._graphify_graph = None
            return None
        if self._graphify_graph is not None and self._graphify_mtime == mtime:
            return self._graphify_graph
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            G = json_graph.node_link_graph(data, edges="links")
        except Exception:
            self._graphify_graph = None
            return None
        self._graphify_graph = G
        self._graphify_mtime = mtime
        return G

    def retrieve_graphify(self, query: str) -> list[dict] | None:
        """BFS graph traversal from keyword-matched start nodes.

        Uncapped novel-file merge into the embedder candidate pool, then BGE
        rerank. This is the booster from the MTPLX 27B 16/16 file-perfect
        partial run (graphify ON, top_n=3, depth-2 BFS).
        Returns None if graphify unavailable or no matches found.
        """
        if not self._check_graphify():
            return None

        G = self._load_graphify_graph()
        if G is None:
            return None

        # Vocab expansion: extract meaningful tokens from the query
        terms = set()
        for word in re.findall(r"[^\W\d_]+", query, re.UNICODE):
            w = word.lower()
            if len(w) >= 3:
                terms.add(w)
        if not terms:
            return None

        # Find start nodes by term overlap with node labels
        scored = []
        for nid, ndata in G.nodes(data=True):
            label = ndata.get("label", "").lower()
            s = sum(1 for t in terms if t in label)
            if s > 0:
                scored.append((s, nid))
        scored.sort(reverse=True)

        if not scored:
            return None

        # BFS from top start nodes, depth 2 — broad context
        start_nodes = [nid for _, nid in scored[:5]]
        subgraph_nodes = set(start_nodes)
        frontier = set(start_nodes)
        for _ in range(2):
            next_frontier = set()
            for n in frontier:
                for neighbor in G.neighbors(n):
                    if neighbor not in subgraph_nodes:
                        next_frontier.add(neighbor)
                        subgraph_nodes.add(neighbor)
            frontier = next_frontier

        # Collect source files from the subgraph, scored by term relevance
        file_scores: dict[str, int] = {}
        best_symbol: dict[str, str] = {}
        for nid in subgraph_nodes:
            ndata = G.nodes[nid]
            sf = ndata.get("source_file", "")
            if not sf:
                continue
            label = ndata.get("label", "").lower()
            s = sum(1 for t in terms if t in label)
            if sf not in file_scores or s > file_scores[sf]:
                file_scores[sf] = s
                best_symbol[sf] = ndata.get("label", sf)

        if not file_scores:
            return None

        results = []
        for sf, s in sorted(file_scores.items(), key=lambda x: -x[1]):
            try:
                text = (self.repo / sf).read_text()
            except Exception:
                text = f"[graphify: {best_symbol.get(sf, sf)}]"
            results.append({
                "file": sf,
                "symbol": best_symbol.get(sf, sf),
                "text": text[:3000],
                "score": s,
                "source": "graphify",
            })

        return results if results else None

    def retrieve(self, query: str, top_k: int = 10, rerank_top_n: int = 5,
                 use_graphify: bool = True) -> list[dict]:
        """Retrieve relevant code chunks.

        Strategy:
        1. Embedder → chromadb top-k (always)
        2. Graphify BFS traversal → candidate files (if available)
        3. Merge both sources, deduplicate by file
        4. BGE reranker scores all merged candidates
        5. Return top rerank_top_n by reranker score
        """
        import chromadb
        model = self._get_embedder()
        reranker = self._get_reranker()

        # 1. Embedder → chromadb
        instructed = "Instruct: Given a code search query, retrieve relevant code snippets that match the query\nQuery: " + query
        client = chromadb.PersistentClient(path=str(self.repo / ".scout_index"))
        self._collection = client.get_collection(self.collection_name)
        qe = model.encode([instructed], prompt_name="query").tolist()
        results = self._collection.query(query_embeddings=qe, n_results=top_k)

        chunks: list[dict] = []
        seen_files: set[str] = set()
        for i in range(len(results["ids"][0])):
            f = results["metadatas"][0][i]["file"]
            if f not in seen_files:
                seen_files.add(f)
                chunks.append({
                    "id": results["ids"][0][i],
                    "file": f,
                    "symbol": results["metadatas"][0][i]["symbol"],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "source": "embeddings",
                })

        # 2. Graphify BFS traversal → merge into candidate set
        if use_graphify:
            graphify_results = self.retrieve_graphify(query)
            if graphify_results:
                added = 0
                for gr in graphify_results:
                    if gr["file"] not in seen_files:
                        seen_files.add(gr["file"])
                        chunks.append(gr)
                        added += 1
                if added:
                    print(f"  [scout] graphify +{added} novel files")

        # 3+4. Reranker scores all merged candidates → sort → top-n
        pairs = [(query, c["text"][:2000]) for c in chunks]
        scores = reranker.predict(pairs).tolist()
        if not isinstance(scores, list):
            scores = [scores]
        for c, score in zip(chunks, scores):
            c["rerank_score"] = float(score)

        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        return chunks[:rerank_top_n]

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"torch_dtype": "auto"})
            self._embedder.max_seq_length = 512
        return self._embedder

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print("[scout] Loading BGE-Reranker-v2-m3...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        return self._reranker

    def index(self, glob_pattern: str = "**/*.py", force: bool = False):
        import chromadb
        model = self._get_embedder()
        chunks = []
        for f in sorted(self.repo.glob(glob_pattern)):
            if ".venv" in str(f) or "__pycache__" in str(f):
                continue
            try: source = f.read_text()
            except Exception: continue
            for chunk in _extract_chunks(source, str(f.relative_to(self.repo))):
                chunks.append(chunk)
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
        client = chromadb.PersistentClient(path=str(self.repo / ".scout_index"))
        if force:
            try: client.delete_collection(self.collection_name)
            except Exception: pass
        self._collection = client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file": c["file"], "symbol": c["symbol"], "start_line": c["start_line"]} for c in chunks]
        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"[scout] Indexed {len(chunks)} chunks")

    def build_scout_prompt(self, query: str, chunks: list[dict], max_chars_per_file: int = 8000) -> str:
        """Build a scout prompt that keeps the model inside the candidate set.

        Fable-Fusion analysis: retrieval already ranks the GT file #1 on most
        file-misses; the model then drifts to hub facades (server/openai.py) or
        emits paths that were never candidates. Fix:
        - show an explicit ranked candidate list with rerank scores
        - require FILE to be copied from that list
        - prefer definition modules / matching symbols over re-exports
        - feed symbol-centered windows, not giant hub file dumps
        """
        # Preserve retrieval order (already rerank-sorted).
        ordered_files: list[str] = []
        best_chunk: dict[str, dict] = {}
        for c in chunks:
            fname = c["file"]
            if fname not in best_chunk:
                ordered_files.append(fname)
                best_chunk[fname] = c

        candidate_lines = []
        for i, fname in enumerate(ordered_files, 1):
            c = best_chunk[fname]
            score = c.get("rerank_score")
            score_s = f"{float(score):.3f}" if score is not None else "n/a"
            sym = c.get("symbol") or "?"
            src = c.get("source") or "retr"
            candidate_lines.append(f"{i}. {fname}  (score={score_s}, symbol={sym}, via={src})")
        candidate_block = "\n".join(candidate_lines) if candidate_lines else "(none)"

        sections: list[str] = []
        for i, fname in enumerate(ordered_files, 1):
            c = best_chunk[fname]
            content = self._file_window_for_chunk(fname, c, max_chars_per_file)
            # Lead with the retrieved symbol snippet when present — denser signal
            # than a raw file dump for definition-vs-facade disambiguation.
            snippet = (c.get("text") or "").strip()
            head = ""
            if snippet:
                head = f"// retrieved symbol preview ({c.get('symbol', '?')}):\n{snippet[:1500]}\n\n"
            sections.append(
                f"// CANDIDATE {i}/{len(ordered_files)}: {fname}\n"
                f"// rerank_score={c.get('rerank_score', 'n/a')} source={c.get('source', '?')}\n"
                f"{head}{content}"
            )
        ctx = "\n\n".join(sections)

        return textwrap.dedent(f"""\
        You are a code scout. Pick the single best definition site for the query.

        Hard rules:
        1. FILE must be copied EXACTLY from the CANDIDATES list below. Never invent a path.
        2. Prefer the module that DEFINES the behavior (class/def body, docstring match),
           not a facade/re-export/wrapper that merely calls or imports it.
        3. Prefer higher rerank_score when two candidates are equally plausible.
        4. Prefer a smaller focused module over a large hub file (e.g. server/openai.py)
           unless the hub file is truly the only definition.
        5. If several candidates match, pick the one whose symbol/docstring best matches
           the query wording.

        Respond in exactly this shape:
        FILE: path/to/file.py
        SYMBOL: name
        LINES: start-end
        ```language
        code
        ```

        ## CANDIDATES (choose FILE from this list only)
        {candidate_block}

        ## Candidate code
        {ctx}

        ## Query
        {query}
        """)

    def _file_window_for_chunk(self, fname: str, chunk: dict, max_chars: int) -> str:
        """Return a symbol-centered window instead of a giant leading file dump."""
        try:
            raw = (self.repo / fname).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return f"[err: {fname}]"

        lines = raw.splitlines()
        if not lines:
            return "[empty]"

        # Center on the retrieved symbol when we know its line.
        start_line = int(chunk.get("start_line") or 1)
        # 1-based start_line from indexer; keep a generous definition window.
        lo = max(0, start_line - 1 - 20)
        hi = min(len(lines), lo + 220)
        # If still huge on chars, tighten.
        window = lines[lo:hi]
        text = f"[lines {lo + 1}-{lo + len(window)} of {len(lines)}]\n" + "\n".join(window)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text



def _extract_chunks(source: str, filepath: str) -> list[dict]:
    chunks = []
    try: tree = ast.parse(source)
    except SyntaxError: return chunks
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try: cs = ast.get_source_segment(source, node)
            except Exception: continue
            if not cs: continue
            docstring = ast.get_docstring(node)
            preview = cs
            if docstring:
                lines = cs.split("\n"); sl = []
                for l in lines:
                    sl.append(l)
                    if docstring in l or '"""' in l: break
                preview = "\n".join(sl[:10])
            chunks.append({"file": filepath, "symbol": node.name, "start_line": node.lineno, "text": preview})
    return chunks


def _extract_code_block(response: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def _parse_file_ref(response: str, candidates: list[str] | None = None) -> str | None:
    """Extract FILE path; if candidates given, snap to the best candidate match.

    Models sometimes emit a hub path that was only mentioned in prose, or a
    basename/absolute variant. Prefer an exact/suffix match inside candidates.
    """
    m = re.search(r"(?:FILE:\s*|in\s+)([\w/\-_.]+\.\w+)", response, re.IGNORECASE)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"`([\w/\-_.]+\.\w+)`", response)
        raw = m.group(1) if m else None
    if raw is None:
        return None
    if not candidates:
        return raw

    # Exact match first.
    if raw in candidates:
        return raw
    # Suffix / basename match against candidates (stable order = rerank order).
    raw_name = Path(raw).name
    for c in candidates:
        if c == raw or c.endswith("/" + raw) or raw.endswith("/" + c) or Path(c).name == raw_name:
            return c
    # Model invented a non-candidate path: fall back to top-ranked candidate
    # only if the response clearly failed the hard rule. Safer for file@1
    # when retrieval is trusted and the model drifted to a hub path.
    # Do NOT silently remap if nothing matches basename — return raw for metrics.
    return raw
