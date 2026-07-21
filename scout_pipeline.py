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

    def build_scout_prompt(self, query: str, chunks: list[dict], max_chars_per_file: int = 15000) -> str:
        file_contents = {}
        for c in chunks:
            fname = c["file"]
            if fname not in file_contents:
                try:
                    content = (self.repo / fname).read_text()
                    if len(content) > max_chars_per_file:
                        start = max(0, c.get("start_line", 1) - 30)
                        lines = content.split("\n")
                        window = lines[start:start + 200]
                        content = f"[lines {start+1}-{start+len(window)}]\n" + "\n".join(window)
                        if start + 200 < len(lines): content += f"\n[{len(lines) - start - 200} more lines]"
                except Exception: content = f"[err: {fname}]"
                file_contents[fname] = content
        ctx = "\n\n".join(f"// FILE: {f}\n{c}" for f, c in file_contents.items())
        return textwrap.dedent(f"""\
        You are a code scout. Find the right files and line ranges.
        Respond: FILE: path/to/file.py
        SYMBOL: name
        LINES: start-end
        ```language
        code
        ```
        ## Codebase
        {ctx}
        ## Query
        {query}
        """)


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


def _parse_file_ref(response: str) -> str | None:
    m = re.search(r"(?:FILE:\s*|in\s+)([\w/\-_.]+\.\w+)", response, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"`([\w/\-_.]+\.\w+)`", response)
    return m.group(1) if m else None
