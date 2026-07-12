"""Scout pipeline: Qwen3-Embedding-0.6B -> BGE-Reranker-v2-m3 -> scout LLM."""
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

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            self._embedder.max_seq_length = 512
        return self._embedder

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print("[scout] Loading BGE-Reranker-v2-m3 (first use)...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        return self._reranker

    def index(self, glob_pattern: str = "**/*.py", force: bool = False):
        import chromadb
        model = self._get_embedder()
        print(f"[scout] Indexing {self.repo} ...")

        chunks = []
        for f in sorted(self.repo.glob(glob_pattern)):
            if ".venv" in str(f) or "__pycache__" in str(f):
                continue
            try:
                source = f.read_text()
            except Exception:
                continue
            rel = str(f.relative_to(self.repo))
            for chunk in _extract_chunks(source, rel):
                chunks.append(chunk)

        print(f"[scout] Extracted {len(chunks)} chunks")
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        client = chromadb.PersistentClient(path=str(self.repo / ".scout_index"))
        if force:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file": c["file"], "symbol": c["symbol"], "start_line": c["start_line"]}
                      for c in chunks]
        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"[scout] Indexed {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10, rerank_top_n: int = 5) -> list[dict]:
        import chromadb
        model = self._get_embedder()
        reranker = self._get_reranker()

        instructed = (
            "Instruct: Given a code search query, retrieve relevant code snippets "
            "that match the query\nQuery: " + query
        )

        client = chromadb.PersistentClient(path=str(self.repo / ".scout_index"))
        self._collection = client.get_collection(self.collection_name)
        qe = model.encode([instructed], prompt_name="query").tolist()
        results = self._collection.query(query_embeddings=qe, n_results=top_k)

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "symbol": results["metadatas"][0][i]["symbol"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        pairs = [(query, c["text"]) for c in chunks]
        scores = reranker.predict(pairs).tolist()
        if not isinstance(scores, list):
            scores = [scores]
        for c, score in zip(chunks, scores):
            c["rerank_score"] = float(score)

        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        return chunks[:rerank_top_n]

    def build_scout_prompt(self, query: str, chunks: list[dict], max_chars_per_file: int = 15000) -> str:
        """Build scout prompt. Truncates each file to avoid OOM on large codebases."""
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
                        content = f"[truncated, lines {start+1}-{start+len(window)}]\n" + "\n".join(window)
                        if start + 200 < len(lines):
                            content += f"\n[{len(lines) - start - 200} more lines]"
                except Exception:
                    file_contents[fname] = f"[could not read {fname}]"
                else:
                    file_contents[fname] = content

        parts = [f"// FILE: {fname}\n{content}" for fname, content in file_contents.items()]
        context = "\n\n".join(parts)

        return textwrap.dedent(f"""\
        You are a code scout. Your job is to find the right files and line ranges
        based on the prompt's request. Return the file path, symbol name, line range,
        and exact code.

        Respond in this format:

        FILE: path/to/file.py
        SYMBOL: function_name
        LINES: 42-67
        ```language
        [exact code]
        ```

        ## Codebase

        {context}

        ## Query

        {query}
        """)


def _extract_chunks(source: str, filepath: str) -> list[dict]:
    chunks = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                cs = ast.get_source_segment(source, node)
            except Exception:
                continue
            if not cs:
                continue
            docstring = ast.get_docstring(node)
            preview = cs
            if docstring:
                lines = cs.split("\n")
                sig_lines = []
                for line in lines:
                    sig_lines.append(line)
                    if docstring in line or '"""' in line:
                        break
                preview = "\n".join(sig_lines[:10])
            chunks.append({"file": filepath, "symbol": node.name, "start_line": node.lineno, "text": preview})
    return chunks


def _extract_code_block(response: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def _parse_file_ref(response: str) -> str | None:
    m = re.search(r"(?:FILE:\s*|in\s+)([\w/\-_.]+\.\w+)", response, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"`([\w/\-_.]+\.\w+)`", response)
    return m.group(1) if m else None
