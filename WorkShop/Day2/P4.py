import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict

from dotenv import load_dotenv
load_dotenv()


# ── PDF loading ──────────────────────────────────────────────────────────────

def load_pdf(path: str) -> List[Tuple[str, Dict]]:
    """
    Load a PDF file and return a list of (text, metadata) tuples — one per page.
    Requires pypdf: pip install pypdf
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    filename = Path(path).name

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((
                text.strip(),
                {
                    "source":   filename,
                    "page":     page_num + 1,   # 1-indexed for human readability
                    "filepath": str(path),
                }
            ))

    print(f"  [PDF] {filename}: {len(pages)} non-empty pages loaded")
    return pages


def load_corpus(input_path: str) -> List[Tuple[str, Dict]]:
    """
    Load all PDFs from a folder, or a single PDF file.
    Falls back to .txt files if no PDFs are found.
    """
    p = Path(input_path)
    all_pages = []

    if p.is_dir():
        pdf_files = list(p.glob("*.pdf"))
        txt_files = list(p.glob("*.txt"))

        if not pdf_files and not txt_files:
            print(f"  [Warning] No PDF or TXT files found in {input_path}")
            return []

        for pdf_path in pdf_files:
            all_pages.extend(load_pdf(str(pdf_path)))

        # TXT fallback: treat each file as a single-page document
        for txt_path in txt_files:
            text = txt_path.read_text(encoding="utf-8")
            all_pages.append((text, {"source": txt_path.name, "page": 1, "filepath": str(txt_path)}))

    elif p.suffix.lower() == ".pdf":
        all_pages = load_pdf(str(p))
    elif p.suffix.lower() == ".txt":
        text = p.read_text(encoding="utf-8")
        all_pages = [(text, {"source": p.name, "page": 1, "filepath": str(p)})]
    else:
        raise ValueError(f"Unsupported input: {input_path}. Provide a PDF, TXT, or folder.")

    print(f"\n[Corpus] Loaded {len(all_pages)} page(s) from {input_path}")
    return all_pages


# ── Chunker ───────────────────

def chunk_recursive(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """Recursive character chunking — same implementation as lab2_3_chunking.py."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps or len(text) <= size:
            return [text] if text.strip() else []
        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        results, current = [], ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if len(part.strip()) > size:
                    results.extend(_split(part.strip(), seps[1:]))
                    current = ""
                else:
                    current = part.strip()
        if current:
            results.append(current)
        return results

    raw_chunks = _split(text, separators)
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i-1][-overlap:]
        overlapped.append((prev_tail + " " + raw_chunks[i]).strip())
    return overlapped


# ── Embedders ────────────────────────────────────────────────────────────────

def get_embedder(embedder_name: str):
    """
    Return an embedding function matching the (texts: List[str]) -> List[List[float]] signature.
    Selecting at CLI time keeps the rest of the ingest loop model-agnostic.
    """
    if embedder_name == "ollama":
        import ollama
        OLLAMA_MODEL = "nomic-embed-text"

        def embed_fn(texts: List[str]) -> List[List[float]]:
            results = []
            for text in texts:
                resp = ollama.embed(model=OLLAMA_MODEL, input=text)
                results.append(resp["embeddings"][0])
            return results

        return embed_fn, 768

    elif embedder_name == "gemini":
        from google import genai as _genai
        from google.genai import types as _types

        _client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        GEMINI_EMBED_MODEL = "gemini-embedding-exp-03-07"

        def embed_fn(texts: List[str]) -> List[List[float]]:
            results = []
            for text in texts:
                resp = _client.models.embed_content(
                    model=GEMINI_EMBED_MODEL,
                    contents=text,
                    config={"task_type": "RETRIEVAL_DOCUMENT"},
                )
                results.append(resp.embeddings[0].values)
            return results

        return embed_fn, 768

    else:
        raise ValueError(f"Unknown embedder '{embedder_name}'. Choose: ollama | gemini")


# ── Main ingest pipeline ──────────────────────────────────────────────────────

def ingest(
    input_path: str,
    backend: str = "chroma",
    embedder: str = "ollama",
    chunk_size: int = 800,
    overlap: int = 120,
    collection: str = "capstone_corpus",
) -> None:
    """
    Full ingest pipeline:
    1. Load PDFs/TXT from input_path
    2. Chunk each page using recursive chunking
    3. Embed all chunks
    4. Upsert into the chosen VectorStore backend

    Progress is printed at each stage. Run this once per corpus; subsequent runs
    are idempotent (upsert semantics — existing IDs are updated, not duplicated).
    """
    print(f"\n{'='*60}")
    print(f" INGEST PIPELINE")
    print(f"  input    : {input_path}")
    print(f"  backend  : {backend}")
    print(f"  embedder : {embedder}")
    print(f"  chunk    : size={chunk_size}, overlap={overlap}")
    print(f"  collection: {collection}")
    print(f"{'='*60}")

    # Step 1 — Load
    t_load = time.perf_counter()
    pages = load_corpus(input_path)
    if not pages:
        print("[Ingest] No content loaded — aborting.")
        return
    print(f"[1/4] Loaded {len(pages)} page(s) in {time.perf_counter()-t_load:.2f}s")

    # Step 2 — Chunk
    t_chunk = time.perf_counter()
    all_chunks: List[str] = []
    all_meta:   List[Dict] = []

    for page_text, page_meta in pages:
        page_chunks = chunk_recursive(page_text, size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            all_meta.append({
                **page_meta,
                "chunk_idx":  chunk_idx,
                "chunk_size": chunk_size,
                "overlap":    overlap,
            })

    print(f"[2/4] Chunked → {len(all_chunks)} chunks in {time.perf_counter()-t_chunk:.2f}s")

    # Step 3 — Embed (in batches of 50 to respect free-tier rate limits)
    t_embed = time.perf_counter()
    embed_fn, dim = get_embedder(embedder)
    all_embeddings: List[List[float]] = []

    BATCH = 50
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        batch_embs = embed_fn(batch)
        all_embeddings.extend(batch_embs)
        print(f"  Embedded batch {i//BATCH + 1}/{(len(all_chunks)-1)//BATCH + 1} "
              f"({len(all_embeddings)}/{len(all_chunks)} chunks)")

    print(f"[3/4] Embedded {len(all_embeddings)} chunks (dim={dim}) in {time.perf_counter()-t_embed:.2f}s")

    # Step 4 — Upsert into vector store
    from WorkShop.Day2.D2P2 import get_vector_store
    t_store = time.perf_counter()

    store = get_vector_store(backend=backend, collection_name=collection, dim=dim)
    ids = [f"{collection}_{i}" for i in range(len(all_chunks))]

    store.add(ids=ids, documents=all_chunks, metadatas=all_meta, embeddings=all_embeddings)
    store.persist()

    print(f"[4/4] Stored {store.count()} documents → '{collection}' ({backend}) "
          f"in {time.perf_counter()-t_store:.2f}s")

    # Smoke query — verify end-to-end
    print("\n[Smoke Query] Testing retrieval...")
    from embedding_utils import embed_query_ollama, embed_ollama
    smoke_q = "What are the rules?"
    smoke_emb = embed_query_ollama(smoke_q)
    smoke_results = store.query(smoke_emb, k=2)
    for r in smoke_results:
        print(f"  score={r['score']:.3f} | {r['document'][:100]}...")

    print(f"\n[Done] Ingest complete. {len(all_chunks)} chunks in '{collection}' via {backend}.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into a vector store.")
    parser.add_argument("--input",      default="./sample_corpus",  help="PDF, TXT, or folder path")
    parser.add_argument("--backend",    default="chroma",           choices=["chroma", "faiss", "pinecone"])
    parser.add_argument("--embedder",   default="ollama",           choices=["ollama", "gemini"])
    parser.add_argument("--chunk-size", default=800,  type=int,     help="Max chars per chunk (default: 800)")
    parser.add_argument("--overlap",    default=120,  type=int,     help="Char overlap between chunks (default: 120)")
    parser.add_argument("--collection", default="capstone_corpus",  help="Collection / index name")

    args = parser.parse_args()

    ingest(
        input_path=args.input,
        backend=args.backend,
        embedder=args.embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        collection=args.collection,
    )

# Execution
# # Ingest with Ollama embeddings into Chroma (fastest, no API key needed)
# python P4.py --backend chroma --embedder ollama --input ./sample_corpus

# # Switch to FAISS backend — same corpus, different backend, same interface
# python P4.py --backend faiss --embedder ollama --input ./sample_corpus

# # Use Gemini embedder (requires GEMINI_API_KEY in .env)
# python P4.py --backend chroma --embedder gemini --input ./sample_corpus