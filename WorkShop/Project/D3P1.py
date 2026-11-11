import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np          # MMR matrix math
import ollama               # local embedding path
import chromadb             
from dotenv import load_dotenv
from google import genai    # Gemini client for query rewriting + generation

# Allow this module to be imported from any working directory by ensuring the
# local project folder and its sibling workshop folders are on sys.path.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root.parent))

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
# Free-tier chat model used for query rewriting and the one-shot RAG demo.
GEMINI_CHAT_MODEL = "gemini-3.6-flash"

# Local fallback brain for the quality/latency comparison (8 GB RAM friendly).
OLLAMA_CHAT_MODEL = "gemma2:2b"

# Embedding model 
# Mixing embedders between ingest and query silently destroys retrieval quality.
OLLAMA_EMBED_MODEL = "nomic-embed-text"     # 768-dim, local, offline

# Collection created by ingest.py on Day 2 — the capstone corpus.
COLLECTION_NAME = "capstone_corpus"

# When MMR is ON we fetch this many candidates per requested result so the
# re-ranker has a pool to trade relevance for diversity from.
FETCH_MULTIPLIER = 3

# One Gemini client for the whole file (connection reuse).
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Embedding helpers 
# module-level demo output. The one invariant that matters is the model name.

def embed_query_ollama(query: str) -> List[float]:
    """Embed one query string with nomic-embed-text → 768-dim float vector."""
    resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=query)
    return resp["embeddings"][0]          # one text in → one vector in a list


def embed_ollama(texts: List[str]) -> List[List[float]]:
    """Embed a small batch of texts (used later to re-embed MMR candidates)."""
    vectors = []
    for text in texts:
        resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)
        vectors.append(resp["embeddings"][0])
    return vectors


# ── Gemini call with exponential backoff ───────────

def gemini_text(prompt: str, model: str = GEMINI_CHAT_MODEL, max_retries: int = 5) -> str:
    """
    Single-turn Gemini text call with exponential backoff on 429 rate limits.
    Waits 2^attempt seconds + random jitter (jitter prevents the thundering-herd
    problem when many clients retry at the same instant).
    Non-rate-limit errors (bad key, bad request) raise immediately.
    """
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (resp.text or "").strip()
        except Exception as e:
            # google-genai surfaces 429 as "429" or "RESOURCE_EXHAUSTED" in the message.
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper()
            if not is_rate_limit or attempt == max_retries - 1:
                raise                      # non-retryable, or retries exhausted
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [backoff] 429 on attempt {attempt+1}. Waiting {wait:.1f}s...")
            time.sleep(wait)


print("[retriever] Ready — embedding model:", OLLAMA_EMBED_MODEL, "| chat model:", GEMINI_CHAT_MODEL)

# **************************************************************************************************************
# Why rewrite? Retrieval matches the query string as-is. Conversational or
# anaphoric questions ("what about its fine?") embed poorly because the vector
# store has no memory of what "its" refers to. ONE extra Gemini call before
# retrieval converts them into standalone search queries.

REWRITE_PROMPT = """You are a search-query rewriter for a document retrieval system.
Rewrite the user question as a single standalone search query that could be
matched against a campus handbook. Resolve pronouns and conversation references.
Do NOT answer the question. Do NOT add facts that are not implied by the question.
If the question is already standalone, return it unchanged.

User question: {question}

Standalone search query:"""


def rewrite_query(question: str, model: str = GEMINI_CHAT_MODEL) -> str:
    """
    Return a retrieval-friendly standalone version of the question.
    Cost: exactly 1 Gemini call (with backoff). Falls back to the original
    question if the model returns an empty string — an empty rewrite would
    silently destroy retrieval quality.
    """
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = gemini_text(prompt, model=model)
    return rewritten if rewritten else question

# print(rewrite_query("what about its fine?"))                 # → standalone version
# print(rewrite_query("How many books can a UG student borrow?"))  # → returned unchanged
# print(rewrite_query("and the lab rules too?"))               # → resolved into a full query

# *************************************************************************************************
def mmr_rerank(
    query_emb: List[float],
    candidate_embeddings: List[List[float]],
    k: int,
    lam: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance — returns the INDICES of the k selected candidates.

        score(d) = lam * sim(query, d) - (1 - lam) * max_{s already selected} sim(d, s)

    lam = 1.0 → pure relevance (identical to plain top-k)
    lam = 0.0 → pure diversity (spreads selections across the candidate pool)
    Default 0.7 → keep relevance dominant, suppress near-duplicates.

    Input candidates are expected roughly relevance-sorted (Chroma returns them
    that way), but the algorithm does not depend on the order.
    """
    # Degenerate cases: nothing to choose from, or fewer candidates than k.
    if len(candidate_embeddings) <= k:
        return list(range(len(candidate_embeddings)))

    # (N, D) matrix of L2-normalised candidate vectors.
    # After normalisation, a row-dot-row product IS cosine similarity.
    mat = np.asarray(candidate_embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)      # guard against zero vectors

    # Normalise the query the same way so mat @ q = cosine sims to the query.
    q = np.asarray(query_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)

    sim_q = mat @ q                 # shape (N,) — similarity of every candidate to the query

    selected: List[int] = []        # indices chosen so far
    remaining = list(range(len(mat)))  # indices still in the pool

    while remaining and len(selected) < k:
        if not selected:
            # First pick: nothing to be redundant with → plain best match.
            best = remaining[int(np.argmax(sim_q[remaining]))]
        else:
            # Similarity of each remaining candidate to each already-selected one.
            redundancy = mat[remaining] @ mat[selected].T   # shape (len(remaining), len(selected))
            max_red = redundancy.max(axis=1)                # worst-case duplication per candidate
            mmr_scores = lam * sim_q[remaining] - (1.0 - lam) * max_red
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(best)
        remaining.remove(best)      # list.remove is O(N) — fine at k*3 ≤ ~30 candidates

    return selected


# **Execution steps:** *****IMPORTANT****
# Run these three commands separately to demonstrate the effect of lambda:

# Experiment 1 – Pure relevance
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=1.0))"

# Expected: [0, 1]

# Experiment 2 – Balanced relevance and diversity
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=0.7))"

# Expected: [0, 1]

# Experiment 3 – More diversity
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=0.3))"

# Expected: [0, 2]

# ***************************************************************************************************************************
_collection = None


def _get_collection():
    """
    Return the capstone_corpus Chroma collection, failing fast with an
    actionable message if Day 2's ingest was never run.
    """
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        try:
            _collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:
            raise SystemExit(
                f"[Error] Collection '{COLLECTION_NAME}' not found in ./chroma_db.\n"
                f"        Run Day 2's P4.py first:\n"
                f"        python P4.py --backend chroma --embedder ollama --input ./sample_corpus"
            )
    return _collection


def retrieve(
    question: str,
    k: int = 5,
    mmr: bool = False,
    rewrite: bool = False,
    lam: float = 0.7,
) -> Dict[str, Any]:
    """
    One retrieval call with three switches. This is Session 1's Chroma-direct
    lab version (raw vectors needed for MMR); Session 2's rag_engine.py routes
    through the Day 2 VectorStore abstraction instead so --backend works.

    Returns:
        {
          "query_used": str,            # original or rewritten question
          "results": [ {id, document, metadata, score, embedding}, ... ],
          "timings": {rewrite_s, embed_s, retrieve_s}
        }
    score = cosine similarity (1 - cosine distance); higher = more relevant.
    """
    col = _get_collection()
    timings: Dict[str, float] = {}

    # ── Switch: rewrite (0 or 1 Gemini call) ────────────────────────────────
    t0 = time.perf_counter()
    search_q = rewrite_query(question) if rewrite else question
    timings["rewrite_s"] = round(time.perf_counter() - t0, 3) if rewrite else 0.0

    # ── Always: embed the (possibly rewritten) query ────────────────────────
    # Same model as ingest — nomic-embed-text — no exceptions.
    t0 = time.perf_counter()
    q_emb = embed_query_ollama(search_q)
    timings["embed_s"] = round(time.perf_counter() - t0, 3)

    # ── Fetch: over-fetch when MMR needs a pool to choose from ──────────────
    t0 = time.perf_counter()
    fetch_k = k * FETCH_MULTIPLIER if mmr else k
    resp = col.query(
        query_embeddings=[q_emb],
        n_results=fetch_k,
        # ids come back automatically; we also want text, metadata, distance AND
        # the stored vectors (MMR needs the candidate embeddings).
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    # Chroma returns nested lists (one list per query vector) — we sent one query.
    raw: List[Dict[str, Any]] = []
    for id_, doc, meta, dist, emb in zip(
        resp["ids"][0],
        resp["documents"][0],
        resp["metadatas"][0],
        resp["distances"][0],
        resp["embeddings"][0],
    ):
        raw.append({
            "id":        id_,
            "document":  doc,
            "metadata":  meta,
            "score":     1.0 - dist,      # cosine distance → cosine similarity
            "embedding": list(emb),       # kept only so MMR can operate on it
        })

    # ── Switch: mmr (re-rank the pool, keep top k) ──────────────────────────
    if mmr and len(raw) > k:
        order = mmr_rerank(q_emb, [r["embedding"] for r in raw], k, lam)
        results = [raw[i] for i in order]
    else:
        results = raw[:k]
    timings["retrieve_s"] = round(time.perf_counter() - t0, 3)

    return {"query_used": search_q, "results": results, "timings": timings}

# Execution of the code: )
# python -c "from P1 import retrieve; out = retrieve('What is the fine for late book return?', k=3, mmr=True); print('Query used:', out['query_used']); print('Results:'); [print('Score:', round(r['score'], 3), '|', r['document'][:90]) for r in out['results']]"

# **************************************************************************************************************************************************************************************
AB_QUESTIONS = [
    "What are the library opening hours?",
    "How much is the fine for returning a book late?",
    "Can I renew a borrowed book?",
    # "What happens if I lose a library book?",
    # "How many books can a UG student borrow?",
    # "How do I book the AI lab?",
    # "When is the fee payment deadline for odd semesters?",
    # "What is the late fee penalty for fee payment?",
    # "How are government scholarships disbursed?",
]


def run_ab_experiment() -> None:
    """
    Prints the top-1 hit per variant per question so the class can vote.
    Two extra deep-dives (full top-5 for plain vs mmr) make the diversity
    effect visible — not just asserted. 
    """
    variants = [
        ("plain",   dict(mmr=False, rewrite=False)),
        ("mmr",     dict(mmr=True,  rewrite=False)),
        ("rewrite", dict(mmr=False, rewrite=True)),
        ("both",    dict(mmr=True,  rewrite=True)),
    ]

    print("\n" + "=" * 100)
    print("  A/B/C/D — top-1 result per variant (vote the best answer for each Q)")
    print("=" * 100)

    for qi, q in enumerate(AB_QUESTIONS, 1):
        print(f"\nQ{qi}: {q}")
        for name, kwargs in variants:
            out = retrieve(q, k=5, **kwargs)
            top = out["results"][0] if out["results"] else None
            if top is None:
                print(f"  [{name:<8}] (no results)")
                continue
            # Collapse newlines so the terminal table stays readable.
            snippet = top["document"][:95].replace("\n", " ")
            print(f"  [{name:<8}] score={top['score']:.3f} | {snippet}")

    # full top-5 for plain vs mmr on the first two questions ───
    # Hrere you can see MMR working: plain returns near-duplicates,
    # mmr spreads across different sections of the handbook.
    for q in AB_QUESTIONS[:2]:
        print(f"\n── Top-5 deep dive: '{q}' ──")
        for name, use_mmr in (("plain", False), ("mmr", True)):
            out = retrieve(q, k=5, mmr=use_mmr)
            print(f"  {name}:")
            for i, r in enumerate(out["results"], 1):
                print(f"    [{i}] score={r['score']:.3f} | {r['document'][:80].replace(chr(10), ' ')}")

# **********************************************************************************************************
from D3P2 import build_prompt


def answer_once(
    question: str,
    llm: str = "gemini",
    k: int = 5,
    mmr: bool = False,
    rewrite: bool = False,
) -> Dict[str, Any]:
    """
    Minimal end-to-end RAG: retrieve → build numbered-context prompt → generate.
    Session 1 uses it ONLY to verify the template and record the backend delta;
    Session 2's rag_engine.py replaces it with the full production engine.

    Returns {"text", "results", "latency_s"}.
    """
    t_all = time.perf_counter()

    # Steps 1–3 of the pipeline (rewrite / embed / retrieve) live in retrieve().
    retrieved = retrieve(question, k=k, mmr=mmr, rewrite=rewrite)
    results = retrieved["results"]

    # Step 4: render the prompt contract around the retrieved blocks.
    prompt = build_prompt(question, results)

    # Step 5: generate with whichever free-tier LLM was requested.
    t_gen = time.perf_counter()
    if llm == "gemini":
        text = gemini_text(prompt)                       # backoff included
    elif llm == "ollama":
        resp = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp["message"].get("content") or "").strip()
    else:
        raise ValueError("llm must be 'gemini' or 'ollama'")
    gen_s = time.perf_counter() - t_gen

    return {
        "text": text,
        "results": results,
        # Per-stage latency so the delta discussion is quantitative, not vibes.
        "latency_s": {
            **retrieved["timings"],
            "generate": round(gen_s, 3),
            "total": round(time.perf_counter() - t_all, 3),
        },
    }


# ── Main entry point: A/B experiment + backend delta demo ───────────────────

if __name__ == "__main__":
    print(" Advanced retrieval A/B (plain | mmr | rewrite | both)")
    
    # Part 1: the retrieval vote table (Gemini called only for rewrite variants).
    run_ab_experiment()

    # Part 2: prompt-contract wiring + Gemini vs Ollama delta.
    # Three questions is enough to compare quality and latency honestly.
    DEMO_QUESTIONS = [
        "What is the fine for returning a book late?",
        "How do I book the AI lab?",
        "Can a UG student borrow 5 books?",
    ]

    print("  ONE-SHOT RAG — gemini-3.6-flash vs gemma2:2b (same prompt, same k)")
    
    for llm_name in ("gemini", "ollama"):
        print(f"\n── backend: {llm_name} ──")
        for q in DEMO_QUESTIONS:
            out = answer_once(q, llm=llm_name, k=5)
            lat = out["latency_s"]
            print(f"Q: {q}")
            print(f"  latency: total={lat['total']:.2f}s (retrieve={lat['retrieve_s']:.3f}s, "
                  f"generate={lat['generate']:.2f}s)")
            # First 300 chars keeps the terminal readable; full text is in the lab notes.
            print(f"  answer : {out['text'][:300]}")

    print("\n[Lab] Record quality + latency observations")
    print("      → section 'Gemini vs Ollama one-shot delta'.")