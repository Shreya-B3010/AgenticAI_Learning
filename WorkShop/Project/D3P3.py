# workshop\Day3\P3.py
import os
import re
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Generator

import ollama
from dotenv import load_dotenv
from google import genai

import sys
from pathlib import Path

# 1. Get the absolute path of the Day3 directory itself
day3_dir = Path(__file__).resolve().parent

# 2. INSERT Day3 at the front (index 0) so Python looks here first!
sys.path.insert(0, str(day3_dir))          # Fixes local imports (P1, P2)
sys.path.append(str(day3_dir.parent))     # Keeps cross-folder imports (Day2.P2)

# Now your original imports will work perfectly from anywhere:
from D2P2 import get_vector_store
from D3P1 import rewrite_query, mmr_rerank, gemini_text, embed_query_ollama, embed_ollama, \
    GEMINI_CHAT_MODEL, OLLAMA_CHAT_MODEL
from D3P2 import build_prompt

load_dotenv()

# One Gemini client for streaming calls (non-streaming goes through gemini_text()).
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEFAULT_COLLECTION = "capstone_corpus"
FETCH_MULTIPLIER = 3          # over-fetch factor when MMR is on (same as P1.py)

# Matches citation markers like [2] or [13] in generated text.
CITE_RE = re.compile(r"\[(\d+)\]")

# Substrings that signal the model obeyed the abstention rule (case-insensitive match).
ABSTAIN_MARKERS = ("not in the knowledge base", "i don't know", "i do not know")


# ── Pipeline preparer: rewrite → embed → retrieve(+MMR?) → prompt ────────────
def _prepare(
    question: str,
    backend: str,
    topk: int,
    mmr: bool,
    rewrite: bool,
    collection: str,
) -> Dict[str, Any]:
    """
    Steps 1–4 of this pipeline. Shared by answer() (non-streaming, used by the
    agent tool + RAGAS) and the CLI (streaming) so there is exactly ONE
    retrieval path in the codebase.

    Returns {prompt, results, query_used, latency{rewrite_s, embed_s, retrieve_s}}.
    """
    latency: Dict[str, float] = {}

    # Step 1 — optional query rewrite (0 or 1 Gemini call, backoff inside).
    t0 = time.perf_counter()
    search_q = rewrite_query(question) if rewrite else question
    latency["rewrite_s"] = round(time.perf_counter() - t0, 3) if rewrite else 0.0

    # Step 2 — embed the (possibly rewritten) query.
    # HARD RULE: same embedder as ingest. There is deliberately no --embedder
    # flag at query time — mixing embedders between ingest and query silently
    # destroys retrieval quality (Day 2 lesson, enforced here in code).
    t0 = time.perf_counter()
    q_emb = embed_query_ollama(search_q)
    latency["embed_s"] = round(time.perf_counter() - t0, 3)

    # Step 3 — retrieve through the Day 2 abstraction (backend switch lives here).
    t0 = time.perf_counter()
    store = get_vector_store(backend=backend, collection_name=collection, dim=len(q_emb))
    fetch_k = topk * FETCH_MULTIPLIER if mmr else topk
    results = store.query(q_emb, k=fetch_k)

    if mmr and len(results) > topk:
        # FAISS/Pinecone results don't carry their stored vectors, so we re-embed
        # the small candidate set (<= topk*3 texts) to get vectors for MMR.
        # At this scale that is one local embedding batch — negligible latency.
        # (A production VectorStore would optionally return embeddings instead.)
        cand_embs = embed_ollama([r["document"] for r in results])
        order = mmr_rerank(q_emb, cand_embs, topk)
        results = [results[i] for i in order]
    else:
        results = results[:topk]
    latency["retrieve_s"] = round(time.perf_counter() - t0, 3)

    # Step 4 — render the prompt contract.
    # NOTE: the prompt uses the ORIGINAL question (conversational nuance may
    # have been dropped by the rewrite); only retrieval used search_q.
    prompt = build_prompt(question, results)

    return {
        "prompt": prompt,
        "results": results,
        "query_used": search_q,
        "latency": latency,
    }

# *************************************************************************************
def _generate(prompt: str, llm: str) -> tuple:
    """
    Non-streaming generation. Returns (text, seconds).
    Gemini path uses gemini_text() — 429 backoff.
    """
    t0 = time.perf_counter()
    if llm == "gemini":
        text = gemini_text(prompt)
    elif llm == "ollama":
        resp = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp["message"].get("content") or "").strip()
    else:
        raise ValueError(f"Unknown llm '{llm}'. Choose: gemini | ollama")
    return text, time.perf_counter() - t0


def _generate_stream(prompt: str, llm: str, max_retries: int = 5) -> Generator[str, None, None]:
    """
    Yield answer text token-by-token (Gemini) or chunk-by-chunk (Ollama).

    Backoff subtlety: retries are ONLY attempted if the failure happens BEFORE
    the first token is yielded — retrying mid-stream would duplicate text the
    user already saw. After the first token, a failure raises to the caller.
    """
    if llm == "ollama":
        stream = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,                    # chunk-by-chunk generator
        )
        for chunk in stream:
            piece = chunk["message"].get("content") or ""
            if piece:
                yield piece
        return

    if llm != "gemini":
        raise ValueError(f"Unknown llm '{llm}'. Choose: gemini | ollama")

    # Gemini streaming path with pre-first-token backoff.
    for attempt in range(max_retries):
        yielded_any = False                 # tracks whether output already reached the user
        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
            )
            for chunk in stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
            return                          # stream completed normally
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper()
            # Give up if: not a rate-limit, retries exhausted, or output already shipped.
            if not is_rate_limit or attempt == max_retries - 1 or yielded_any:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"\n  [backoff] 429 on attempt {attempt+1}. Waiting {wait:.1f}s...")
            time.sleep(wait)

def _citations(text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map every [n] marker in the answer to the source/page/score of chunk n.
    Markers out of range (hallucinated numbers) are silently dropped — the
    citations list only ever contains real, traceable chunks.
    """
    markers = sorted({int(m) for m in CITE_RE.findall(text)})   # de-dup + sort
    citations = []
    for n in markers:
        if 1 <= n <= len(results):
            meta = results[n - 1].get("metadata", {})
            citations.append({
                "marker": n,
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", "?"),
                "score":  round(results[n - 1].get("score", 0.0), 3),
            })
    return citations


def _abstained(text: str) -> bool:
    """True if the answer contains the abstention phrase from the prompt contract."""
    lowered = text.lower()
    return any(marker in lowered for marker in ABSTAIN_MARKERS)


def answer(
    question: str,
    backend: str = "chroma",
    llm: str = "gemini",
    topk: int = 5,
    mmr: bool = False,
    rewrite: bool = False,
    collection: str = DEFAULT_COLLECTION,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    THE public entry point — a clean function with a JSON-shaped return.

    This exact signature becomes:
      - an agent TOOL in naive_agent.py          (Session 3, today)
      - a LangGraph node in agent_langgraph.py   (Day 4)

    Args:
        min_score: optional guardrail — if the BEST retrieval score is below
            this threshold, abstain WITHOUT calling the LLM (saves quota and
            prevents a confident answer built on irrelevant chunks).
            None (default) disables the pre-emptive check; the prompt-level
            abstention rule still applies either way.

    Returns:
        {
          "text": str,           # full answer (may be the abstention phrase)
          "citations": [...],    # [{marker, source, page, score}]
          "scores": [...],       # retrieval scores, in context-block order
          "abstained": bool,     # did the model (or guardrail) refuse?
          "latency": {...},      # rewrite_s, embed_s, retrieve_s, generate_s, total_s
          "retrieved": [...],    # raw result dicts (for debugging / RAGAS)
          "query_used": str,     # original or rewritten query
        }
    """
    prep = _prepare(question, backend, topk, mmr, rewrite, collection)
    results = prep["results"]

    # ── Optional pre-emptive abstention (cheap guardrail, LLM not called) ───
    if min_score is not None and (not results or results[0].get("score", 0.0) < min_score):
        latency = dict(prep["latency"])
        latency["generate_s"] = 0.0
        latency["total_s"] = round(sum(latency.values()), 3)
        return {
            "text": "I don't know — the answer is not in the knowledge base.",
            "citations": [],
            "scores": [round(r.get("score", 0.0), 3) for r in results],
            "abstained": True,
            "latency": latency,
            "retrieved": results,
            "query_used": prep["query_used"],
        }

    # ── Normal path: generate, then post-process ────────────────────────────
    text, gen_s = _generate(prep["prompt"], llm)

    latency = dict(prep["latency"])
    latency["generate_s"] = round(gen_s, 3)
    latency["total_s"] = round(sum(latency.values()), 3)   # all four stages

    return {
        "text": text,
        "citations": _citations(text, results),
        "scores": [round(r.get("score", 0.0), 3) for r in results],
        "abstained": _abstained(text),
        "latency": latency,
        "retrieved": results,
        "query_used": prep["query_used"],
    }

# Execution:
# python -c "from P3 import answer; print(answer('What are the library hours?')['text'][:200])"
# python -c "from P3 import answer; r=answer('How much is the fine for late book return?'); print(r['text']); print(r['citations']); print(r['latency'])"

# ***************************************************************************************************************
def _report(result_text: str, results: List[Dict], citations: List[Dict],
            latency: Dict[str, float], abstained: bool) -> None:
    """Print the post-answer report: citations, abstention flag, latency table."""
    if citations:
        print("[citations]")
        for c in citations:
            print(f"  [{c['marker']}] {c['source']} p.{c['page']} (score {c['score']})")
    else:
        print("[citations] (none — model cited nothing or abstained)")

    print(f"[abstained] {'yes' if abstained else 'no'}")

    # Latency breakdown — the measurement that drives every tuning decision.
    parts = " · ".join(f"{stage}={value:.3f}s" for stage, value in latency.items())
    print(f"[latency]   {parts}")


def _run_once(question: str, args: argparse.Namespace) -> None:
    """Prepare the pipeline once, then generate (streaming or not) + report."""
    prep = _prepare(
        question,
        backend=args.backend,
        topk=args.topk,
        mmr=(args.mmr == "on"),
        rewrite=(args.rewrite == "on"),
        collection=args.collection,
    )
    results = prep["results"]

    if args.stream == "on":
        print("[answer] ", end="", flush=True)
        parts: List[str] = []
        t0 = time.perf_counter()
        for token in _generate_stream(prep["prompt"], args.llm):
            print(token, end="", flush=True)   # token-by-token UX (Day 1 lesson)
            parts.append(token)
        gen_s = time.perf_counter() - t0
        text = "".join(parts)
        print()                                # newline after the streamed answer
    else:
        text, gen_s = _generate(prep["prompt"], args.llm)
        print(f"[answer] {text}")

    latency = dict(prep["latency"])
    latency["generate_s"] = round(gen_s, 3)
    latency["total_s"] = round(sum(latency.values()), 3)

    _report(text, results, _citations(text, results), latency, _abstained(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 3 RAG engine — citable, abstaining, backend-swappable."
    )
    parser.add_argument("--question",   default=None,
                        help="Question to answer (omit for interactive mode)")
    parser.add_argument("--backend",    default="chroma", choices=["chroma", "faiss", "pinecone"],
                        help="VectorStore backend from Day 2 (default: chroma)")
    parser.add_argument("--llm",        default="gemini", choices=["gemini", "ollama"],
                        help="Generation brain (default: gemini-3.6-flash)")
    parser.add_argument("--topk",       default=5, type=int,
                        help="Context blocks to retrieve (default: 5)")
    parser.add_argument("--mmr",        default="off", choices=["on", "off"],
                        help="MMR diversity re-rank (default: off)")
    parser.add_argument("--rewrite",    default="off", choices=["on", "off"],
                        help="Query rewrite before retrieval (default: off)")
    parser.add_argument("--stream",     default="on", choices=["on", "off"],
                        help="Stream tokens to console (default: on)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Collection / index name (default: capstone_corpus)")
    args = parser.parse_args()

    # Single-question mode.
    if args.question:
        _run_once(args.question, args)
        return

    # Interactive REPL — the class demo loop.
    print("=" * 66)
    print("  RAG ENGINE — interactive mode")
    print(f"  backend={args.backend} · llm={args.llm} · topk={args.topk} "
          f"· mmr={args.mmr} · rewrite={args.rewrite} · stream={args.stream}")
    print("  Type a question. Type 'quit' to exit.")
    print("=" * 66)
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not q:
            continue
        try:
            _run_once(q, args)
        except Exception as e:
            # Never crash the REPL — print the error and keep the session alive.
            print(f"[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


# Execution (run all four — this is the acceptance test):

#  # 1) Baseline: plain top-k, Gemini, streamed, cited
# python P3.py --question "What is the fine for late book return?"

# # 2) Both Session 1 levers ON — compare latency and citations against run 1
# python P3.py --question "What is the fine for late book return?" --mmr on --rewrite on

# # 3) Local brain — record quality/latency delta vs run 1
# python P3.py --question "When is the fee payment deadline for odd semesters?" --llm ollama

# # 4) Off-corpus question — the abstention guardrail must fire
# python P3.py --question "Who won the FIFA World Cup in 2022?" --llm ollama