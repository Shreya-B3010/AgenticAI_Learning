import os
import re
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from contextlib import asynccontextmanager
from uuid import uuid4

# ── Path bootstrap: ONE line that makes this file work in TWO places ────────
# Locally (workshop\Day6): harmless — D5P1's own path logic already reaches
#   workshop\Day2, Day3, Day4.
# On Vercel (project root == Day6): ESSENTIAL — the deployed bundle contains
#   ONLY this folder, so the Day2/Day3/Day4 copies inside it must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing D5P1 also prints the known "[retriever] Ready ..." side-effect line.
# D5P1 brings: build_multi (the graph), BUDGET (cost ledger), msg_text (extractor).
from D5P1 import build_multi, BUDGET, msg_text
# GEMINI_MODEL / OLLAMA_MODEL are only for reporting which brain+model is
# configured (/health + the graph_built/startup log lines).
# Safe now: importing D5P1 above already set up sys.path for the Day4 package.
from D4P2 import GEMINI_MODEL, OLLAMA_MODEL

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver   # in-process chat memory


# ── Configuration (env vars, never hardcoded — Day 6 secrets hygiene) ───────
# AGENT_BRAIN: "gemini" (default, free cloud) | "ollama" (local fallback when
# the Gemini daily quota dies — same dual-path design as Days 1-5).
BRAIN = os.environ.get("AGENT_BRAIN", "gemini")

# The model that actually answers for this brain — /health and the
# graph_built/startup log lines report THIS, so an ollama demo never lies
# about being on gemini (and vice versa).
MODEL = OLLAMA_MODEL if BRAIN == "ollama" else GEMINI_MODEL

# Per-IP rate limit: free-tier RPM is small, so the SERVICE should say no
# before Google does. Sliding window = last 60s, max 10 asks per client IP.
RATE_LIMIT, RATE_WINDOW_S = 10, 60.0

# Structured log file, written NEXT TO this file (agent.log — Session 2 reads it).
LOG_PATH = Path(__file__).with_name("agent.log")

# One lock around every agent run: the Day 5 BUDGET ledger and MemorySaver are
# process-global, so runs are SERIALIZED (one question at a time). Honest
# trade: throughput = 1 concurrent run — which is all the free tier allows anyway.
AGENT_LOCK = threading.Lock()


def log_event(**fields) -> None:
    """One JSON line per event — the Session 1 seed of observability.

    JSON-lines is the format every log tool understands (grep, jq, the
    trace_viewer.py built in Session 2). Prints to stdout (uvicorn/Vercel
    consoles) AND appends to agent.log (offline viewing). Logging must NEVER
    break a request: file errors are swallowed on purpose.
    """
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **fields}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass                       # read-only filesystem (some hosts): skip


class SlidingWindowLimiter:
    """Tiny in-memory per-key rate limiter (the 'sketch' from the curriculum).

    Keeps the timestamps of recent calls per client IP; anything older than
    the window is dropped. NOT a wall — a courtesy: it protects YOUR free-tier
    quota from a runaway UI refresh loop. Multi-instance deployments need a
    shared store (Redis) — out of scope, noted in the briefing.
    """

    def __init__(self, max_calls: int, window_s: float):
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_s:
                hits.popleft()      # expire entries outside the window
            if len(hits) >= self.max_calls:
                return False        # over the limit -> caller returns HTTP 429
            hits.append(now)
            return True


LIMITER = SlidingWindowLimiter(RATE_LIMIT, RATE_WINDOW_S)

# ── API contract (Pydantic = validation + the auto-generated /docs schema) ──
class AskRequest(BaseModel):
    """POST /ask body. session_id IS the LangGraph thread_id: same id = same
    memory, different id = blank slate (proven in Step 4's isolation test)."""
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The user's question")
    session_id: str = Field(default_factory=lambda: f"ui-{uuid4().hex[:8]}",
                            min_length=1, max_length=64,
                            description="Chat session key -> thread_id")


class AskResponse(BaseModel):
    """POST /ask answer. citations are the [n] markers found in the answer
    text (a capstone with structured citations would return them directly)."""
    answer: str
    session_id: str
    route: str          # retrieval | general | clarify  (the orchestrator's pick)
    status: str         # ok | <node>:failed | <node>:ok-after-retry
    citations: list[int]
    latency_ms: int
    budget: str         # Day 5's cost line: llm_calls=... tokens~... tools=[...]


CITE_RE = re.compile(r"\[(\d{1,2})\]")   # handbook citation markers like [3]


# ── Lifespan: cold-start engineering (compile the graph ONCE, not per request) ─
# Compiling builds the whole orchestrator-worker + the nested retrieval
# sub-graph — cheap, but doing it per request would be wasteful. This runs at
# server start (uvicorn) so the FIRST user request is fast.
# On Vercel the runtime may skip lifespan events, so get_graph() below also
# lazy-builds on first use — same code path, both hosts covered.
def get_graph():
    """Return the compiled agent graph, building it on first call."""
    if getattr(app.state, "graph", None) is None:
        # MemorySaver = per-process chat memory (fine locally; Vercel caveat
        # in Block D: memory dies with the instance).
        app.state.graph = build_multi(brain=BRAIN, checkpointer=MemorySaver())
        log_event(event="graph_built", brain=BRAIN, model=MODEL)
    return app.state.graph



@asynccontextmanager
async def lifespan(_: FastAPI):
    get_graph()                      # warm the graph at startup (0 LLM calls)
    log_event(event="startup", brain=BRAIN, model=MODEL)
    yield
    # Nothing to close: MemorySaver is in-process; uvicorn owns the socket.


app = FastAPI(title="Campus Agent API",
              description="Day 5's hardened multi-agent behind POST /ask",
              version="1.0", lifespan=lifespan)

# CORS: browsers enforce it on JS calls. The Streamlit client posts server-side
# (no CORS involved), but allow its origin anyway so browser-based tools and
# the /docs page on other origins can reach the API too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Zero-quota liveness probe: use it to verify the service before spending
    a single Gemini call (docker/vercel deploys check THIS first)."""
    return {"status": "ok", "brain": BRAIN, "model": MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    """One question through the multi-agent graph.

    Sync `def` on purpose (not async): LangGraph/LangChain calls are blocking,
    and FastAPI runs sync endpoints on a worker threadpool — the event loop
    stays free to answer /health while a question is in flight.
    """
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"

    # 1) Rate limit — cheap, quota-aware refusal BEFORE any LLM work.
    if not LIMITER.allow(client):
        log_event(event="ask", session_id=req.session_id, http_status=429,
                  error="rate_limited", question=req.question[:80])
        raise HTTPException(status_code=429,
                            detail=f"Slow down: max {RATE_LIMIT} asks per "
                                   f"{int(RATE_WINDOW_S)}s from one client.",
                            headers={"Retry-After": str(int(RATE_WINDOW_S))})

    # 2) Run the agent. One lock = serialized runs (BUDGET + memory safety).
    try:
        with AGENT_LOCK:
            BUDGET.reset()           # every request starts a clean Day 5 ledger
            state = get_graph().invoke(
                {
                    # With a checkpointer these messages APPEND to the thread's
                    # history (add_messages reducer) — that is the chat memory.
                    "messages": [("user", req.question)],
                    "question": req.question,   # orchestrator reads THIS field
                    "route": "",                # filled by the orchestrator node
                    "route_reason": "",
                    "specialist_output": "",
                    "specialist_status": "",
                },
                config={"configurable": {"thread_id": req.session_id}},
            )
            budget_line = BUDGET.summary()      # llm_calls=3/14 tokens~... tools=[...]
    except Exception as e:
        # 3) Free-tier reality: quota death is a 503 with a plan, not a traceback.
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            log_event(event="ask", session_id=req.session_id, http_status=503,
                      error="gemini_quota", question=req.question[:80])
            raise HTTPException(
                status_code=503,
                detail="Gemini free-tier quota exhausted — retry after the "
                       "Pacific-midnight reset"
                       + (" or restart locally with AGENT_BRAIN=ollama."
                          if BRAIN == "gemini" else "."))
        raise                        # anything else: still visible (never hidden)

    # 4) Shape the response (citations are the [n] markers in the answer).
    latency_ms = int((time.perf_counter() - t0) * 1000)
    answer = msg_text(state["messages"][-1])
    citations = sorted({int(n) for n in CITE_RE.findall(answer)})
    route = state.get("route", "")
    status = state.get("specialist_status", "ok")

    log_event(event="ask", session_id=req.session_id, route=route,
              status=status, latency_ms=latency_ms, budget=budget_line,
              http_status=200, question=req.question[:80])
    return AskResponse(answer=answer, session_id=req.session_id, route=route,
                       status=status, citations=citations,
                       latency_ms=latency_ms, budget=budget_line)