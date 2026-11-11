# workshop\Day4\P2.py
import json
import argparse
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from dotenv import load_dotenv

# LangGraph core — Session 1 uses StateGraph/START/END/add_messages/ToolNode.
# MemorySaver + interrupt/Command are Session 2's; imported now so this header
# never needs editing between sessions.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver      
from langgraph.types import interrupt, Command           

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI   # cloud free-tier brain
from langchain_ollama import ChatOllama                     # local free-tier brain

import sys
from pathlib import Path

# 1. Get the workspace root and Day3 path explicitly
project_root = Path(__file__).resolve().parent.parent
day3_dir = project_root / "Day3"

# 2. Add Day3 first so Day3 scripts prioritize Day3 files, then add project root
sys.path.insert(0, str(day3_dir))
sys.path.append(str(project_root))

# 3. Day 3's engine — the search tool literally calls this function.
from D3P3 import answer as rag_answer

load_dotenv()


# ── Free-tier brain configuration (same models as Days 1-3) ──────────────────
GEMINI_MODEL = "gemini-3.6-flash"   # cloud, free. Out of daily quota? Try "gemini-3.8-flash" (also free, may 503).
OLLAMA_MODEL = "qwen2.5:3b"          # local, tool-calling capable (gemma3:4b has no tool support)

MAX_ITERATIONS = 5                  # kill-switch — Day 5 turns this into a full guardrail (budget + allow-list)

# Brain used by the search_knowledge_base tool's INNER RAG generation.
# build_agent_graph(brain=...) sets this so --brain ollama swaps the
# retrieval generator too (Day 5 brain plumbing).
SEARCH_LLM = "gemini"


def msg_text(message: Any) -> str:
    """
    Extract readable plain text from a LangChain message.

    Why this exists: Gemini responses sometimes return `content` as a LIST of
    typed blocks ( [{'type': 'text', 'text': '...'}] ) instead of a plain
    string — printing that dumps Python repr + signatures into the terminal.
    Handles str, list-of-blocks (dict or object form), and anything else.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # Only text blocks; skip thinking/signature-style blocks.
                if block.get("type") in (None, "text"):
                    parts.append(block.get("text", ""))
            else:  # object-style block with a .text attribute
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def make_llm(brain: str = "gemini"):
    """
    One factory for both free-tier brains — used by every LLM call today
    (agent node, orchestrator, synthesizer) so a brain swap is ONE argument.

    NOTE: deliberately NO temperature=. gemini-3.6-flash uses fixed sampling
    defaults; passing temperature only produces an SDK warning.
    """
    if brain == "gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    if brain == "ollama":
        return ChatOllama(model=OLLAMA_MODEL)
    raise ValueError(f"Unknown brain '{brain}'. Choose: gemini | ollama")

# Execution
# python -c "import P2; print('header OK')"

# ***************************************************************************************************************
@tool
def search_knowledge_base(question: str) -> Dict[str, Any]:
    """
    Search the campus handbook knowledge base and return an answer with citations.
    Use this for ANY fact about the college: library rules, fees, labs, attendance,
    scholarships, hostel, exams. The result contains 'abstained=true' when the
    handbook does not cover the topic — in that case DO NOT search again; tell
    the user the information is not in the knowledge base.
    """
    try:
        # Day 3 engine, unchanged. The INNER generator follows SEARCH_LLM
        # (default gemini) so Session 2's brain-swap can move it too.
        result = rag_answer(question, backend="chroma", llm=SEARCH_LLM, topk=5)
        return {
            "answer":    result["text"],
            "citations": result["citations"],
            "abstained": result["abstained"],
        }
    except Exception as e:
        # Errors become JSON OBSERVATIONS, not exceptions — the model reads
        # them and can self-correct. (Day 5 makes this a guardrail rule.)
        return {"error": f"search failed: {e}"}


@tool
def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression and return the numeric result. Use this
    whenever arithmetic is needed over numbers found in tool results (fees,
    fines, totals, percentages). Supports +, -, *, /, ** and parentheses.
    Numeric expression only — no units, no words. Example: '12 * 5'.
    """
    try:
        # Same safe-eval allowlist as Days 1/3: empty __builtins__ => the
        # string cannot import modules or touch attributes. Never eval
        # untrusted strings without this pattern.
        value = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return {"result": round(float(value), 6)}
    except Exception as e:
        return {"error": f"could not evaluate '{expression}': {e}"}


# The tool list the model sees (order = preference hint for small models).
AGENT_TOOLS = [search_knowledge_base, calculator]

# System instruction: role + citation habit + the abstain-not-guess rule.
# (Same contract as naive_agent.py — behavior must match.)
SYSTEM_PROMPT = """You are a campus knowledge agent.
- For any fact about the college, call search_knowledge_base.
- For arithmetic over retrieved numbers, call calculator.
- Call ONE tool at a time so results stay easy to follow.
- If a search returns abstained=true, do not retry the search — answer that the
  information is not in the knowledge base.
- Cite handbook facts using the citation markers returned by the search tool.
- Never invent handbook rules, fees, or dates."""

class AgentState(TypedDict):
    """
    The state that flows through the agent graph — the three fields the
    curriculum requires, each with a job:
      messages         full conversation history (reducer: append)
      tool_call_count  cost/observability — tool calls spent this run
      iterations       loop guard — agent-node executions (cap = MAX_ITERATIONS)
    """
    messages: Annotated[list, add_messages]
    tool_call_count: int
    iterations: int


def build_agent_graph(
    brain: str = "gemini",
    require_approval: bool = False,   # Step 3: pause before tool execution
    auto_approve: bool = False,       # Step 3: unattended runs (brain-swap)
    checkpointer=None,                # Session 2: pass MemorySaver()
):
    r"""
    (different brains) without editing code. Session 2 extends this signature
    with require_approval / auto_approve / checkpointer — the node logic below
    does not change.

    Graph shape (the naive_agent.py while-loop, drawn):

        START -> agent --(tool calls?)--> tools --> agent --(tool calls?)--> ...
                       \--(final text / cap reached)--> END
    """
    # Bind tools ONCE per compiled graph (binding is cheap, but doing it once
    # mirrors how production agents configure their model at startup).
    global SEARCH_LLM
    SEARCH_LLM = brain                    # search tool's inner generator follows --brain
    llm_with_tools = make_llm(brain).bind_tools(AGENT_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """One LLM call. Returns a partial update: the model's message plus
        the two counters. System message is prepended here (not stored in
        state) so history stays user/assistant/tool only."""
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            # Increment by the number of calls this response made (usually 0 or 1).
            "tool_call_count": state.get("tool_call_count", 0) + len(response.tool_calls),
            "iterations": state.get("iterations", 0) + 1,
        }

    def should_continue(state: AgentState) -> str:
        """
        THE conditional edge — 'did the model request a tool?'

        The prebuilt `tools_condition` (from langgraph.prebuilt) returns
        exactly "tools" or "__end__" for this same check. We write our own
        because we ALSO need the iteration kill-switch — the Day 5 guardrail
        preview: no runaway loops, ever, regardless of what the model wants.
        """
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return "__end__"                       # cap hit: stop, even mid-tooling
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):      # model asked for a tool
            return "tools"
        return "__end__"                           # final text: done

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)          # the LLM decision step
    # builder.add_node("tools", ToolNode(AGENT_TOOLS))  # executes tools, emits ToolMessages
    builder.add_node("tools", make_tools_node(require_approval, auto_approve))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        # path_map: what each return value means (name -> node, "__end__" -> END)
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "agent")             # THE LOOP: tools -> agent -> ...

    # Session 1: no checkpointer — every invoke starts from scratch (stateless,
    # exactly like naive_agent.py). Session 2 passes MemorySaver() here.
    return builder.compile(checkpointer=checkpointer) 

# Execution: python -c "import D4P2; g = D4P2.build_agent_graph(); print('graph built:', [n for n in g.get_graph().nodes])"
# python -c "import D4P2; print(D4P2.make_llm('ollama').invoke('Reply with: Ollama connection successful'))"
# python -c "import D4P2; print(D4P2.search_knowledge_base.invoke({'question':'What are the library timings?'}))"


# ***********************************************************************************************************************************
# def run_agent(question: str, brain: str = "gemini") -> str:
#     """
#     One question through the graph (Session 1 version — stateless).
#     Compiling per call is cheap; Session 2 replaces this with a version that
#     reuses ONE compiled graph so memory persists across calls.
#     """
#     graph = build_agent_graph(brain=brain)
#     state = graph.invoke({
#         "messages": [("user", question)],
#         "tool_call_count": 0,      # per-run counters: reset on every new question
#         "iterations": 0,
#     })
#     return msg_text(state["messages"][-1])

def run_agent(question: str, brain: str = "gemini",
              thread_id: str = "session-1", graph=None) -> str:
    """Session 2: same signature + thread_id. Pass `graph` to REUSE one
    compiled graph (memory persists); omit it and a fresh stateless graph
    is compiled (fine for one-shot demos — its memory dies with the call)."""
    g = graph or build_agent_graph(brain=brain, checkpointer=MemorySaver())
    return ask(g, question, make_config(thread_id))

def trace_agent(question: str, brain: str = "gemini") -> None:
    """
    Same graph, stream_mode="updates" — students SEE the loop:
      [agent] -> [tools] -> [agent] -> ... -> END
    This is the framework's gift: yesterday's while-loop was invisible;
    today every iteration is a printed line with counters attached.
    """
    graph = build_agent_graph(brain=brain)
    print(f"[trace] {question}")
    for update in graph.stream({"messages": [("user", question)],
                                "tool_call_count": 0, "iterations": 0},
                               stream_mode="updates"):
        node = list(update.keys())[0]
        payload = update[node]
        if node == "agent":
            last = payload["messages"][-1]
            if last.tool_calls:
                names = [tc["name"] for tc in last.tool_calls]
                detail = f"TOOL CALL -> {names}"
            else:
                detail = "FINAL: " + msg_text(last)[:110]
            print(f"  [agent] iter={payload.get('iterations')} tools_total="
                  f"{payload.get('tool_call_count')} {detail}")
        elif node == "tools":
            # ToolNode returns {"messages": [ToolMessage, ...]}
            print(f"  [tools] {msg_text(payload['messages'][-1])[:110]}")




# *********************************************************************************************
def make_config(thread_id: str) -> dict:
    """
    EVERY graph call on a checkpointer needs this config. thread_id is the
    memory key — like a session id in a chat app. Same thread = same memory;
    different thread = blank slate. (Without a checkpointer this config is
    simply ignored, so code can pass it unconditionally.)
    """
    return {"configurable": {"thread_id": thread_id}}


def ask(graph, question: str, cfg: dict) -> str:
    """
    Run ONE question against a compiled graph — and if the graph pauses for
    human approval (Session 2, Step 3), keep asking until it finishes.
    Shared by --question mode, the REPL, and the HITL demo: one invoke path
    in the whole codebase.
    """
    state = graph.invoke({
        "messages": [("user", question)],
        "tool_call_count": 0,     # per-run counters reset each question
        "iterations": 0,
    }, config=cfg)

    # Loop (not just an if): a run can interrupt MORE than once — e.g. the
    # model approves one calculator call, then asks for another on a later turn.
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value   # what interrupt() paused with
        decision = approval_cli(payload)            # defined in Step 3
        state = graph.invoke(Command(resume=decision), config=cfg)

    return msg_text(state["messages"][-1])

def demo_memory(brain: str = "gemini") -> None:
    """
    5-turn, two-thread memory test (the Session 2 deliverable check):
      thread 'arun' — 3 turns: name/roll given, then RECALLED (needs history)
      thread 'bipin'   — 2 turns: independent memory, must NOT know Arun
    Then state inspection: what does the checkpointer think is happening?
    """
    # ONE graph for both threads — threads are just different memory keys.
    graph = build_agent_graph(brain=brain, checkpointer=MemorySaver())

    alice_turns = [
        "Hello! My name is Arun and my roll number is 42. Remember these.",
        "What is my name?",          # cannot answer without turn-1 history
        "What is my roll number?",
    ]
    for i, q in enumerate(alice_turns, 1):
        answer = ask(graph, q, make_config("arun"))
        print(f"[arun #{i}] Q: {q}")
        print(f"[arun #{i}] A: {answer[:150]}")

    # State inspection — the two fields you will use all day:
    #   .next    which nodes would run next (() = finished/idle)
    #   .values  the full stored state
    snap = graph.get_state(make_config("arun"))
    print(f"\n[inspect] thread=arun  next={snap.next}  "
          f"stored_messages={len(snap.values['messages'])}")

    # Second thread — same graph, blank memory. Must NOT say "Arun".
    answer = ask(graph, "What is my name?", make_config("bipin"))
    print(f"[bipin] A: {answer[:150]}   <- must NOT reveal Arun")
    print("[memory] PASS if bipin's answer shows no knowledge of Arun/42.")


def demo_timetravel(brain: str = "gemini") -> None:
    """
    Rewind + re-decide:
      1) run a question that triggers a tool -> graph pauses at interrupt
      2) inspect get_state_history()  (returns NEWEST FIRST)
      3) FORK from the waiting-at-tools checkpoint:
           invoke(None, {thread_id, checkpoint_id})
         -> re-executes from that snapshot and PAUSES AGAIN
      4) decide differently this time: Command(resume="reject")

    CAUTION (verified while writing this lab): do NOT pass
    Command(resume=...) together with checkpoint_id in ONE invoke.
    LangGraph then replays the ORIGINAL branch's saved writes and your new
    decision is silently ignored — always fork first, resume second.
    """
    # Approval must be ON, or there is no interrupt point to rewind to.
    graph = build_agent_graph(brain=brain, require_approval=True,
                              checkpointer=MemorySaver())
    tid = "timetravel-demo"
    cfg = make_config(tid)

    state = graph.invoke({"messages": [("user", "What is 12 * 5?")],
                          "tool_call_count": 0, "iterations": 0}, config=cfg)
    print("[1] paused at:", graph.get_state(cfg).next)        # -> ('tools',)

    # History is NEWEST FIRST: [0] = latest snapshot.
    history = list(graph.get_state_history(cfg))
    for i, snap in enumerate(history):
        print(f"    [{i}] next={snap.next}")

    # Find the snapshot where the graph was WAITING to run tools...
    target = next(s for s in history if s.next == ("tools",))
    checkpoint_id = target.config["configurable"]["checkpoint_id"]

    # ...and fork from it. No resume yet: the fork re-runs and pauses again.
    fork_cfg = {"configurable": {"thread_id": tid, "checkpoint_id": checkpoint_id}}
    graph.invoke(None, config=fork_cfg)
    print("[2] forked; paused again at:", graph.get_state(fork_cfg).next)

    # THIS time the human decides differently -> a different branch of history.
    state = graph.invoke(Command(resume="reject"), config=cfg)
    print("[3] after reject:", msg_text(state["messages"][-1])[:140])

def make_tools_node(require_approval: bool, auto_approve: bool):
    """
    Build the graph's tools node with an optional human approval gate.

    WHY dynamic interrupt() instead of compile(interrupt_before=["tools"]):
    interrupt_before pauses the graph but carries NO payload into the node —
    there is nowhere for the approver's yes/no to land. interrupt() pauses
    AND returns the human's decision as the value the node continues with.
    Approvals need the payload; dynamic is the only correct choice.
    """
    raw_tools = ToolNode(AGENT_TOOLS)   # still does the real execution

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        pending = [{"name": tc["name"], "args": tc["args"]}
                   for tc in (getattr(last, "tool_calls", None) or [])]

        if require_approval and not auto_approve:
            # *** THE PAUSE *** — invoke() returns control to the caller with
            # state["__interrupt__"] set; resume with Command(resume=...).
            # The payload is what approval_cli() shows the human.
            decision = interrupt({"action": "approve_tool_calls",
                                  "tool_calls": pending})
        else:
            decision = "approve"   # unattended mode: brain-swap, demos, REPL

        if decision != "approve":
            # Rejection is a MESSAGE (role: user), not an exception — the model
            # reads it next turn and can change course. Exceptions kill runs;
            # observations enable self-correction (Day 5 guardrail rule).
            return {"messages": [HumanMessage(
                content="The human REJECTED this tool call. Do not call any tools "
                        "now; tell the user the action was not approved.")]}
        return raw_tools.invoke(state)   # approved -> ToolNode executes for real

    return tools_node


def approval_cli(payload: dict) -> str:
    """Tiny approval UX: show the pending tool calls, ask y/n (loop until valid)."""
    print("\n=== HUMAN APPROVAL REQUIRED ===")
    for tc in payload.get("tool_calls", []):
        # json.dumps for readable args (ASCII-safe output on Windows consoles)
        print(f"  {tc['name']}({json.dumps(tc['args'])})")
    while True:
        answer = input("Approve? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return "approve"
        if answer in ("n", "no"):
            return "reject"
        print("Type y or n.")


def demo_hitl(brain: str = "gemini") -> None:
    """Both branches of the gate, on separate threads (separate memories)."""
    graph = build_agent_graph(brain=brain, require_approval=True,
                              checkpointer=MemorySaver())

    print("--- case A: APPROVE the calculator ---")
    print("A ->", ask(graph, "What is 12 * 5?", make_config("hitl-approve")))

    print("\n--- case B: REJECT the calculator ---")
    print("B ->", ask(graph, "What is 12 * 5?", make_config("hitl-reject")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 4 — LangGraph agent"
    )

    parser.add_argument("--question", default=None)

    parser.add_argument(
        "--brain",
        default="gemini",
        choices=["gemini", "ollama"]
    )

    parser.add_argument(
        "--trace",
        action="store_true"
    )

    parser.add_argument(
        "--demo",
        choices=["memory", "timetravel", "hitl"],
        default=None
    )

    args = parser.parse_args()

    # Demo execution
    if args.demo == "memory":
        demo_memory(args.brain)
        return

    if args.demo == "timetravel":
        demo_timetravel(args.brain)
        return

    if args.demo == "hitl":
        demo_hitl(args.brain)
        return

    # Existing question execution
    if args.question:
        if args.trace:
            trace_agent(args.question, args.brain)
        else:
            print(run_agent(args.question, args.brain))
        return

    # Existing interactive REPL continues here

    # Interactive REPL — the class demo loop. Exceptions never kill the session.
    print("=" * 66)
    print(f"  LANGGRAPH AGENT — brain={args.brain} (stateless, Session 1)")
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
            print("Agent:", run_agent(q, args.brain))
        except Exception as e:
            # Free-tier reality: surface 429s with an actionable next step.
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("[quota] Gemini daily quota exhausted. Re-run with --brain ollama "
                      "or retry after the Pacific-time reset.")
            else:
                print(f"[error] {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()

# Execution:
# # 1) SINGLE-HOP — expect: one search, then final text (1-2 iterations)
# python D4P2.py --question "What are the library opening hours?" --trace

# # 2) MULTI-HOP — expect: search -> calculator -> cited final with the Rs 60 math
# python D4P2.py --question "A book is 12 days overdue. According to the handbook, what fine do I owe for the late return? Show the calculation and cite the regulation." --trace

# # 3) OFF-CORPUS — expect: <=1 search, honest refusal, NO invented citations
# python D4P2.py --question "Who won the FIFA World Cup in 2022?" --trace

# Execution:    python D4P2.py --demo memory
#               python D4P2.py --demo timetravel


# Execution:    python D4P2.py --demo hitl