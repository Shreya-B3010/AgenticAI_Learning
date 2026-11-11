from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END   # START/END are special node sentinels
from langgraph.graph.message import add_messages     # the reducer that merges lists
from langchain_core.messages import AIMessage


class FlowState(TypedDict):
    # Annotated[list, add_messages] = REDUCER: when a node returns {"messages": [x]},
    # LangGraph APPENDS x to the list instead of overwriting it. This one line is
    # why conversation history survives every node — the single most important
    # detail in all of LangGraph state design.
    messages: Annotated[list, add_messages]
    step: str                                       # plain field: last writer wins


def greet(state: FlowState) -> dict:
    """A node = a plain Python function. It returns a PARTIAL state update —
    only the keys it wants to change. LangGraph merges the rest."""
    return {"messages": [AIMessage(content="Hello from node greet.")], "step": "greeted"}


def follow_up(state: FlowState) -> dict:
    return {"messages": [AIMessage(content="Hello from node follow_up.")], "step": "followed"}


def main() -> None:
    # 1) Builder: blueprint. Nothing runs yet.
    builder = StateGraph(FlowState)
    builder.add_node("greet", greet)
    builder.add_node("follow_up", follow_up)

    # 2) Edges: fixed routing. START -> greet -> follow_up -> END.
    #    (Conditional edges — the "decide where to go next" kind — appear in
    #     P2.py; here the path is linear so the API stays tiny.)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", "follow_up")
    builder.add_edge("follow_up", END)

    # 3) compile(): blueprint -> runnable graph. No checkpointer yet (Session 2).
    graph = builder.compile()

    # --- invoke: run to completion, return the FINAL state ---
    final_state = graph.invoke({"messages": [("user", "hi")], "step": ""})
    # Tuples ("user", "text") are accepted sugar: add_messages converts them
    # into proper HumanMessage objects.
    print("[invoke] step =", final_state["step"])
    print("[invoke] messages =", [m.content for m in final_state["messages"]])

    # --- stream_mode="values": a FULL state snapshot after every step ---
    # Use when: debugging "what did the state look like at stage N?"
    print("\n[stream: values]")
    for snapshot in graph.stream({"messages": [("user", "hi")], "step": ""},
                                 stream_mode="values"):
        print(f"  step={snapshot['step']!r}  nmsgs={len(snapshot['messages'])}")

    # --- stream_mode="updates": only the CHANGED keys, keyed by node name ---
    # Use when: watching the loop run ("which node fired, what did it return?").
    # This is the mode the agent trace in Step 5 uses.
    print("\n[stream: updates]")
    for update in graph.stream({"messages": [("user", "hi")], "step": ""},stream_mode="updates"):
        node_name = list(update.keys())[0]           # {"greet": {...}} -> "greet"
        print(f"  node={node_name}  returned keys={list(update[node_name].keys())}")

    # --- stream_mode="messages": every MESSAGE as it is emitted, with metadata ---
    # Use when: token-by-token UX (metadata['langgraph_node'] says who produced it).
    # NOTE: you only see a message when its node FINISHES unless the node itself
    # streams from the model (llm.stream()) — nodes are ordinary functions.
    print("\n[stream: messages]")
    for message, metadata in graph.stream({"messages": [("user", "hi")], "step": ""},
                                          stream_mode="messages"):
        print(f"  [{metadata.get('langgraph_node')}] {message.content}")


if __name__ == "__main__":
    main()


# Execution: python P1.py