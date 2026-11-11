import os
import uuid

import requests
import streamlit as st

# Where the API lives. Override for a remote host:
#   $env:AGENT_API="http://localhost:8000"  (PowerShell)
API = os.environ.get("AGENT_API", "http://localhost:8000")

st.set_page_config(page_title="Campus Agent", page_icon="🎓")
st.title("Campus Knowledge Agent")
st.caption(f"API: {API}")

# Per-tab identity: created once, then reused across every rerun.
if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []      # [{role, content}, ...] for redisplay

# Replay history on every Streamlit rerun (that is how st.chat_* works).
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask the handbook..."):
    # 1) Show + remember the user's turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2) Call the service (server-side request => no CORS involved here)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):       # the "thinking..." state
            try:
                resp = requests.post(
                    f"{API}/ask",
                    json={"question": prompt,
                          "session_id": st.session_state.session_id},
                    timeout=180,              # free-tier answers take 5-60s
                )
                resp.raise_for_status()
                data = resp.json()
                st.write(data["answer"])
                # Footer: the service's own observability fields, surfaced.
                cites = ", ".join(f"[{c}]" for c in data["citations"])
                st.caption(f"{cites or 'no citations'} · "
                           f"route={data['route']} · "
                           f"{data['latency_ms']} ms · {data['budget']}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"]})
            except requests.RequestException as e:
                # Service down / 503 quota / 429 rate limit — honest, not a crash.
                st.error(f"Service error: {e}")