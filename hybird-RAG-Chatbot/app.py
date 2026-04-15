from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.hybrid_rag.graph import HybridRAGAgent
from src.hybrid_rag.ingestion import KnowledgeBase
from src.hybrid_rag.memory import SessionMemoryManager
from src.hybrid_rag.runtime import build_runtime


st.set_page_config(
    page_title="Hybrid Search RAG Chatbot",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def bootstrap() -> tuple[KnowledgeBase, SessionMemoryManager, HybridRAGAgent]:
    if "runtime" not in st.session_state:
        st.session_state.runtime = build_runtime()
    runtime = st.session_state.runtime
    return runtime["kb"], runtime["memory"], runtime["agent"]


def ensure_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload your sources, then ask grounded questions over your knowledge base.",
            }
        ]
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "In-Process"
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://127.0.0.1:8000"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f7fb;
            --surface: rgba(255,255,255,0.70);
            --surface-strong: rgba(255,255,255,0.88);
            --text: #0e2239;
            --muted: #6d8197;
            --line: rgba(14,34,57,0.08);
            --blue: #0f6fff;
            --blue-soft: rgba(15,111,255,0.10);
            --red: #ea4e5a;
            --red-soft: rgba(234,78,90,0.10);
            --glow-blue: rgba(15,111,255,0.14);
            --glow-red: rgba(234,78,90,0.10);
            --shadow: 0 18px 48px rgba(20, 45, 76, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, var(--glow-blue), transparent 26%),
                radial-gradient(circle at 92% 4%, var(--glow-red), transparent 20%),
                linear-gradient(180deg, #fbfdff 0%, #f2f6fb 52%, #eef4fa 100%);
        }
        html, body, [data-testid="stAppViewContainer"], .main {
            overflow-y: auto !important;
            height: auto !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0 !important;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 0.35rem;
            padding-bottom: 6rem;
        }
        [data-testid="stAppHeader"],
        header[data-testid="stHeader"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            height: 0 !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            left: -9999px !important;
            top: -9999px !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stSidebar"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            width: min(352px, 86vw) !important;
            transform: translateX(calc(-100% - 24px));
            transition: transform 220ms ease, box-shadow 220ms ease !important;
            z-index: 9998 !important;
            background: rgba(255,255,255,0.58);
            backdrop-filter: blur(18px);
            border-right: 1px solid rgba(14,34,57,0.06);
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 100% !important;
        }
        [data-testid="stSidebar"][data-codex-open="true"] {
            transform: translateX(0) !important;
            box-shadow: 0 22px 44px rgba(20,45,76,0.18) !important;
        }
        #codex-sidebar-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(10, 18, 32, 0.12);
            backdrop-filter: blur(2px);
            opacity: 0;
            pointer-events: none;
            transition: opacity 200ms ease;
            z-index: 9997;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: 0.18rem;
            padding: 0.18rem 0.46rem;
            border-radius: 999px;
            font-size: 0.60rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--blue);
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15,111,255,0.08);
            backdrop-filter: blur(12px);
        }
        .hero-title {
            margin: 0;
            color: var(--text);
            font-size: 1.86rem;
            line-height: 1.02;
            font-weight: 760;
            letter-spacing: -0.05em;
        }
        .hero-subline {
            margin: 0.18rem 0 0.75rem 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.42;
            max-width: 620px;
        }
        .mode-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.45rem;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(14,34,57,0.08);
            background: rgba(255,255,255,0.86);
            box-shadow: 0 12px 26px rgba(20, 45, 76, 0.06);
            backdrop-filter: blur(12px);
            color: var(--text);
            font-size: 0.84rem;
            font-weight: 680;
        }
        .soft-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 0.72rem 0.82rem;
            backdrop-filter: blur(18px);
        }
        .section-note {
            color: var(--muted);
            margin-top: -0.12rem;
            margin-bottom: 0.55rem;
            line-height: 1.45;
            font-size: 0.88rem;
        }
        .mini-note {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.5;
        }
        .source-shell {
            background: rgba(255,255,255,0.70);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.65rem 0.78rem;
            margin-bottom: 0.52rem;
            backdrop-filter: blur(16px);
        }
        .source-title {
            color: var(--text);
            font-weight: 640;
            margin-bottom: 0.16rem;
            word-break: break-word;
            letter-spacing: -0.02em;
            font-size: 0.9rem;
        }
        .source-meta {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.38;
        }
        .badge {
            display: inline-block;
            padding: 0.14rem 0.42rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            color: var(--blue);
            border: 1px solid rgba(28,100,242,0.08);
            font-size: 0.6rem;
            margin-bottom: 0.24rem;
            backdrop-filter: blur(12px);
        }
        .warning-badge {
            display: inline-block;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            color: var(--red);
            border: 1px solid rgba(223,76,88,0.08);
            font-size: 0.62rem;
            margin-bottom: 0.32rem;
            backdrop-filter: blur(12px);
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(16,36,61,0.06);
            backdrop-filter: blur(12px);
            font-size: 0.82rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: white;
            border-color: transparent;
            background: linear-gradient(135deg, #0f6fff, #3d8bff);
            box-shadow: 0 8px 20px rgba(15,111,255,0.18);
        }
        div[data-testid="stTabs"] {
            margin-bottom: 0.2rem;
        }
        .stChatMessage {
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(250,252,255,0.76));
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(28, 55, 90, 0.07);
            backdrop-filter: blur(14px);
        }
        h2, h3 {
            letter-spacing: -0.03em;
        }
        h2 a, h3 a {
            display: none !important;
        }
        .compact-heading {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.25rem;
        }
        .compact-heading h3 {
            margin: 0;
            color: var(--text);
            font-size: 1rem;
        }
        .compact-label {
            color: var(--muted);
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
        }
        .empty-card {
            background: rgba(255,255,255,0.62);
            border: 1px dashed rgba(16,36,61,0.10);
            border-radius: 14px;
            padding: 0.65rem 0.75rem;
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.35;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(16,36,61,0.08);
            background: rgba(255,255,255,0.76);
            color: var(--text);
            box-shadow: 0 8px 18px rgba(28,55,90,0.05);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f6fff, #2c84ff);
            color: white;
            border-color: transparent;
            box-shadow: 0 12px 24px rgba(15,111,255,0.20);
        }
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(14,34,57,0.08);
            border-radius: 18px;
            padding: 0.58rem 0.72rem;
            box-shadow: 0 12px 26px rgba(20,45,76,0.06);
        }
        div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.24rem;
            line-height: 1.1;
            letter-spacing: -0.04em;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
            border-radius: 14px !important;
            background: rgba(255,255,255,0.72) !important;
            border-color: rgba(16,36,61,0.08) !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,0.62);
            overflow: hidden;
        }
        [data-testid="stCodeBlock"] {
            border-radius: 16px;
            overflow: hidden;
        }
        div[data-testid="stChatInput"] {
            position: sticky;
            bottom: 0.75rem;
            background:
                radial-gradient(circle at top right, rgba(15,111,255,0.08), transparent 20%),
                linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.94));
            backdrop-filter: blur(18px);
            border: 1px solid rgba(14,34,57,0.10);
            border-radius: 22px;
            padding: 0.28rem 0.42rem 0.18rem;
            box-shadow: 0 22px 50px rgba(20, 45, 76, 0.12);
            z-index: 20;
        }
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {
            font-size: 1rem !important;
            font-weight: 500;
        }
        .trace-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            margin-bottom: 0.65rem;
        }
        .trace-pill {
            background: rgba(255,255,255,0.80);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.34rem 0.56rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }
        .trace-pill-label {
            font-size: 0.58rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .trace-pill-value {
            margin-top: 0.02rem;
            font-size: 0.82rem;
            color: var(--text);
            font-weight: 680;
        }
        .library-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.42rem;
            }
            .library-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_sidebar_toggle() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const buttonId = "codex-sidebar-toggle";
        const backdropId = "codex-sidebar-backdrop";

        function getSidebar() {
          return doc.querySelector('[data-testid="stSidebar"]');
        }

        function ensureBackdrop() {
          let backdrop = doc.getElementById(backdropId);
          if (!backdrop) {
            backdrop = doc.createElement("div");
            backdrop.id = backdropId;
            backdrop.addEventListener("click", () => setSidebarOpen(false));
            doc.body.appendChild(backdrop);
          }
          return backdrop;
        }

        function setButtonIcon(open) {
          const btn = doc.getElementById(buttonId);
          if (!btn) return;
          btn.innerHTML = open
            ? `<span style="font-size:24px;line-height:1;color:#10243d;">×</span>`
            : `<span style="display:block;width:18px;height:2px;background:#10243d;border-radius:99px;box-shadow:0 6px 0 #10243d,0 -6px 0 #10243d;"></span>`;
        }

        function setSidebarOpen(open) {
          const sidebar = getSidebar();
          const backdrop = ensureBackdrop();
          if (!sidebar) return;
          sidebar.setAttribute("data-codex-open", open ? "true" : "false");
          backdrop.style.opacity = open ? "1" : "0";
          backdrop.style.pointerEvents = open ? "auto" : "none";
          setButtonIcon(open);
        }

        function toggleSidebar() {
          const sidebar = getSidebar();
          if (!sidebar) return;
          const isOpen = sidebar.getAttribute("data-codex-open") === "true";
          setSidebarOpen(!isOpen);
        }

        let btn = doc.getElementById(buttonId);
        if (!btn) {
          btn = doc.createElement("button");
          btn.id = buttonId;
          btn.type = "button";
          btn.setAttribute("aria-label", "Toggle workspace");
          btn.innerHTML = `
            <span style="display:block;width:18px;height:2px;background:#10243d;border-radius:99px;box-shadow:0 6px 0 #10243d,0 -6px 0 #10243d;"></span>
          `;
          Object.assign(btn.style, {
            position: "fixed",
            top: "14px",
            left: "14px",
            width: "44px",
            height: "44px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "14px",
            border: "1px solid rgba(14,34,57,0.08)",
            background: "rgba(255,255,255,0.92)",
            boxShadow: "0 16px 30px rgba(20,45,76,0.12)",
            backdropFilter: "blur(16px)",
            cursor: "pointer",
            zIndex: "10000",
            padding: "0",
            outline: "none"
          });
          btn.onmouseenter = () => { btn.style.transform = "translateY(-1px)"; };
          btn.onmouseleave = () => { btn.style.transform = "translateY(0)"; };
          btn.onclick = toggleSidebar;
          doc.body.appendChild(btn);
        }

        ensureBackdrop();
        setSidebarOpen(false);
        doc.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            setSidebarOpen(false);
          }
        });
        </script>
        """,
        height=0,
        width=0,
    )
def reset_chat_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {"role": "assistant", "content": "New session ready. Upload sources or continue exploring your documents."}
    ]
    st.session_state.last_result = None


def call_api_chat(query: str) -> dict[str, Any]:
    payload = json.dumps({"session_id": st.session_state.session_id, "query": query}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{st.session_state.api_base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def run_query(agent: HybridRAGAgent, query: str) -> dict[str, Any]:
    if st.session_state.ui_mode == "API":
        return call_api_chat(query)
    return agent.run(session_id=st.session_state.session_id, query=query)


def _pretty_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_trace_panel(result: dict[str, Any] | None) -> None:
    st.markdown(
        """
        <div class="compact-heading">
            <h3>Answer Trace</h3>
            <div class="compact-label">Evaluation & evidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result:
        st.info("Run a question and trace, evaluation, and evidence will appear here.")
        return

    metadata = result.get("metadata", {})
    evaluation = metadata.get("evaluation", {})
    runtime_metrics = metadata.get("metrics", {})
    reranker = metadata.get("reranker", {})
    timing_metrics = runtime_metrics.get("timings_ms", {})
    usage_metrics = runtime_metrics.get("usage", {})

    summary_cols = st.columns(2, gap="small")
    with summary_cols[0]:
        st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
    with summary_cols[1]:
        st.metric("Iterations", str(metadata.get("iterations", 1)))

    summary_cols = st.columns(2, gap="small")
    with summary_cols[0]:
        st.metric("Sources", str(len(result.get("sources", []))))
    with summary_cols[1]:
        st.metric("Ranking", "Fallback" if reranker and not reranker.get("available", True) else "Reranked")

    if reranker and not reranker.get("available", True):
        st.markdown(
            '<div class="warning-badge">Reranker disabled by environment.</div>',
            unsafe_allow_html=True,
        )

    if metadata.get("rewritten_query"):
        with st.expander("Rewritten query", expanded=False):
            st.code(metadata["rewritten_query"], language="text")

    eval_items = [(k, v) for k, v in evaluation.items() if isinstance(v, (int, float, bool))]
    if eval_items:
        st.caption("Evaluation")
        eval_cols = st.columns(2, gap="small")
        for idx, (key, value) in enumerate(eval_items):
            with eval_cols[idx % 2]:
                st.metric(_pretty_label(key), _format_metric_value(value))

    if timing_metrics:
        with st.expander("Timings", expanded=True):
            timing_cols = st.columns(2, gap="small")
            timing_items = list(timing_metrics.items())
            for idx, (key, value) in enumerate(timing_items):
                with timing_cols[idx % 2]:
                    st.metric(_pretty_label(key), f"{float(value):.0f} ms")
            if runtime_metrics.get("total_runtime_ms") is not None:
                st.metric("Total Runtime", f"{float(runtime_metrics['total_runtime_ms']):.0f} ms")

    if usage_metrics:
        with st.expander("Token usage", expanded=False):
            usage_cols = st.columns(2, gap="small")
            usage_items = list(usage_metrics.items())
            for idx, (key, value) in enumerate(usage_items):
                with usage_cols[idx % 2]:
                    st.metric(_pretty_label(key), _format_metric_value(value))
            if runtime_metrics.get("estimated_cost_usd") is not None:
                st.metric("Estimated Cost", f"${float(runtime_metrics['estimated_cost_usd']):.4f}")

    trace_path = metadata.get("trace_path")
    if trace_path:
        with st.expander("Trace file", expanded=False):
            st.code(trace_path, language="text")

    if evaluation:
        with st.expander("Full evaluation payload", expanded=False):
            st.json(evaluation)

    st.caption("Evidence")
    if result.get("sources"):
        for idx, source in enumerate(result["sources"], start=1):
            st.markdown(
                f"""
                <div class="source-shell">
                    <div class="badge">Score {source['score']:.3f}</div>
                    <div class="source-title">{idx}. {source['source']}</div>
                    <div class="source-meta">{source['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No evidence snippets yet.")


def render_header(stats: dict[str, Any]) -> None:
    st.markdown('<div class="hero-kicker">Future Minimal Retrieval</div>', unsafe_allow_html=True)
    head_left, head_right = st.columns([0.78, 0.22], gap="small", vertical_alignment="bottom")
    with head_left:
        st.markdown('<div class="hero-title">Quiet surfaces. Sharp retrieval.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subline">Hybrid search over your documents with grounded answers in a cleaner interface.</div>',
            unsafe_allow_html=True,
        )
    with head_right:
        st.markdown(f'<div class="mode-chip">{st.session_state.ui_mode}</div>', unsafe_allow_html=True)

    stat_cols = st.columns(4, gap="small")
    metrics = [
        ("Documents", str(stats["documents"])),
        ("Chunks", str(stats["chunks"])),
        ("Session", st.session_state.session_id[:8]),
        ("Mode", st.session_state.ui_mode),
    ]
    for col, (label, value) in zip(stat_cols, metrics):
        with col:
            st.metric(label, value)


def render_sidebar(kb: KnowledgeBase, memory: SessionMemoryManager) -> None:
    with st.sidebar:
        settings = st.session_state.runtime["settings"]
        st.markdown("## Workspace")
        st.caption("Minimal controls. Everything important within reach.")

        st.session_state.ui_mode = st.radio(
            "Execution Mode",
            ["In-Process", "API"],
            index=0 if st.session_state.ui_mode == "In-Process" else 1,
            label_visibility="visible",
        )
        if st.session_state.ui_mode == "API":
            st.session_state.api_base_url = st.text_input("API Base URL", value=st.session_state.api_base_url)

        if not settings.openrouter_api_key and st.session_state.ui_mode == "In-Process":
            st.warning("Add `OPENROUTER_API_KEY` in `.env` before local chat.")

        st.markdown("### Session")
        st.code(st.session_state.session_id, language="text")
        if st.button("New Session", use_container_width=True):
            reset_chat_session()
            st.rerun()

        sessions = memory.list_sessions()
        if sessions:
            selected_session = st.selectbox(
                "Load saved session",
                [item["session_id"] for item in sessions],
                label_visibility="visible",
            )
            if st.button("Load Session", use_container_width=True):
                exported = memory.export_session(selected_session)
                st.session_state.session_id = selected_session
                st.session_state.messages = [{"role": "assistant", "content": "Loaded saved session history."}]
                for turn in exported["history"]:
                    st.session_state.messages.append({"role": "user", "content": turn["query"]})
                    st.session_state.messages.append({"role": "assistant", "content": turn["response"]})
                st.session_state.last_result = None
                st.rerun()

        st.markdown("### Knowledge Base")
        stats = kb.get_stats()
        a, b = st.columns(2)
        a.metric("Docs", stats["documents"])
        b.metric("Chunks", stats["chunks"])
        if st.button("Clear Knowledge Base", use_container_width=True):
            removed = kb.clear_all()
            st.session_state.last_result = None
            st.success(f"Removed {removed} chunks.")
            st.rerun()


def render_chat_tab(agent: HybridRAGAgent, kb: KnowledgeBase) -> None:
    result = st.session_state.last_result
    chat_col, trace_col = st.columns([1.55, 0.9], gap="large")

    with chat_col:
        st.markdown(
            """
            <div class="compact-heading">
                <h3>Chat</h3>
                <div class="compact-label">Grounded Q&A</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about your documents")
        if prompt:
            if kb.get_stats()["chunks"] == 0:
                st.warning("Upload at least one PDF or article URL first.")
                return
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.status("Running pipeline...", expanded=True) as status:
                    st.write("Analyzing query")
                    st.write("Retrieving and merging context")
                    st.write("Ranking and generating answer")
                    try:
                        result = run_query(agent, prompt)
                    except urllib.error.URLError as exc:
                        st.error(f"API request failed: {exc}")
                        return
                    status.update(label="Answer ready", state="complete")
                st.markdown(result["answer"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.session_state.last_result = result
            st.rerun()

    with trace_col:
        render_trace_panel(result)


def render_ingest_tab(kb: KnowledgeBase) -> None:
    st.markdown(
        """
        <div class="compact-heading">
            <h3>Ingest</h3>
            <div class="compact-label">Source library</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        url_input = st.text_area("Article URLs", placeholder="https://example.com/post-1", height=180)
        if st.button("Process Sources", use_container_width=True, type="primary"):
            with st.status("Ingesting documents...", expanded=True) as status:
                st.write("Loading source content")
                summary = kb.ingest_sources(uploaded_files=uploaded_files or [], raw_urls=url_input)
                st.write("Chunking and indexing")
                status.update(label="Ingestion complete", state="complete")
            if summary["documents_processed"] > 0:
                st.success(
                    f"Processed {summary['documents_processed']} documents and created {summary['chunks_created']} chunks."
                )
            else:
                st.warning("No valid sources were provided.")

    with right:
        details = kb.get_source_details()
        if not details:
            st.info("No sources indexed yet.")
        for start in range(0, len(details), 2):
            row_items = details[start:start + 2]
            row_cols = st.columns(2, gap="medium")
            for col, item in zip(row_cols, row_items):
                with col:
                    st.markdown(
                        f"""
                        <div class="source-shell">
                            <div class="badge">{item['source_type'].upper()}</div>
                            <div class="source-title">{item['source']}</div>
                            <div class="source-meta">
                                {item['chunks']} chunks<br/>
                                {", ".join(item['sections'][:2]) if item['sections'] else "Document"}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    c1, c2 = st.columns([0.72, 0.28])
                    with c1:
                        with st.expander("Details", expanded=False):
                            st.json(item)
                    with c2:
                        if st.button("Remove", key=f"remove_{item['source']}", use_container_width=True):
                            removed = kb.remove_source(item["source"])
                            st.success(f"Removed {removed} chunks.")
                            st.rerun()


def render_sessions_tab(memory: SessionMemoryManager) -> None:
    st.markdown(
        """
        <div class="compact-heading">
            <h3>Sessions</h3>
            <div class="compact-label">Saved history</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sessions = memory.list_sessions()
    if not sessions:
        st.info("No saved sessions available yet.")
        return
    for item in sessions:
        st.markdown(
            f"""
            <div class="source-shell">
                <div class="badge">{item['turns']} turns</div>
                <div class="source-title">{item['session_id']}</div>
                <div class="source-meta">
                    {item['last_user_message'] or "No user message yet"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([0.22, 0.78])
        with c1:
            if st.button("Load", key=f"load_session_{item['session_id']}", use_container_width=True):
                exported = memory.export_session(item["session_id"])
                st.session_state.session_id = item["session_id"]
                st.session_state.messages = [{"role": "assistant", "content": "Loaded saved session history."}]
                for turn in exported["history"]:
                    st.session_state.messages.append({"role": "user", "content": turn["query"]})
                    st.session_state.messages.append({"role": "assistant", "content": turn["response"]})
                st.session_state.last_result = None
                st.rerun()
        with c2:
            with st.expander("Conversation details", expanded=False):
                st.json(memory.export_session(item["session_id"]))


def render_debug_tab(kb: KnowledgeBase) -> None:
    st.markdown(
        """
        <div class="compact-heading">
            <h3>Debug</h3>
            <div class="compact-label">Inspect internals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.0, 1.0], gap="large")
    result = st.session_state.last_result
    with left:
        if not result:
            st.info("Run a query to generate trace data.")
        else:
            with st.expander("Trace metadata", expanded=False):
                st.json(result["metadata"])
            if result["metadata"].get("trace_path"):
                st.caption("Trace file")
                st.code(result["metadata"]["trace_path"], language="text")
    with right:
        if result and result.get("sources"):
            for idx, source in enumerate(result["sources"], start=1):
                st.markdown(
                    f"""
                    <div class="source-shell">
                        <div class="badge">Score {source['score']:.3f}</div>
                        <div class="source-title">{idx}. {source['source']}</div>
                        <div class="source-meta">{source['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Evidence snippets will appear here after a query.")
        with st.expander("Registry preview", expanded=False):
            st.json(kb.get_registry()[:25])


def main() -> None:
    Path("storage").mkdir(exist_ok=True)
    ensure_state()
    kb, memory, agent = bootstrap()
    inject_styles()
    inject_sidebar_toggle()
    render_sidebar(kb, memory)
    render_header(kb.get_stats())
    tabs = st.tabs(["Chat", "Ingest", "Sessions", "Debug"])
    with tabs[0]:
        render_chat_tab(agent, kb)
    with tabs[1]:
        render_ingest_tab(kb)
    with tabs[2]:
        render_sessions_tab(memory)
    with tabs[3]:
        render_debug_tab(kb)


if __name__ == "__main__":
    main()
