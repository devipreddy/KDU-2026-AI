from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import streamlit as st

from app.config import Settings
from app.schemas import ChatRequest
from app.services.assistant import AssistantService


QUICK_ACTIONS = [
    "What is the weather in Bengaluru right now?",
    "Calculate (45 * 73) / sqrt(9)",
    "Search the latest OpenAI announcements",
    "Compare today's AI headlines with the weather in Delhi",
]


def create_assistant_service() -> AssistantService:
    return AssistantService(Settings())


def load_settings() -> Settings:
    return Settings()


def ensure_session_state() -> None:
    settings = load_settings()
    defaults = {
        "session_id": str(uuid4()),
        "messages": [],
        "trace_log": [],
        "pending_prompt": None,
        "latest_route": "-",
        "latest_model": settings.default_model,
        "latest_tools": [],
        "latest_latency_ms": "-",
        "latest_tokens": {"input": 0, "output": 0, "total": 0},
        "latest_cost": {"amount": 0.0, "currency": settings.model_currency},
        "latest_status": "Idle",
        "latest_fallback_mode": None,
        "latest_warning": None,
        "latest_breakers": {
            "search": {
                "name": "serper",
                "state": "closed",
                "failure_rate": 0.0,
                "sample_size": 0,
                "last_error": None,
                "recovery_timeout_seconds": settings.circuit_breaker_recovery_timeout_seconds,
                "retry_after_seconds": 0,
            },
            "weather": {
                "name": "weather",
                "state": "closed",
                "failure_rate": 0.0,
                "sample_size": 0,
                "last_error": None,
                "recovery_timeout_seconds": settings.circuit_breaker_recovery_timeout_seconds,
                "retry_after_seconds": 0,
            },
        },
        "aggregate_requests": 0,
        "aggregate_latency_ms": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_sse_block(block: str) -> tuple[str, dict[str, object]]:
    event_name = "message"
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    payload = json.loads("\n".join(data_lines)) if data_lines else {}
    return event_name, payload


def iter_service_events(service: AssistantService, payload: ChatRequest):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen = service.stream_chat(payload)
    try:
        while True:
            try:
                block = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            yield parse_sse_block(block)
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:
            pass
        try:
            loop.run_until_complete(service.close())
        except Exception:
            pass
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f4f8ff;
            --surface: rgba(255, 255, 255, 0.9);
            --surface-strong: #ffffff;
            --ink: #0f172a;
            --muted: #475569;
            --line: rgba(37, 99, 235, 0.12);
            --blue: #2563eb;
            --blue-soft: rgba(37, 99, 235, 0.1);
            --green: #16a34a;
            --green-soft: rgba(22, 163, 74, 0.12);
            --red: #dc2626;
            --red-soft: rgba(220, 38, 38, 0.12);
            --shadow: 0 18px 50px rgba(37, 99, 235, 0.08);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
        }

        html, body, [class*="css"]  {
            font-family: 'Manrope', sans-serif;
            overflow-y: auto !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.1), transparent 24%),
                radial-gradient(circle at 88% 14%, rgba(22, 163, 74, 0.08), transparent 22%),
                linear-gradient(180deg, #f8fbff 0%, #f4f8ff 100%);
            color: var(--ink);
            overflow-y: auto;
        }

        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
            overflow-y: auto;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stToolbar"], #MainMenu, footer {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stSidebar"] {
            background: rgba(248, 251, 255, 0.82);
            border-right: 1px solid rgba(37, 99, 235, 0.08);
            backdrop-filter: blur(14px);
        }

        [data-testid="stSidebarContent"] {
            overflow-y: auto;
            padding-bottom: 2rem;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 8rem;
            max-width: 1120px;
        }

        .top-shell {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.15rem;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: var(--shadow);
            margin-bottom: 0.85rem;
        }

        .top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .app-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0;
        }

        .app-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0.15rem 0 0;
        }

        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            color: var(--blue);
            background: var(--blue-soft);
        }

        .panel-card {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            padding: 0.9rem 0.95rem;
            backdrop-filter: blur(14px);
            margin-bottom: 0.75rem;
        }

        .panel-heading {
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--blue);
            margin-bottom: 0.6rem;
            font-weight: 800;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }

        .metric-tile {
            border-radius: var(--radius-md);
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(37, 99, 235, 0.08);
            padding: 0.72rem 0.8rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--ink);
        }

        .breaker-stack {
            display: grid;
            gap: 0.55rem;
        }

        .breaker-tile {
            border-radius: var(--radius-md);
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(37, 99, 235, 0.08);
            padding: 0.8rem;
        }

        .breaker-top {
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
            align-items: center;
            margin-bottom: 0.25rem;
        }

        .breaker-name {
            font-weight: 800;
            text-transform: capitalize;
            font-size: 0.84rem;
        }

        .breaker-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .breaker-pill.closed {
            background: var(--green-soft);
            color: var(--green);
        }

        .breaker-pill.open {
            background: var(--red-soft);
            color: var(--red);
        }

        .breaker-pill.half_open {
            background: var(--blue-soft);
            color: var(--blue);
        }

        .trace-body {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.45;
        }

        .trace-stack {
            display: grid;
            gap: 0.45rem;
            max-height: 18rem;
            overflow-y: auto;
            padding-right: 0.2rem;
        }

        .trace-item {
            border-radius: 12px;
            border: 1px solid rgba(37, 99, 235, 0.08);
            background: rgba(255,255,255,0.92);
            padding: 0.65rem 0.72rem;
        }

        .trace-title {
            color: var(--blue);
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: 0.18rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.12);
            background: rgba(255,255,255,0.92);
            color: var(--blue);
            padding: 0.58rem 0.9rem;
            font-weight: 700;
            box-shadow: none;
        }

        div.stButton > button:hover {
            border-color: rgba(37, 99, 235, 0.28);
            color: var(--blue);
        }

        [data-testid="stChatInput"] {
            background: rgba(255,255,255,0.96);
            border: 2px solid rgba(37, 99, 235, 0.12);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
            padding: 0.08rem;
        }

        [data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 0.18rem 0.22rem;
            margin-bottom: 0.35rem;
        }

        [data-testid="stChatMessageContent"] {
            border-radius: 18px;
            padding: 0.15rem 0.05rem 0.12rem;
            color: var(--ink);
        }

        .message-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            margin-top: 0.45rem;
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 800;
            border: 1px solid rgba(37, 99, 235, 0.1);
            background: rgba(255,255,255,0.94);
            color: var(--muted);
        }

        .meta-pill.route {
            color: var(--blue);
            background: var(--blue-soft);
            border-color: rgba(37, 99, 235, 0.06);
        }

        .meta-pill.tool {
            color: var(--green);
            background: var(--green-soft);
            border-color: rgba(22, 163, 74, 0.06);
        }

        .meta-pill.error {
            color: var(--red);
            background: var(--red-soft);
            border-color: rgba(220, 38, 38, 0.06);
        }

        .mini-trace {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }

        .chat-spacer {
            height: 4rem;
        }

        @media (max-width: 920px) {
            .block-container {
                padding-bottom: 9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="top-shell">
            <div class="top-row">
                <div>
                    <h1 class="app-title">Multi-Function AI Assistant</h1>
                    <p class="app-copy">Search, weather, calculator.</p>
                </div>
                <div class="tag-row">
                    <span class="tag">search</span>
                    <span class="tag">weather</span>
                    <span class="tag">calculator</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str) -> str:
    return (
        '<div class="metric-tile">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
    )


def render_status_panel(target) -> None:
    html = (
        '<div class="panel-card">'
        '<div class="panel-heading">Current Run</div>'
        '<div class="metric-grid">'
        f'{metric_tile("Status", str(st.session_state.latest_status))}'
        f'{metric_tile("Route", str(st.session_state.latest_route))}'
        f'{metric_tile("Model", str(st.session_state.latest_model))}'
        f'{metric_tile("Tools", ", ".join(st.session_state.latest_tools) or "-")}'
        f'{metric_tile("LLM", "fallback" if st.session_state.latest_fallback_mode else "ok")}'
        f'{metric_tile("Latency", f"{st.session_state.latest_latency_ms} ms" if st.session_state.latest_latency_ms != "-" else "-")}'
        f'{metric_tile("Session", st.session_state.session_id[:8])}'
        "</div>"
        "</div>"
    )
    target.markdown(html, unsafe_allow_html=True)


def render_usage_panel(target) -> None:
    latest_tokens = st.session_state.latest_tokens
    latest_cost = st.session_state.latest_cost
    latest_cost_display = f"${float(latest_cost.get('amount', 0.0)):.6f}"
    request_count = int(st.session_state.aggregate_requests)
    avg_latency = (
        st.session_state.aggregate_latency_ms / request_count
        if request_count
        else 0
    )
    avg_latency_display = f"{avg_latency:.2f} ms"
    html = (
        '<div class="panel-card">'
        '<div class="panel-heading">Usage and Cost</div>'
        '<div class="metric-grid">'
        f'{metric_tile("Input Tokens", str(latest_tokens.get("input", 0)))}'
        f'{metric_tile("Output Tokens", str(latest_tokens.get("output", 0)))}'
        f'{metric_tile("Total Tokens", str(latest_tokens.get("total", 0)))}'
        f'{metric_tile("Cost", latest_cost_display)}'
        f'{metric_tile("Requests", str(request_count))}'
        f'{metric_tile("Avg Latency", avg_latency_display)}'
        "</div>"
        "</div>"
    )
    target.markdown(html, unsafe_allow_html=True)


def render_breakers_panel(target) -> None:
    breakers = st.session_state.latest_breakers
    tiles = []
    for name, snapshot in breakers.items():
        state = str(snapshot.get("state", "closed"))
        failure_rate = float(snapshot.get("failure_rate", 0.0))
        retry_after = int(snapshot.get("retry_after_seconds", 0))
        details = f"Failure rate {failure_rate:.2f}"
        if retry_after:
            details += f" | Retry after {retry_after}s"
        last_error = snapshot.get("last_error")
        if last_error:
            details += f" | Last error: {last_error}"
        tiles.append(
            '<div class="breaker-tile">'
            '<div class="breaker-top">'
            f'<div class="breaker-name">{name}</div>'
            f'<div class="breaker-pill {state}">{state.replace("_", " ")}</div>'
            "</div>"
            f'<div class="trace-body">{details}</div>'
            "</div>"
        )
    target.markdown(
        '<div class="panel-card"><div class="panel-heading">Circuit Breakers</div>'
        f'<div class="breaker-stack">{"".join(tiles)}</div></div>',
        unsafe_allow_html=True,
    )


def render_trace_panel(target) -> None:
    traces = st.session_state.trace_log[-8:]
    if not traces:
        traces = [{"title": "Waiting", "body": "No events yet."}]
    items = [
        '<div class="trace-item">'
        f'<div class="trace-title">{item["title"]}</div>'
        f'<div class="trace-body">{item["body"]}</div>'
        "</div>"
        for item in reversed(traces)
    ]
    target.markdown(
        '<div class="panel-card"><div class="panel-heading">Trace</div>'
        f'<div class="trace-stack">{"".join(items)}</div></div>',
        unsafe_allow_html=True,
    )


def add_trace(title: str, body: str) -> None:
    st.session_state.trace_log.append({"title": title, "body": body})


def refresh_panels(placeholders: dict[str, st.delta_generator.DeltaGenerator]) -> None:
    render_status_panel(placeholders["status"])
    render_usage_panel(placeholders["usage"])
    render_breakers_panel(placeholders["breakers"])
    render_trace_panel(placeholders["trace"])


def reset_run_state() -> None:
    settings = load_settings()
    st.session_state.latest_route = "-"
    st.session_state.latest_model = settings.default_model
    st.session_state.latest_tools = []
    st.session_state.latest_latency_ms = "-"
    st.session_state.latest_tokens = {"input": 0, "output": 0, "total": 0}
    st.session_state.latest_cost = {"amount": 0.0, "currency": settings.model_currency}
    st.session_state.latest_status = "Queued"
    st.session_state.latest_fallback_mode = None
    st.session_state.latest_warning = None


def _coerce_tools(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        return [str(item) for item in list(value) if item]
    except TypeError:
        return []


def build_message_meta() -> dict[str, object]:
    return {
        "status": st.session_state.latest_status,
        "route": st.session_state.latest_route,
        "tools": list(st.session_state.latest_tools),
        "latency_ms": st.session_state.latest_latency_ms,
        "tokens": int(st.session_state.latest_tokens.get("total", 0)),
        "cost": float(st.session_state.latest_cost.get("amount", 0.0)),
        "fallback_mode": st.session_state.latest_fallback_mode,
        "warning": st.session_state.latest_warning,
    }


def render_message_meta(target, meta: dict[str, object]) -> None:
    pills: list[str] = []
    route = str(meta.get("route", "-"))
    if route and route != "-":
        pills.append(f'<span class="meta-pill route">route {route}</span>')

    tools = _coerce_tools(meta.get("tools"))
    if tools:
        pills.append(f'<span class="meta-pill tool">tools {", ".join(tools)}</span>')

    latency_ms = meta.get("latency_ms")
    if latency_ms not in {None, "-"}:
        pills.append(f'<span class="meta-pill">{latency_ms} ms</span>')

    tokens = meta.get("tokens")
    if isinstance(tokens, int) and tokens > 0:
        pills.append(f'<span class="meta-pill">{tokens} tokens</span>')

    cost = float(meta.get("cost", 0.0) or 0.0)
    if cost > 0:
        pills.append(f'<span class="meta-pill">${cost:.6f}</span>')

    status = str(meta.get("status", ""))
    if status == "Error":
        pills.append('<span class="meta-pill error">error</span>')
    if meta.get("fallback_mode") == "tool_only":
        pills.append('<span class="meta-pill error">llm fallback</span>')

    if not pills:
        return

    target.markdown(
        f'<div class="message-meta">{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )
    warning = str(meta.get("warning") or "").strip()
    if warning:
        target.markdown(
            f'<div class="mini-trace">LLM unavailable, showing tool-grounded output. {warning}</div>',
            unsafe_allow_html=True,
        )


def handle_event(
    event_name: str,
    payload: dict[str, object],
    placeholders: dict[str, st.delta_generator.DeltaGenerator],
) -> None:
    if event_name == "start":
        st.session_state.latest_status = "Planning"
        add_trace("Request", str(payload.get("request_id", "")))
    elif event_name == "route":
        st.session_state.latest_route = str(payload.get("route", "-"))
        st.session_state.latest_model = str(payload.get("model", "-"))
        tool = payload.get("tool")
        if tool:
            st.session_state.latest_tools = [str(tool)]
        add_trace(
            "Route",
            f'{payload.get("route", "unknown")} via {payload.get("model", "unknown model")}',
        )
    elif event_name == "tool_call":
        st.session_state.latest_status = "Using tools"
        tool = str(payload.get("tool", "tool"))
        if tool not in st.session_state.latest_tools:
            st.session_state.latest_tools.append(tool)
        add_trace("Tool Call", f"{tool} with {json.dumps(payload.get('input', {}))}")
    elif event_name == "cache_hit":
        st.session_state.latest_status = "Using cache"
        add_trace("Cache", f"Reused cached data for {payload.get('tool', 'tool')}")
    elif event_name == "tool_result":
        st.session_state.latest_status = "Synthesizing"
        tools = _coerce_tools(payload.get("tools"))
        tool_list = ", ".join(tools) if tools else "tool"
        add_trace("Tool Result", f"{tool_list} returned successfully")
    elif event_name == "warning":
        st.session_state.latest_status = "Fallback"
        st.session_state.latest_fallback_mode = payload.get("fallback_mode")
        st.session_state.latest_warning = str(payload.get("message", "LLM step failed."))
        stage = str(payload.get("stage", "llm"))
        add_trace("LLM Fallback", f"{stage}: {st.session_state.latest_warning}")
    elif event_name == "usage":
        st.session_state.latest_tokens = {
            "input": int(payload.get("input", 0)),
            "output": int(payload.get("output", 0)),
            "total": int(payload.get("total", 0)),
        }
    elif event_name == "cost":
        st.session_state.latest_cost = {
            "amount": float(payload.get("amount", 0.0)),
            "currency": str(payload.get("currency", "USD")),
        }
    elif event_name == "end":
        st.session_state.latest_status = "Idle"
        st.session_state.latest_latency_ms = int(payload.get("latency_ms", 0))
        st.session_state.latest_tools = _coerce_tools(payload.get("tools"))
        st.session_state.latest_fallback_mode = payload.get("fallback_mode")
        st.session_state.aggregate_requests += 1
        st.session_state.aggregate_latency_ms += int(payload.get("latency_ms", 0))
        add_trace("Completed", f"{payload.get('status', 'complete')} in {payload.get('latency_ms', '-') } ms")
    elif event_name == "error":
        st.session_state.latest_status = "Error"
        add_trace("Error", str(payload.get("message", "Unknown error")))
    refresh_panels(placeholders)


def stream_assistant_reply(
    prompt: str,
    placeholders: dict[str, st.delta_generator.DeltaGenerator],
) -> tuple[str, dict[str, object]]:
    service = create_assistant_service()
    payload = ChatRequest(
        query=prompt,
        session_id=st.session_state.session_id,
        metadata={"source": "streamlit"},
    )
    response_text = ""

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        meta_placeholder = st.empty()
        for event_name, payload_data in iter_service_events(service, payload):
            handle_event(event_name, payload_data, placeholders)
            render_message_meta(meta_placeholder, build_message_meta())
            if event_name == "token":
                token = str(payload_data.get("text", ""))
                response_text += token
                stream_placeholder.markdown(f"{response_text}|")
            elif event_name == "error":
                message = str(payload_data.get("message", "Something went wrong."))
                stream_placeholder.error(message)
                st.session_state.latest_breakers = service.get_circuit_breaker_status()
                refresh_panels(placeholders)
                final_meta = build_message_meta()
                render_message_meta(meta_placeholder, final_meta)
                return message, final_meta
        stream_placeholder.markdown(response_text)
        final_meta = build_message_meta()
        render_message_meta(meta_placeholder, final_meta)

    st.session_state.latest_breakers = service.get_circuit_breaker_status()
    refresh_panels(placeholders)
    return response_text, final_meta


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            meta = message.get("meta")
            if meta:
                render_message_meta(st.empty(), meta)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Function AI Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_state()
    render_global_styles()

    sidebar_placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}
    with st.sidebar:
        sidebar_placeholders["status"] = st.empty()
        sidebar_placeholders["usage"] = st.empty()
        sidebar_placeholders["breakers"] = st.empty()
        sidebar_placeholders["trace"] = st.empty()

    refresh_panels(sidebar_placeholders)
    render_header()

    action_columns = st.columns(len(QUICK_ACTIONS))
    for index, action in enumerate(QUICK_ACTIONS):
        if action_columns[index].button(action, key=f"quick_action_{index}"):
            st.session_state.pending_prompt = action

    render_history()

    st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)

    typed_prompt = st.chat_input(
        "Ask anything..."
    )
    prompt = typed_prompt or st.session_state.pending_prompt
    if not prompt:
        return

    st.session_state.pending_prompt = None

    reset_run_state()
    refresh_panels(sidebar_placeholders)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_reply, assistant_meta = stream_assistant_reply(prompt, sidebar_placeholders)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply,
            "meta": assistant_meta,
        }
    )


if __name__ == "__main__":
    main()
