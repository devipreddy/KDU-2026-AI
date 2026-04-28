from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Content Accessibility Suite", layout="wide")

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
API_TIMEOUT_SECONDS = 300


def build_headers(api_key: str | None) -> dict[str, str]:
    if api_key:
        return {"X-API-Key": api_key}
    return {}


def api_get(base_url: str, path: str, api_key: str | None = None) -> Any:
    response = requests.get(
        f"{base_url.rstrip('/')}{path}",
        timeout=API_TIMEOUT_SECONDS,
        headers=build_headers(api_key),
    )
    response.raise_for_status()
    return response.json()


def api_post_file(base_url: str, uploaded_file, force_reprocess: bool, api_key: str | None = None) -> Any:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {"force_reprocess": str(force_reprocess).lower()}
    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/files/process",
        files=files,
        data=data,
        timeout=API_TIMEOUT_SECONDS,
        headers=build_headers(api_key),
    )
    response.raise_for_status()
    return response.json()


def api_post_json(base_url: str, path: str, payload: dict[str, Any], api_key: str | None = None) -> Any:
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
        headers=build_headers(api_key),
    )
    response.raise_for_status()
    return response.json()


def load_files(base_url: str, api_key: str | None) -> list[dict[str, Any]]:
    return api_get(base_url, "/api/v1/files", api_key)


def load_file(base_url: str, file_id: str, api_key: str | None) -> dict[str, Any]:
    return api_get(base_url, f"/api/v1/files/{file_id}", api_key)


def load_job(base_url: str, job_id: str, api_key: str | None) -> dict[str, Any]:
    return api_get(base_url, f"/api/v1/files/jobs/{job_id}", api_key)


def load_costs(base_url: str, api_key: str | None) -> dict[str, Any]:
    return api_get(base_url, "/api/v1/costs", api_key)


def render_sidebar() -> tuple[str, str, Any, bool, bool]:
    with st.sidebar:
        st.title("Accessibility Suite")
        backend_url = st.text_input("FastAPI URL", value=DEFAULT_BACKEND_URL)
        api_key = st.text_input("API Key", value="", type="password")
        uploaded_file = st.file_uploader(
            "Upload a PDF, image, or audio file",
            type=["pdf", "jpg", "jpeg", "png", "mp3", "wav"],
        )
        force_reprocess = st.checkbox("Reprocess matching file hash", value=False)
        process_clicked = st.button("Process File", type="primary", use_container_width=True)

        st.caption("Primary models: GPT-4o-mini vision/text, text-embedding-3-small, local Whisper.")
    return backend_url, api_key, uploaded_file, force_reprocess, process_clicked


def main() -> None:
    backend_url, api_key, uploaded_file, force_reprocess, process_clicked = render_sidebar()
    st.session_state.setdefault("active_file_id", None)
    st.session_state.setdefault("search_results", None)
    st.session_state.setdefault("active_job_id", None)

    if process_clicked and uploaded_file is not None:
        with st.spinner("Submitting file to the background processing queue..."):
            try:
                response = api_post_file(backend_url, uploaded_file, force_reprocess, api_key)
                st.session_state["active_job_id"] = response["job"]["id"]
                if response["cached"]:
                    if response.get("file"):
                        st.session_state["active_file_id"] = response["file"]["id"]
                    st.sidebar.success("Matched an existing processed file. Reused cached outputs.")
                else:
                    st.sidebar.success("Job queued. Use Refresh Job Status to watch progress.")
            except requests.HTTPError as exc:
                message = exc.response.text if exc.response is not None else str(exc)
                st.sidebar.error(f"Processing failed: {message}")
            except Exception as exc:
                st.sidebar.error(f"Processing failed: {exc}")

    current_job = None
    if st.session_state.get("active_job_id"):
        try:
            current_job = load_job(backend_url, st.session_state["active_job_id"], api_key)
            if current_job.get("file"):
                st.session_state["active_file_id"] = current_job["file"]["id"]
            if current_job["job"]["status"] == "completed" and current_job.get("file"):
                st.session_state["active_job_id"] = None
        except Exception:
            current_job = None

    left, right = st.columns([1.1, 2.2], gap="large")

    with left:
        if current_job:
            st.subheader("Current Job")
            st.write(f"**{current_job['job']['file_name']}**")
            st.caption(current_job["job"]["status"].title())
            if current_job["job"].get("progress_message"):
                st.write(current_job["job"]["progress_message"])
            if current_job["job"].get("error_message"):
                st.error(current_job["job"]["error_message"])
            st.button("Refresh Job Status", use_container_width=True)

        st.subheader("Processed Files")
        try:
            files = load_files(backend_url, api_key)
        except Exception as exc:
            st.error(f"Could not reach the backend: {exc}")
            return

        if not files:
            if current_job:
                st.info("Processing is underway. Refresh job status to load results when complete.")
            else:
                st.info("No files processed yet. Upload something from the sidebar to begin.")
            return

        file_lookup = {item["id"]: item for item in files}
        current_file_id = st.session_state["active_file_id"] or files[0]["id"]
        if current_file_id not in file_lookup:
            current_file_id = files[0]["id"]

        selected_file_id = st.selectbox(
            "Choose a file",
            options=[item["id"] for item in files],
            format_func=lambda file_id: _file_option_label(file_lookup[file_id]),
            index=[item["id"] for item in files].index(current_file_id),
        )
        st.session_state["active_file_id"] = selected_file_id

        selected_summary = file_lookup[selected_file_id]
        st.metric("Status", selected_summary["status"].title())
        st.metric("Chunks Indexed", selected_summary["chunk_count"])
        st.write("**Tags**")
        if selected_summary["topic_tags"]:
            st.write(", ".join(selected_summary["topic_tags"]))
        else:
            st.caption("Tags will appear after processing completes.")

    with right:
        current_file = load_file(backend_url, st.session_state["active_file_id"], api_key)
        render_main_panel(backend_url, api_key, current_file)


def render_main_panel(backend_url: str, api_key: str | None, current_file: dict[str, Any]) -> None:
    st.title(current_file.get("title") or current_file["file_name"])
    st.caption(f"{current_file['file_type'].upper()} | {current_file['status'].title()} | {current_file['chunk_count']} chunks")

    file_cost = sum(call["estimated_cost_usd"] for call in current_file["api_calls"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated File Cost", f"${file_cost:.6f}")
    col2.metric("API Calls", len(current_file["api_calls"]))
    col3.metric("Transcript Length", f"{len((current_file.get('cleaned_text') or '').split()):,} words")

    tabs = st.tabs(["Transcript", "Summary", "Key Points", "Search", "Costs"])

    with tabs[0]:
        transcript = current_file.get("cleaned_text") or current_file.get("extracted_text") or ""
        st.text_area("Accessible text", value=transcript, height=420)
        page_details = current_file.get("extraction_metadata", {}).get("page_details", [])
        if page_details:
            st.write("**Page Routing**")
            st.dataframe(pd.DataFrame(page_details), use_container_width=True, hide_index=True)
        speaker_segments = current_file.get("extraction_metadata", {}).get("speaker_segments", [])
        if speaker_segments:
            st.write("**Speaker Segments**")
            st.dataframe(pd.DataFrame(speaker_segments), use_container_width=True, hide_index=True)

    with tabs[1]:
        if current_file.get("description"):
            st.write("**Description**")
            st.write(current_file["description"])
        st.write("**Summary**")
        st.write(current_file.get("summary") or "Summary unavailable.")
        if current_file.get("topic_tags"):
            st.write("**Topic Tags**")
            st.write(", ".join(current_file["topic_tags"]))

    with tabs[2]:
        key_points = current_file.get("key_points") or []
        if not key_points:
            st.info("Key points are not available yet.")
        for point in key_points:
            st.markdown(f"- {point}")

    with tabs[3]:
        scope = st.radio("Search scope", options=["Current file", "All files"], horizontal=True)
        query = st.text_input("Semantic query", placeholder="Find sections about termination policy")
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)
        if st.button("Run Search", use_container_width=True):
            if not query.strip():
                st.warning("Enter a query to search.")
            else:
                payload = {
                    "query": query.strip(),
                    "file_id": current_file["id"] if scope == "Current file" else None,
                    "top_k": top_k,
                }
                try:
                    st.session_state["search_results"] = api_post_json(backend_url, "/api/v1/search", payload, api_key)
                except requests.HTTPError as exc:
                    message = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Search failed: {message}")
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

        results = st.session_state.get("search_results")
        if results and results.get("hits"):
            for index, hit in enumerate(results["hits"], start=1):
                st.write(f"**{index}. {hit['file_name']}**")
                st.caption(f"Score {hit['score']:.4f}" + (f" | Page {hit['page_number']}" if hit.get("page_number") else ""))
                st.write(hit["content"])
                with st.expander("Context"):
                    st.write(hit["context"])
        elif results and not results.get("hits"):
            st.info("No matching chunks found.")

    with tabs[4]:
        costs = load_costs(backend_url, api_key)
        st.write("**Current File Calls**")
        if current_file["api_calls"]:
            st.dataframe(
                pd.DataFrame(current_file["api_calls"])[
                    ["operation", "provider", "model", "input_tokens", "output_tokens", "estimated_cost_usd"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No API usage recorded for this file.")

        st.write("**Overall Cost Dashboard**")
        overall_cols = st.columns(3)
        overall_cols[0].metric("Total Cost", f"${costs['total_cost_usd']:.6f}")
        overall_cols[1].metric("Processed Files", len(costs["files"]))
        overall_cols[2].metric("Tracked Operations", len(costs["by_operation"]))

        if costs["files"]:
            st.write("**Per File Breakdown**")
            st.dataframe(pd.DataFrame(costs["files"]), use_container_width=True, hide_index=True)
        if costs["by_operation"]:
            st.write("**By Operation**")
            st.dataframe(pd.DataFrame(costs["by_operation"]), use_container_width=True, hide_index=True)
        if costs["by_model"]:
            st.write("**By Model**")
            st.dataframe(pd.DataFrame(costs["by_model"]), use_container_width=True, hide_index=True)


def _file_option_label(item: dict[str, Any]) -> str:
    title = item.get("title") or item["file_name"]
    return f"{title} ({item['file_type']}, {item['status']})"


if __name__ == "__main__":
    main()
