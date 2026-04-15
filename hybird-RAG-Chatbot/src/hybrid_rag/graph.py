from __future__ import annotations

import json
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .config import Settings
from .constants import UNKNOWN_ANSWER
from .evaluation import ResponseEvaluator
from .ingestion import KnowledgeBase
from .memory import SessionMemoryManager
from .observability import TraceStore
from .prompts import ANSWER_PROMPT, QUERY_ANALYZER_PROMPT, QUERY_REWRITE_PROMPT, SUMMARY_PROMPT
from .retrieval import HybridRetriever
from .schemas import GraphState


class HybridRAGAgent:
    def __init__(
        self,
        settings: Settings,
        knowledge_base: KnowledgeBase,
        memory: SessionMemoryManager,
        trace_store: TraceStore,
    ) -> None:
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.memory = memory
        self.trace_store = trace_store
        self.llm = ChatOpenAI(
            model=self.settings.openrouter_model,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            temperature=0.1,
            max_tokens=self.settings.llm_max_tokens,
            default_headers={
                "HTTP-Referer": self.settings.openrouter_site_url,
                "X-Title": self.settings.openrouter_app_name,
            },
        )
        self.retriever = HybridRetriever(
            settings=self.settings,
            vectorstore=self.knowledge_base.vectorstore,
            registry=self.knowledge_base.get_registry(),
        )
        self.evaluator = ResponseEvaluator(
            confidence_threshold=self.settings.confidence_threshold,
            min_context_coverage=self.settings.min_context_coverage,
        )
        self.app = self._build_graph()

    def _serialize_history(self, chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return "No prior chat history."
        return "\n".join(f"{item['role']}: {item['content']}" for item in chat_history[-6:])

    def _usage_from_response(self, response: Any) -> dict[str, float]:
        usage = getattr(response, "usage_metadata", None) or {}
        response_meta = getattr(response, "response_metadata", {}) or {}
        token_usage = response_meta.get("token_usage", {})
        input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", token_usage.get("total_tokens", input_tokens + output_tokens))
        estimated_cost = (
            (input_tokens / 1000) * self.settings.input_token_cost_per_1k
            + (output_tokens / 1000) * self.settings.output_token_cost_per_1k
        )
        return {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(total_tokens),
            "estimated_cost": float(estimated_cost),
        }

    def _track_metric(self, state: GraphState, name: str, duration_ms: float, usage: dict[str, float] | None = None) -> dict[str, Any]:
        metrics = dict(state.get("metadata", {}).get("metrics", {}))
        timings = dict(metrics.get("timings_ms", {}))
        timings[name] = round(duration_ms, 2)
        usage_totals = dict(metrics.get("usage", {}))
        if usage:
            usage_totals["input_tokens"] = usage_totals.get("input_tokens", 0.0) + usage.get("input_tokens", 0.0)
            usage_totals["output_tokens"] = usage_totals.get("output_tokens", 0.0) + usage.get("output_tokens", 0.0)
            usage_totals["total_tokens"] = usage_totals.get("total_tokens", 0.0) + usage.get("total_tokens", 0.0)
            usage_totals["estimated_cost"] = usage_totals.get("estimated_cost", 0.0) + usage.get("estimated_cost", 0.0)
        metrics["timings_ms"] = timings
        metrics["usage"] = usage_totals
        return metrics

    def _invoke_llm(self, state: GraphState, metric_name: str, messages: list[Any]) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter()
        max_tokens = self.settings.llm_max_tokens
        if metric_name in {"analyze_query", "rewrite_query"}:
            max_tokens = self.settings.llm_analysis_max_tokens
        elif metric_name == "update_memory":
            max_tokens = self.settings.llm_summary_max_tokens
        response = self.llm.invoke(messages, max_tokens=max_tokens)
        duration_ms = (time.perf_counter() - started) * 1000
        usage = self._usage_from_response(response)
        metrics = self._track_metric(state, metric_name, duration_ms, usage)
        return response, metrics

    def load_memory(self, state: GraphState) -> GraphState:
        session = self.memory.load(state["session_id"])
        return {
            **state,
            "chat_history": session.chat_history,
            "conversation_summary": session.summary,
            "iteration_count": state.get("iteration_count", 0),
            "metadata": state.get("metadata", {}),
        }

    def analyze_query(self, state: GraphState) -> GraphState:
        response, metrics = self._invoke_llm(
            state,
            "analyze_query",
            QUERY_ANALYZER_PROMPT.format_messages(
                summary=state["conversation_summary"],
                chat_history=self._serialize_history(state["chat_history"]),
                query=state["query"],
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        needs_rewrite = False
        try:
            parsed = json.loads(content)
            needs_rewrite = bool(parsed.get("needs_rewrite", False))
        except json.JSONDecodeError:
            lowered = state["query"].lower()
            needs_rewrite = any(token in lowered for token in ["it", "they", "them", "that", "those", "its"])
        return {
            **state,
            "query_needs_rewrite": needs_rewrite,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def route_query(self, state: GraphState) -> str:
        return "rewrite_query" if state.get("query_needs_rewrite") else "hybrid_retrieve"

    def rewrite_query(self, state: GraphState) -> GraphState:
        response, metrics = self._invoke_llm(
            state,
            "rewrite_query",
            QUERY_REWRITE_PROMPT.format_messages(
                summary=state["conversation_summary"],
                chat_history=self._serialize_history(state["chat_history"]),
                query=state["query"],
            ),
        )
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
        return {
            **state,
            "rewritten_query": rewritten.strip(),
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
                "rewritten_query": rewritten.strip(),
            },
        }

    def hybrid_retrieve(self, state: GraphState) -> GraphState:
        self.retriever.refresh_registry(self.knowledge_base.get_registry())
        active_query = state.get("rewritten_query") or state["query"]
        started = time.perf_counter()
        retrieved = self.retriever.hybrid_search(active_query)
        metrics = self._track_metric(state, "hybrid_retrieve", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "retrieved_docs": retrieved,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def deduplicate_results(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        deduplicated = self.retriever.deduplicate(state.get("retrieved_docs", []))
        metrics = self._track_metric(state, "deduplicate_results", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "deduplicated_docs": deduplicated,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def rerank_documents(self, state: GraphState) -> GraphState:
        active_query = state.get("rewritten_query") or state["query"]
        started = time.perf_counter()
        reranked = self.retriever.rerank(active_query, state.get("deduplicated_docs", []))
        filtered = self.retriever.filter_relevant(reranked)
        metrics = self._track_metric(state, "rerank_documents", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "reranked_docs": filtered,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def select_top_k(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        selected = self.retriever.select_top_k(state.get("reranked_docs", []))
        metrics = self._track_metric(state, "select_top_k", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "selected_context": selected,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def construct_prompt(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        context_parts = []
        for index, doc in enumerate(state.get("selected_context", []), start=1):
            source = doc["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {index}: {source}]\n{doc['page_content']}")
        prompt_context = "\n\n".join(context_parts) if context_parts else "No relevant context retrieved."
        metrics = self._track_metric(state, "construct_prompt", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
                "prompt_context": prompt_context,
            },
        }

    def generate_answer(self, state: GraphState) -> GraphState:
        response, metrics = self._invoke_llm(
            state,
            "generate_answer",
            ANSWER_PROMPT.format_messages(
                summary=state["conversation_summary"],
                context=state["metadata"]["prompt_context"],
                query=state["query"],
            ),
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {
            **state,
            "answer": answer.strip(),
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def evaluate_answer(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        evaluation = self.evaluator.evaluate(
            query=state["query"],
            answer=state.get("answer", ""),
            selected_context=state.get("selected_context", []),
        )
        answer = state.get("answer", "").strip()
        if evaluation["needs_disclaimer"] and answer and answer != UNKNOWN_ANSWER:
            answer = f"{answer}\n\nNote: This answer has limited support in the retrieved context."
        metrics = self._track_metric(state, "evaluate_answer", (time.perf_counter() - started) * 1000)
        return {
            **state,
            "answer": answer or UNKNOWN_ANSWER,
            "confidence": evaluation["confidence"],
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
                "evaluation": {
                    **evaluation["details"],
                    "needs_disclaimer": evaluation["needs_disclaimer"],
                },
            },
        }

    def evaluation_router(self, state: GraphState) -> str:
        if (
            state.get("confidence", 0.0) < self.settings.confidence_threshold
            and state.get("iteration_count", 0) < self.settings.max_iterations
        ):
            return "hybrid_retrieve"
        return "update_memory"

    def update_memory(self, state: GraphState) -> GraphState:
        history = state.get("chat_history", []) + [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": state.get("answer", UNKNOWN_ANSWER)},
        ]
        summary_response, metrics = self._invoke_llm(
            state,
            "update_memory",
            SUMMARY_PROMPT.format_messages(chat_history=self._serialize_history(history))
        )
        summary = summary_response.content if isinstance(summary_response.content, str) else str(summary_response.content)
        self.memory.update(
            session_id=state["session_id"],
            query=state["query"],
            answer=state.get("answer", UNKNOWN_ANSWER),
            summary=summary.strip(),
        )
        return {
            **state,
            "conversation_summary": summary.strip(),
            "metadata": {
                **state.get("metadata", {}),
                "metrics": metrics,
            },
        }

    def _trace_payload(self, result: GraphState) -> dict[str, Any]:
        return {
            "session_id": result.get("session_id"),
            "query": result.get("query"),
            "rewritten_query": result.get("rewritten_query"),
            "conversation_summary": result.get("conversation_summary"),
            "iteration_count": result.get("iteration_count"),
            "confidence": result.get("confidence"),
            "answer": result.get("answer"),
            "retrieved_docs": result.get("retrieved_docs", []),
            "deduplicated_docs": result.get("deduplicated_docs", []),
            "reranked_docs": result.get("reranked_docs", []),
            "selected_context": result.get("selected_context", []),
            "metadata": result.get("metadata", {}),
        }

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_memory", self.load_memory)
        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("hybrid_retrieve", self.hybrid_retrieve)
        graph.add_node("deduplicate_results", self.deduplicate_results)
        graph.add_node("rerank_documents", self.rerank_documents)
        graph.add_node("select_top_k", self.select_top_k)
        graph.add_node("construct_prompt", self.construct_prompt)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("evaluate_answer", self.evaluate_answer)
        graph.add_node("update_memory", self.update_memory)

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            self.route_query,
            {"rewrite_query": "rewrite_query", "hybrid_retrieve": "hybrid_retrieve"},
        )
        graph.add_edge("rewrite_query", "hybrid_retrieve")
        graph.add_edge("hybrid_retrieve", "deduplicate_results")
        graph.add_edge("deduplicate_results", "rerank_documents")
        graph.add_edge("rerank_documents", "select_top_k")
        graph.add_edge("select_top_k", "construct_prompt")
        graph.add_edge("construct_prompt", "generate_answer")
        graph.add_edge("generate_answer", "evaluate_answer")
        graph.add_conditional_edges(
            "evaluate_answer",
            self.evaluation_router,
            {"hybrid_retrieve": "hybrid_retrieve", "update_memory": "update_memory"},
        )
        graph.add_edge("update_memory", END)
        return graph.compile()

    def run(self, session_id: str, query: str) -> dict[str, Any]:
        state: GraphState = {
            "session_id": session_id,
            "query": query,
            "rewritten_query": "",
            "iteration_count": 0,
            "metadata": {},
        }
        result = self.app.invoke(state)
        total_runtime_ms = sum(result.get("metadata", {}).get("metrics", {}).get("timings_ms", {}).values())
        result.setdefault("metadata", {})
        result["metadata"].setdefault("metrics", {})
        result["metadata"]["metrics"]["total_runtime_ms"] = round(total_runtime_ms, 2)
        trace_path = self.trace_store.write_trace(session_id, self._trace_payload(result))
        sources = [
            {
                "text": doc["page_content"][:300] + ("..." if len(doc["page_content"]) > 300 else ""),
                "source": doc["metadata"].get("source", "unknown"),
                "score": float(doc.get("combined_score", 0.0)),
            }
            for doc in result.get("selected_context", [])
        ]
        return {
            "answer": result.get("answer", UNKNOWN_ANSWER),
            "sources": sources,
            "confidence": float(result.get("confidence", 0.0)),
            "metadata": {
                "iterations": result.get("iteration_count", 1),
                "rewritten_query": result.get("metadata", {}).get("rewritten_query", ""),
                "trace_path": trace_path,
                "evaluation": result.get("metadata", {}).get("evaluation", {}),
                "metrics": result.get("metadata", {}).get("metrics", {}),
                "reranker": self.retriever.get_reranker_status(),
            },
        }
