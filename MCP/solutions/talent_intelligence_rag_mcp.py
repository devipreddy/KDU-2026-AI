"""
Talent Intelligence RAG MCP server.

Provides ingestion, retrieval, candidate similarity, gap analysis, and
interview-question generation over a small hiring knowledge base.
"""

import asyncio
import json
import os
from typing import Annotated, List, Optional

import mcp.server.stdio
import mcp.types as types
from dotenv import load_dotenv
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl, BaseModel, Field

from utils.api_utils import (
    JOB_DESCRIPTION_INVALID,
    PROMPT_NOT_FOUND,
    RESOURCE_NOT_FOUND,
    RESUME_NOT_FOUND,
    DomainError,
    INSUFFICIENT_EVIDENCE,
    TOOL_RESPONSE_SCHEMA,
    prompt_result,
    success_tool_result,
    text_resource_result,
    tool_result_from_exception,
)
from utils.provider_utils import init_langchain_provider_components
from utils.rag_utils import (
    explain_answer_sources,
    find_similar_candidates,
    generate_gap_analysis,
    generate_interview_questions,
    ingest_candidate_corpus,
    persist_query_result,
    query_talent_knowledge_base,
)
from utils.resume_utils import ensure_dir_exists, ensure_files_exist, list_resume_files, read_resume

load_dotenv()

server = Server("talent_intelligence_rag")

RESUME_DIR = os.environ.get("RESUME_DIR", "./assets")
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")
QUERY_STATE_DIR = os.environ.get("QUERY_STATE_DIR", "./query_state")
embeddings, llm, provider_config = init_langchain_provider_components(temperature=0)


class IngestCandidateCorpus(BaseModel):
    candidate_files: Annotated[
        Optional[List[str]],
        Field(description="Optional subset of resume PDF file names to ingest into the knowledge base"),
    ] = None
    job_descriptions: Annotated[
        Optional[List[str]],
        Field(description="Optional job descriptions to ingest as contextual artifacts"),
    ] = None
    notes: Annotated[
        Optional[List[str]],
        Field(description="Optional recruiter notes or interview observations to ingest"),
    ] = None


class QueryKnowledgeBase(BaseModel):
    query: Annotated[str, Field(description="Hiring question to answer from the knowledge base")]
    top_k: Annotated[int, Field(description="Number of supporting sources to retrieve", ge=1, le=10)] = 5


class FindSimilarCandidates(BaseModel):
    candidate_file: Annotated[str, Field(description="Reference resume PDF file name")]
    top_k: Annotated[int, Field(description="Number of similar candidates to return", ge=1, le=10)] = 3


class GenerateGapAnalysis(BaseModel):
    candidate_file: Annotated[str, Field(description="Resume PDF file name")]
    job_description: Annotated[str, Field(description="Job description to compare against")]


class GenerateInterviewQuestions(BaseModel):
    candidate_file: Annotated[str, Field(description="Resume PDF file name")]
    job_description: Annotated[str, Field(description="Job description to compare against")]
    question_count: Annotated[int, Field(description="Number of interview questions to generate", ge=1, le=10)] = 5


class ExplainAnswerSources(BaseModel):
    query_id: Annotated[str, Field(description="Identifier of a previous knowledge-base query")]


def _validate_candidate_files(candidate_files):
    if not candidate_files:
        return candidate_files

    missing = ensure_files_exist(candidate_files, RESUME_DIR)
    if missing:
        raise DomainError(RESUME_NOT_FOUND, f"Resume files not found: {', '.join(missing)}")
    return candidate_files


def _require_job_description(job_description):
    if not job_description or len(job_description.strip()) < 20:
        raise DomainError(JOB_DESCRIPTION_INVALID, "Job description must be present and at least 20 characters long.")


def _trace(source_count=None):
    trace = {
        "provider": provider_config.provider,
        "retrieval_mode": "dense" if embeddings else "heuristic",
        "llm_used": llm is not None,
    }
    if source_count is not None:
        trace["evidence_count"] = source_count
    return trace


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="ingest_candidate_corpus",
            description="Ingest resumes, job descriptions, and notes into the hiring knowledge base",
            inputSchema=IngestCandidateCorpus.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="query_talent_knowledge_base",
            description="Answer a hiring question using retrieved evidence from the knowledge base",
            inputSchema=QueryKnowledgeBase.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="find_similar_candidates",
            description="Find resumes similar to a reference candidate",
            inputSchema=FindSimilarCandidates.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="generate_gap_analysis",
            description="Identify candidate strengths and gaps against a job description",
            inputSchema=GenerateGapAnalysis.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="generate_interview_questions",
            description="Create evidence-backed interview questions for a candidate",
            inputSchema=GenerateInterviewQuestions.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="explain_answer_sources",
            description="Explain why the retrieved sources were selected for a previous query",
            inputSchema=ExplainAnswerSources.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
    ]


@server.list_resources()
async def list_resources():
    resources = [
        types.Resource(
            uri="kb://candidates",
            name="Candidate Knowledge Base",
            description="Summary of resumes currently ingested into the talent knowledge base",
            mimeType="application/json",
        ),
        types.Resource(
            uri="kb://jobs",
            name="Job Description Knowledge Base",
            description="Summary of job descriptions currently ingested into the talent knowledge base",
            mimeType="application/json",
        ),
        types.Resource(
            uri="kb://notes",
            name="Recruiter Notes Knowledge Base",
            description="Summary of recruiter notes currently ingested into the talent knowledge base",
            mimeType="application/json",
        ),
    ]

    for resume_file in list_resume_files(RESUME_DIR):
        candidate_id = os.path.splitext(resume_file)[0]
        resources.append(
            types.Resource(
                uri=f"kb://candidate/{candidate_id}",
                name=f"Candidate KB {candidate_id}",
                description="Resume text for a specific candidate",
                mimeType="application/json",
            )
        )

    for file_name in sorted(os.listdir(QUERY_STATE_DIR)) if os.path.exists(QUERY_STATE_DIR) else []:
        if not file_name.lower().endswith(".json"):
            continue
        query_id = os.path.splitext(file_name)[0]
        resources.append(
            types.Resource(
                uri=f"kb://query/{query_id}/sources",
                name=f"Query Sources {query_id}",
                description="Retrieved sources and explanations for a previous query",
                mimeType="application/json",
            )
        )

    return resources


@server.read_resource()
async def read_resource(uri: AnyUrl):
    resource_uri = str(uri)
    kb_path = os.path.join(os.path.abspath(KNOWLEDGE_BASE_DIR), "knowledge_base.json")
    kb_payload = {"artifacts": []}
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as handle:
            kb_payload = json.load(handle)

    if resource_uri == "kb://candidates":
        payload = [
            {
                "candidate_id": artifact.get("candidate_id"),
                "source": artifact.get("source"),
            }
            for artifact in kb_payload.get("artifacts", [])
            if artifact.get("artifact_type") == "resume"
        ]
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri == "kb://jobs":
        payload = [
            {"artifact_id": artifact.get("artifact_id"), "source": artifact.get("source"), "content": artifact.get("content")}
            for artifact in kb_payload.get("artifacts", [])
            if artifact.get("artifact_type") == "job_description"
        ]
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri == "kb://notes":
        payload = [
            {"artifact_id": artifact.get("artifact_id"), "source": artifact.get("source"), "content": artifact.get("content")}
            for artifact in kb_payload.get("artifacts", [])
            if artifact.get("artifact_type") == "note"
        ]
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri.startswith("kb://candidate/"):
        candidate_id = resource_uri[len("kb://candidate/") :]
        for artifact in kb_payload.get("artifacts", []):
            if artifact.get("artifact_type") == "resume" and artifact.get("candidate_id") == candidate_id:
                return text_resource_result(resource_uri, json.dumps(artifact, indent=2))
        raise DomainError(RESOURCE_NOT_FOUND, f"Candidate knowledge resource not found: {resource_uri}")

    if resource_uri.startswith("kb://query/") and resource_uri.endswith("/sources"):
        query_id = resource_uri[len("kb://query/") : -len("/sources")]
        payload = explain_answer_sources(QUERY_STATE_DIR, query_id)
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    raise DomainError(RESOURCE_NOT_FOUND, f"Unknown resource: {resource_uri}")


@server.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="ask_hiring_copilot",
            description="Guide the model to answer a hiring question using the talent knowledge base",
            arguments=[types.PromptArgument(name="query", description="Hiring question", required=True)],
        ),
        types.Prompt(
            name="generate_interview_focus_areas",
            description="Guide the model to create interview focus areas for a candidate",
            arguments=[
                types.PromptArgument(name="candidate_file", description="Resume PDF file name", required=True),
                types.PromptArgument(name="job_description", description="Job description text", required=True),
            ],
        ),
        types.Prompt(
            name="discover_candidates_for_new_role",
            description="Guide the model to discover candidates for a new role using the talent corpus",
            arguments=[types.PromptArgument(name="query", description="Role or talent search query", required=True)],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None):
    arguments = arguments or {}

    if name == "ask_hiring_copilot":
        text = (
            "Use the `query_talent_knowledge_base` tool for the following hiring question.\n"
            f"Query: {arguments.get('query', '')}\n"
            "Then summarize the answer and clearly cite the most important retrieved sources."
        )
        return prompt_result("Ask the hiring copilot a grounded question", text)

    if name == "generate_interview_focus_areas":
        text = (
            "Use the `generate_gap_analysis` and `generate_interview_questions` tools.\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            f"Job description: {arguments.get('job_description', '')}\n"
            "Summarize the candidate's strongest areas and the top risks to probe."
        )
        return prompt_result("Generate interview focus areas from retrieved evidence", text)

    if name == "discover_candidates_for_new_role":
        text = (
            "Use the `query_talent_knowledge_base` tool to search the talent corpus.\n"
            f"Query: {arguments.get('query', '')}\n"
            "Return grounded recommendations and mention which candidates should be reviewed next."
        )
        return prompt_result("Discover candidates for a new role", text)

    raise DomainError(PROMPT_NOT_FOUND, f"Unknown prompt: {name}")


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "ingest_candidate_corpus":
            args = IngestCandidateCorpus(**arguments)
            payload = ingest_candidate_corpus(
                kb_dir=KNOWLEDGE_BASE_DIR,
                resume_dir=RESUME_DIR,
                candidate_files=_validate_candidate_files(args.candidate_files),
                job_descriptions=args.job_descriptions,
                notes=args.notes,
            )
            return success_tool_result(
                data=payload,
                trace=_trace(),
                narrative="Ingested artifacts into the talent knowledge base.",
            )

        if name == "query_talent_knowledge_base":
            args = QueryKnowledgeBase(**arguments)
            payload = query_talent_knowledge_base(
                kb_dir=KNOWLEDGE_BASE_DIR,
                query=args.query,
                embeddings=embeddings,
                llm=llm,
                top_k=args.top_k,
            )
            stored = persist_query_result(QUERY_STATE_DIR, payload)
            status = "success" if stored.get("sources") else "degraded"
            warnings = [] if stored.get("sources") else [{"code": INSUFFICIENT_EVIDENCE, "message": "No sources were retrieved."}]
            return success_tool_result(
                data=stored,
                run_id=stored["query_id"],
                warnings=warnings,
                trace=_trace(source_count=len(stored.get("sources", []))),
                narrative="Answered a grounded talent intelligence query.",
                status=status,
            )

        if name == "find_similar_candidates":
            args = FindSimilarCandidates(**arguments)
            _validate_candidate_files([args.candidate_file])
            payload = {
                "candidate_file": args.candidate_file,
                "similar_candidates": find_similar_candidates(
                    kb_dir=KNOWLEDGE_BASE_DIR,
                    candidate_file=args.candidate_file,
                    resume_dir=RESUME_DIR,
                    embeddings=embeddings,
                    top_k=args.top_k,
                ),
            }
            return success_tool_result(
                data=payload,
                trace=_trace(source_count=len(payload["similar_candidates"])),
                narrative="Found similar candidates in the talent corpus.",
            )

        if name == "generate_gap_analysis":
            args = GenerateGapAnalysis(**arguments)
            _validate_candidate_files([args.candidate_file])
            _require_job_description(args.job_description)
            payload = generate_gap_analysis(
                candidate_file=args.candidate_file,
                job_description=args.job_description,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
            )
            return success_tool_result(
                data=payload,
                trace=_trace(source_count=len(payload.get("evidence", []))),
                narrative="Generated candidate gap analysis against the job description.",
            )

        if name == "generate_interview_questions":
            args = GenerateInterviewQuestions(**arguments)
            _validate_candidate_files([args.candidate_file])
            _require_job_description(args.job_description)
            payload = generate_interview_questions(
                candidate_file=args.candidate_file,
                job_description=args.job_description,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                question_count=args.question_count,
            )
            return success_tool_result(
                data=payload,
                trace=_trace(source_count=len(payload.get("evidence", []))),
                narrative="Generated evidence-backed interview questions.",
            )

        if name == "explain_answer_sources":
            args = ExplainAnswerSources(**arguments)
            payload = explain_answer_sources(QUERY_STATE_DIR, args.query_id)
            return success_tool_result(
                data=payload,
                run_id=args.query_id,
                trace=_trace(source_count=len(payload.get("source_explanations", []))),
                narrative="Explained why the query sources were selected.",
            )

        raise McpError(types.INVALID_PARAMS, f"Unknown tool: {name}")
    except ValueError as exc:
        raise McpError(types.INVALID_PARAMS, str(exc))
    except DomainError as exc:
        return tool_result_from_exception(exc)
    except Exception as exc:
        return tool_result_from_exception(exc)


async def main():
    ensure_dir_exists(RESUME_DIR)
    ensure_dir_exists(KNOWLEDGE_BASE_DIR)
    ensure_dir_exists(QUERY_STATE_DIR)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="talent_intelligence_rag",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
