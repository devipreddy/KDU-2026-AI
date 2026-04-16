"""
Explainable Shortlisting MCP server.

Ranks and compares candidates while exposing the evidence behind each decision.
"""

import asyncio
import json
import os
from typing import Annotated, List

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
    TOOL_RESPONSE_SCHEMA,
    prompt_result,
    success_tool_result,
    text_resource_result,
    tool_result_from_exception,
)
from utils.langchain_utils import prepare_resume_documents
from utils.provider_utils import init_langchain_provider_components
from utils.resume_utils import ensure_dir_exists, ensure_files_exist, list_resume_files, read_resume
from utils.shortlisting_utils import (
    compare_candidates,
    explain_candidate_fit,
    generate_shortlist,
    rank_candidates,
    show_candidate_evidence,
)
from utils.state_utils import JsonRecordStore

load_dotenv()

server = Server("resume_shortlister_explainable")

RESUME_DIR = os.environ.get("RESUME_DIR", "./assets")
RANKING_STATE_DIR = os.environ.get("RANKING_STATE_DIR", "./ranking_state")
embeddings, llm, provider_config = init_langchain_provider_components(temperature=0)
ranking_store = JsonRecordStore(RANKING_STATE_DIR)


class RankCandidates(BaseModel):
    job_description: Annotated[str, Field(description="Job description used for ranking candidates")]
    candidate_files: Annotated[List[str], Field(description="Resume PDF file names to evaluate")]
    top_k: Annotated[int, Field(description="Number of candidates to return", ge=1, le=20)] = 5
    scoring_profile: Annotated[
        str,
        Field(description="Ranking profile: balanced, skills_first, experience_first, or entry_level"),
    ] = "balanced"


class CompareCandidates(BaseModel):
    job_description: Annotated[str, Field(description="Job description used for candidate comparison")]
    candidate_files: Annotated[List[str], Field(description="Two or more resume PDF file names to compare")]
    scoring_profile: Annotated[
        str,
        Field(description="Comparison profile: balanced, skills_first, experience_first, or entry_level"),
    ] = "balanced"


class ExplainFit(BaseModel):
    file_path: Annotated[str, Field(description="Resume PDF file name to explain")]
    job_description: Annotated[str, Field(description="Job description used for fit explanation")]
    scoring_profile: Annotated[
        str,
        Field(description="Explanation profile: balanced, skills_first, experience_first, or entry_level"),
    ] = "balanced"


class ShowEvidence(BaseModel):
    file_path: Annotated[str, Field(description="Resume PDF file name to inspect")]
    job_description: Annotated[str, Field(description="Job description requirement or full job description")]


class GenerateShortlist(BaseModel):
    job_description: Annotated[str, Field(description="Job description used to generate the shortlist")]
    candidate_files: Annotated[List[str], Field(description="Resume PDF file names to shortlist")]
    top_k: Annotated[int, Field(description="Number of shortlisted candidates to return", ge=1, le=20)] = 3
    scoring_profile: Annotated[
        str,
        Field(description="Shortlist profile: balanced, skills_first, experience_first, or entry_level"),
    ] = "balanced"


def _require_job_description(job_description):
    if not job_description or len(job_description.strip()) < 20:
        raise DomainError(JOB_DESCRIPTION_INVALID, "Job description must be present and at least 20 characters long.")


def _require_candidate_files(candidate_files):
    missing = ensure_files_exist(candidate_files, RESUME_DIR)
    if missing:
        raise DomainError(RESUME_NOT_FOUND, f"Resume files not found: {', '.join(missing)}")
    return candidate_files


def _trace(mode=None, evidence_count=None):
    trace = {
        "provider": provider_config.provider,
        "retrieval_mode": "dense" if embeddings else "heuristic",
        "llm_used": llm is not None,
    }
    if mode is not None:
        trace["mode"] = mode
    if evidence_count is not None:
        trace["evidence_count"] = evidence_count
    return trace


def _persist_ranking_payload(payload, record_type):
    stored = dict(payload)
    stored["record_type"] = record_type
    stored["job_id"] = payload["run_id"]
    ranking_store.create(stored, record_id=payload["run_id"])
    return stored


def _resource_resume_items():
    resources = []
    for resume_file in list_resume_files(RESUME_DIR):
        candidate_id = os.path.splitext(resume_file)[0]
        resources.append(
            types.Resource(
                uri=f"resume://{candidate_id}",
                name=f"Resume: {candidate_id}",
                description=f"Full extracted resume for {candidate_id}",
                mimeType="application/json",
            )
        )
        resources.append(
            types.Resource(
                uri=f"resume://{candidate_id}/chunks",
                name=f"Resume Chunks: {candidate_id}",
                description=f"Chunked resume sections for {candidate_id}",
                mimeType="application/json",
            )
        )
    return resources


def _resource_ranking_items():
    resources = []
    for record in ranking_store.list_records():
        run_id = record.get("run_id") or record.get("record_id")
        if not run_id:
            continue

        resources.append(
            types.Resource(
                uri=f"ranking://{run_id}",
                name=f"Ranking Run {run_id}",
                description="Persisted explainable ranking result",
                mimeType="application/json",
            )
        )
        resources.append(
            types.Resource(
                uri=f"job://{run_id}",
                name=f"Job Description {run_id}",
                description="Job description associated with a ranking run",
                mimeType="application/json",
            )
        )
        for candidate in record.get("ranked_candidates", []) or record.get("shortlist", []):
            candidate_id = os.path.splitext(os.path.basename(candidate["candidate_file"]))[0]
            resources.append(
                types.Resource(
                    uri=f"evidence://{run_id}/{candidate_id}",
                    name=f"Evidence {run_id}/{candidate_id}",
                    description="Evidence passages supporting the ranking decision",
                    mimeType="application/json",
                )
            )
    return resources


def _candidate_file_from_id(candidate_id):
    preferred = f"{candidate_id}.pdf"
    available = list_resume_files(RESUME_DIR)
    if preferred in available:
        return preferred
    for resume_file in available:
        if os.path.splitext(resume_file)[0] == candidate_id:
            return resume_file
    raise DomainError(RESUME_NOT_FOUND, f"Candidate resource not found: {candidate_id}")


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="rank_candidates",
            description="Rank multiple resumes against a job description with explainable evidence",
            inputSchema=RankCandidates.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="compare_candidates",
            description="Compare multiple candidates side by side for a specific role",
            inputSchema=CompareCandidates.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="explain_fit",
            description="Explain why a specific candidate matches a job description",
            inputSchema=ExplainFit.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="show_evidence",
            description="Show the resume sections that best support a requirement or job description",
            inputSchema=ShowEvidence.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="generate_shortlist",
            description="Generate a concise recruiter shortlist with recommendation and rationale",
            inputSchema=GenerateShortlist.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
    ]


@server.list_resources()
async def list_resources():
    return _resource_resume_items() + _resource_ranking_items()


@server.read_resource()
async def read_resource(uri: AnyUrl):
    resource_uri = str(uri)

    if resource_uri.startswith("resume://") and resource_uri.endswith("/chunks"):
        candidate_id = resource_uri[len("resume://") : -len("/chunks")]
        file_path = _candidate_file_from_id(candidate_id)
        resume_text = read_resume(file_path, RESUME_DIR)
        if not resume_text:
            raise DomainError(RESUME_NOT_FOUND, f"Could not read resume for resource {resource_uri}")
        chunks = prepare_resume_documents(resume_text, file_path)["chunks"]
        payload = [
            {
                "chunk_index": chunk.metadata.get("chunk_index"),
                "source": chunk.metadata.get("source"),
                "text": chunk.page_content,
            }
            for chunk in chunks
        ]
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri.startswith("resume://"):
        candidate_id = resource_uri[len("resume://") :]
        file_path = _candidate_file_from_id(candidate_id)
        resume_text = read_resume(file_path, RESUME_DIR)
        if not resume_text:
            raise DomainError(RESUME_NOT_FOUND, f"Could not read resume for resource {resource_uri}")
        payload = {
            "candidate_id": candidate_id,
            "candidate_file": file_path,
            "resume_text": resume_text,
        }
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri.startswith("ranking://"):
        run_id = resource_uri[len("ranking://") :]
        record = ranking_store.load(run_id)
        return text_resource_result(resource_uri, json.dumps(record, indent=2))

    if resource_uri.startswith("job://"):
        run_id = resource_uri[len("job://") :]
        record = ranking_store.load(run_id)
        payload = {"job_id": run_id, "job_description": record.get("job_description")}
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri.startswith("evidence://"):
        _, path = resource_uri.split("://", 1)
        run_id, candidate_id = path.split("/", 1)
        record = ranking_store.load(run_id)
        candidates = record.get("ranked_candidates") or record.get("shortlist") or []
        for candidate in candidates:
            if os.path.splitext(os.path.basename(candidate["candidate_file"]))[0] == candidate_id:
                payload = {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "candidate_file": candidate["candidate_file"],
                    "evidence": candidate.get("evidence", []),
                    "matched_keywords": candidate.get("matched_keywords", []),
                    "missing_keywords": candidate.get("missing_keywords", []),
                }
                return text_resource_result(resource_uri, json.dumps(payload, indent=2))
        raise DomainError(RESOURCE_NOT_FOUND, f"Evidence resource not found: {resource_uri}")

    raise DomainError(RESOURCE_NOT_FOUND, f"Unknown resource: {resource_uri}")


@server.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="shortlist_for_role",
            description="Guide the model to generate a shortlist for a role using the server tools",
            arguments=[
                types.PromptArgument(name="job_description", description="Job description text", required=True),
                types.PromptArgument(name="candidate_files", description="Comma-separated resume file names", required=True),
                types.PromptArgument(name="top_k", description="Optional shortlist size", required=False),
            ],
        ),
        types.Prompt(
            name="justify_candidate_ranking",
            description="Guide the model to justify a candidate ranking using explain_fit or ranking resources",
            arguments=[
                types.PromptArgument(name="candidate_file", description="Resume PDF file name", required=True),
                types.PromptArgument(name="job_description", description="Job description text", required=True),
            ],
        ),
        types.Prompt(
            name="summarize_candidate_vs_jd",
            description="Guide the model to summarize how a candidate maps to a job description",
            arguments=[
                types.PromptArgument(name="candidate_file", description="Resume PDF file name", required=True),
                types.PromptArgument(name="job_description", description="Job description text", required=True),
            ],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None):
    arguments = arguments or {}

    if name == "shortlist_for_role":
        text = (
            "Use the `generate_shortlist` tool with the following inputs.\n"
            f"Job description: {arguments.get('job_description', '')}\n"
            f"Candidate files: {arguments.get('candidate_files', '')}\n"
            f"Top K: {arguments.get('top_k', '3')}\n"
            "After the tool returns, summarize the shortlist in recruiter language and cite the evidence-backed rationale."
        )
        return prompt_result("Shortlist candidates for a role", text)

    if name == "justify_candidate_ranking":
        text = (
            "Use the `explain_fit` tool to justify the candidate ranking.\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            f"Job description: {arguments.get('job_description', '')}\n"
            "Focus on evidence passages, matched keywords, missing keywords, and confidence."
        )
        return prompt_result("Justify why a candidate ranked the way they did", text)

    if name == "summarize_candidate_vs_jd":
        text = (
            "Use the `explain_fit` tool and then write a concise recruiter summary.\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            f"Job description: {arguments.get('job_description', '')}\n"
            "Include strengths, gaps, and final recommendation."
        )
        return prompt_result("Summarize candidate fit against a job description", text)

    raise DomainError(PROMPT_NOT_FOUND, f"Unknown prompt: {name}")


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "rank_candidates":
            args = RankCandidates(**arguments)
            _require_job_description(args.job_description)
            candidate_files = _require_candidate_files(args.candidate_files)
            payload = rank_candidates(
                job_description=args.job_description,
                candidate_files=candidate_files,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                top_k=args.top_k,
                scoring_profile=args.scoring_profile,
            )
            stored = _persist_ranking_payload(payload, record_type="ranking")
            return success_tool_result(
                data=stored,
                run_id=stored["run_id"],
                trace=_trace(
                    mode=stored["ranked_candidates"][0]["mode"] if stored["ranked_candidates"] else None,
                    evidence_count=sum(len(candidate.get("evidence", [])) for candidate in stored["ranked_candidates"]),
                ),
                narrative="Ranked candidates with explainable evidence.",
            )

        if name == "compare_candidates":
            args = CompareCandidates(**arguments)
            _require_job_description(args.job_description)
            candidate_files = _require_candidate_files(args.candidate_files)
            payload = compare_candidates(
                job_description=args.job_description,
                candidate_files=candidate_files,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                scoring_profile=args.scoring_profile,
            )
            return success_tool_result(
                data=payload,
                run_id=payload["run_id"],
                trace=_trace(),
                narrative="Compared candidates side by side.",
            )

        if name == "explain_fit":
            args = ExplainFit(**arguments)
            _require_job_description(args.job_description)
            _require_candidate_files([args.file_path])
            payload = explain_candidate_fit(
                file_path=args.file_path,
                job_description=args.job_description,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                scoring_profile=args.scoring_profile,
            )
            return success_tool_result(
                data=payload,
                trace=_trace(mode=payload.get("mode"), evidence_count=len(payload.get("evidence", []))),
                narrative="Generated explainable fit assessment for the candidate.",
            )

        if name == "show_evidence":
            args = ShowEvidence(**arguments)
            _require_job_description(args.job_description)
            _require_candidate_files([args.file_path])
            payload = show_candidate_evidence(
                file_path=args.file_path,
                job_description=args.job_description,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
            )
            return success_tool_result(
                data=payload,
                trace=_trace(evidence_count=len(payload.get("evidence", []))),
                narrative="Retrieved top evidence passages for the candidate.",
            )

        if name == "generate_shortlist":
            args = GenerateShortlist(**arguments)
            _require_job_description(args.job_description)
            candidate_files = _require_candidate_files(args.candidate_files)
            payload = generate_shortlist(
                job_description=args.job_description,
                candidate_files=candidate_files,
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                top_k=args.top_k,
                scoring_profile=args.scoring_profile,
            )
            stored = _persist_ranking_payload(payload, record_type="shortlist")
            return success_tool_result(
                data=stored,
                run_id=stored["run_id"],
                trace=_trace(
                    evidence_count=sum(len(candidate.get("matched_keywords", [])) for candidate in stored["shortlist"])
                ),
                narrative="Generated recruiter shortlist with recommendation and rationale.",
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
    ensure_dir_exists(RANKING_STATE_DIR)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="resume_shortlister_explainable",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
