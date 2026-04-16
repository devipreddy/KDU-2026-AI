"""
Enhanced Resume Shortlister MCP Tool with LangChain

This tool extends the basic resume shortlister with LangChain capabilities for
resume analysis, skill extraction, and job matching.
"""

import asyncio
import os
from typing import Annotated

import mcp.server.stdio
import mcp.types as types
from dotenv import load_dotenv
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field

from utils.api_utils import JOB_DESCRIPTION_INVALID, RESUME_NOT_FOUND, TOOL_RESPONSE_SCHEMA, DomainError, success_tool_result, tool_result_from_exception
from utils.langchain_utils import extract_skills_with_langchain
from utils.provider_utils import init_langchain_provider_components
from utils.resume_utils import ensure_dir_exists, ensure_files_exist
from utils.shortlisting_utils import build_candidate_analysis

load_dotenv()

server = Server("resume_shortlister_enhanced")

RESUME_DIR = os.environ.get("RESUME_DIR", "./assets")
embeddings, llm, provider_config = init_langchain_provider_components(temperature=0)


class MatchResume(BaseModel):
    file_path: Annotated[str, Field(description="Path to the resume PDF file")]
    job_description: Annotated[str, Field(description="Job description to match against")]


class ExtractSkills(BaseModel):
    file_path: Annotated[str, Field(description="Path to the resume PDF file")]


def _validate_resume(file_path):
    missing = ensure_files_exist([file_path], RESUME_DIR)
    if missing:
        raise DomainError(RESUME_NOT_FOUND, f"Resume file not found: {file_path}")


def _validate_job_description(job_description):
    if not job_description or len(job_description.strip()) < 20:
        raise DomainError(JOB_DESCRIPTION_INVALID, "Job description must be present and at least 20 characters long.")


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="match_resume",
            description="Match a resume against a job description",
            inputSchema=MatchResume.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="extract_skills",
            description="Extract skills from a resume",
            inputSchema=ExtractSkills.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "match_resume":
            args = MatchResume(**arguments)
            _validate_resume(args.file_path)
            _validate_job_description(args.job_description)

            payload = build_candidate_analysis(
                file_path=args.file_path,
                resume_dir=RESUME_DIR,
                job_description=args.job_description,
                embeddings=embeddings,
                llm=llm,
                scoring_profile="balanced",
            )
            payload["provider"] = provider_config.provider
            return success_tool_result(
                data=payload,
                trace={
                    "provider": provider_config.provider,
                    "retrieval_mode": "dense" if embeddings else "heuristic",
                    "llm_used": llm is not None,
                    "evidence_count": len(payload.get("evidence", [])),
                },
                narrative="Matched resume against the job description.",
            )

        if name == "extract_skills":
            args = ExtractSkills(**arguments)
            _validate_resume(args.file_path)

            from utils.resume_utils import read_resume

            resume_text = read_resume(args.file_path, RESUME_DIR)
            if not resume_text:
                raise DomainError(RESUME_NOT_FOUND, f"Failed to read resume: {args.file_path}")

            payload = {
                "file_path": args.file_path,
                "provider": provider_config.provider,
                "skills": extract_skills_with_langchain(resume_text, llm),
            }
            return success_tool_result(
                data=payload,
                trace={
                    "provider": provider_config.provider,
                    "retrieval_mode": "none",
                    "llm_used": llm is not None,
                },
                narrative="Extracted skills from the resume.",
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

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="resume_shortlister_enhanced",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
