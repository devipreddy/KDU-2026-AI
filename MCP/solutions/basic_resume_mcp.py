"""
Basic Resume Shortlister MCP Tool

This tool allows Claude to view and access resume PDFs for shortlisting candidates.
"""

import asyncio
import os
from typing import Annotated

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field

from utils.api_utils import RESUME_NOT_FOUND, TOOL_RESPONSE_SCHEMA, DomainError, success_tool_result, tool_result_from_exception
from utils.resume_utils import ensure_dir_exists, ensure_files_exist, list_resume_files, read_resume

server = Server("resume_shortlister")

RESUME_DIR = os.environ.get("RESUME_DIR", "./assets")


class ReadResume(BaseModel):
    file_path: Annotated[str, Field(description="Path to the resume PDF file")]


class ListResumes(BaseModel):
    pass


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="read_resume",
            description="Read and extract text from a resume PDF file",
            inputSchema=ReadResume.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="list_resumes",
            description="List all available resume files",
            inputSchema=ListResumes.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "read_resume":
            args = ReadResume(**arguments)
            missing = ensure_files_exist([args.file_path], RESUME_DIR)
            if missing:
                raise DomainError(RESUME_NOT_FOUND, f"Resume file not found: {args.file_path}")

            resume_text = read_resume(args.file_path, RESUME_DIR)
            if not resume_text:
                raise DomainError(RESUME_NOT_FOUND, f"Could not read resume at {args.file_path}")

            payload = {
                "file_path": args.file_path,
                "resume_text": resume_text,
            }
            return success_tool_result(data=payload, narrative="Read and extracted resume text.")

        if name == "list_resumes":
            resume_files = list_resume_files(RESUME_DIR)
            payload = {
                "resume_count": len(resume_files),
                "resume_files": resume_files,
            }
            return success_tool_result(data=payload, narrative="Listed available resume files.")

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
                server_name="resume_shortlister",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
