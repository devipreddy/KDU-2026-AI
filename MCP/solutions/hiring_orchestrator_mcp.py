"""
Hiring Orchestrator MCP server.

Coordinates shortlist generation with recruiter-facing workflow artifacts.
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
    DOWNSTREAM_CONNECTOR_ERROR,
    PROMPT_NOT_FOUND,
    RESOURCE_NOT_FOUND,
    RESUME_NOT_FOUND,
    TOOL_RESPONSE_SCHEMA,
    WORKFLOW_BLOCKED,
    DomainError,
    prompt_result,
    success_tool_result,
    text_resource_result,
    tool_result_from_exception,
)
from utils.provider_utils import init_langchain_provider_components
from utils.resume_utils import ensure_dir_exists, ensure_files_exist
from utils.workflow_utils import (
    WorkflowStore,
    add_interview_kit,
    add_outreach_draft,
    add_scheduling_request,
    cancel_workflow,
    create_workflow,
    get_workflow_status,
    resume_workflow,
    update_candidate_stage,
)

load_dotenv()

server = Server("hiring_orchestrator")

RESUME_DIR = os.environ.get("RESUME_DIR", "./assets")
WORKFLOW_STATE_DIR = os.environ.get("WORKFLOW_STATE_DIR", "./workflow_state")
embeddings, llm, provider_config = init_langchain_provider_components(temperature=0)
workflow_store = WorkflowStore(WORKFLOW_STATE_DIR)


class StartHiringWorkflow(BaseModel):
    job_description: Annotated[str, Field(description="Job description used to build the shortlist")]
    candidate_files: Annotated[List[str], Field(description="Resume PDF file names to shortlist")]
    role_title: Annotated[Optional[str], Field(description="Optional role title for recruiter artifacts")] = None
    recruiter_name: Annotated[Optional[str], Field(description="Optional recruiter name used in outreach")] = None
    organization_name: Annotated[Optional[str], Field(description="Optional organization name used in outreach")] = None
    top_k: Annotated[int, Field(description="Number of candidates to shortlist", ge=1, le=20)] = 3
    scoring_profile: Annotated[
        str,
        Field(description="Ranking profile: balanced, skills_first, experience_first, or entry_level"),
    ] = "balanced"


class WorkflowCandidateAction(BaseModel):
    workflow_id: Annotated[str, Field(description="Workflow identifier")]
    candidate_file: Annotated[str, Field(description="Candidate file from the workflow shortlist")]


class ScheduleInterview(BaseModel):
    workflow_id: Annotated[str, Field(description="Workflow identifier")]
    candidate_file: Annotated[str, Field(description="Candidate file from the workflow shortlist")]
    interviewer_emails: Annotated[List[str], Field(description="Interviewer email addresses")]
    time_slots: Annotated[List[str], Field(description="Proposed interview time slots in ISO-8601 text")]


class UpdateCandidateStage(BaseModel):
    workflow_id: Annotated[str, Field(description="Workflow identifier")]
    candidate_file: Annotated[str, Field(description="Candidate file from the workflow shortlist")]
    stage: Annotated[str, Field(description="New workflow stage for the candidate")]
    notes: Annotated[Optional[str], Field(description="Optional notes about the stage transition")] = None


class WorkflowControl(BaseModel):
    workflow_id: Annotated[str, Field(description="Workflow identifier")]
    reason: Annotated[Optional[str], Field(description="Optional reason for the control action")] = None


class GetWorkflowStatus(BaseModel):
    workflow_id: Annotated[str, Field(description="Workflow identifier")]


def _require_candidate_files(candidate_files):
    missing = ensure_files_exist(candidate_files, RESUME_DIR)
    if missing:
        raise DomainError(RESUME_NOT_FOUND, f"Resume files not found: {', '.join(missing)}")
    return candidate_files


def _trace(step_count=None):
    trace = {
        "provider": provider_config.provider,
        "llm_used": llm is not None,
        "retrieval_mode": "dense" if embeddings else "heuristic",
    }
    if step_count is not None:
        trace["step_count"] = step_count
    return trace


def _workflow_resource_list():
    resources = []
    for workflow in workflow_store.list_records():
        workflow_id = workflow.get("workflow_id") or workflow.get("record_id")
        if not workflow_id:
            continue
        resources.append(
            types.Resource(
                uri=f"workflow://{workflow_id}",
                name=f"Workflow {workflow_id}",
                description="Persisted workflow state",
                mimeType="application/json",
            )
        )
        resources.append(
            types.Resource(
                uri=f"workflow://{workflow_id}/timeline",
                name=f"Workflow Timeline {workflow_id}",
                description="Recent workflow events and transitions",
                mimeType="application/json",
            )
        )
        for candidate_file in workflow.get("candidate_state", {}).keys():
            candidate_id = os.path.splitext(os.path.basename(candidate_file))[0]
            resources.append(
                types.Resource(
                    uri=f"candidate://{candidate_id}/status",
                    name=f"Candidate Status {candidate_id}",
                    description="Workflow status for a shortlisted candidate",
                    mimeType="application/json",
                )
            )
            resources.append(
                types.Resource(
                    uri=f"interviewkit://{candidate_id}/{workflow_id}",
                    name=f"Interview Kit {candidate_id}/{workflow_id}",
                    description="Generated interview kit for the candidate in a workflow",
                    mimeType="application/json",
                )
            )
    return resources


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="start_hiring_workflow",
            description="Create a shortlist and initialize a recruiter workflow",
            inputSchema=StartHiringWorkflow.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="draft_candidate_outreach",
            description="Generate outreach copy for a shortlisted candidate",
            inputSchema=WorkflowCandidateAction.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="generate_interview_kit",
            description="Generate a recruiter and hiring-manager interview kit for a candidate",
            inputSchema=WorkflowCandidateAction.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="schedule_interview",
            description="Create a scheduling request payload for downstream calendar automation",
            inputSchema=ScheduleInterview.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="update_candidate_stage",
            description="Update the workflow stage for a candidate",
            inputSchema=UpdateCandidateStage.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="get_workflow_status",
            description="Fetch the current workflow state, steps, and recent events",
            inputSchema=GetWorkflowStatus.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="resume_workflow",
            description="Resume a blocked or paused workflow",
            inputSchema=WorkflowControl.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
        types.Tool(
            name="cancel_workflow",
            description="Cancel a workflow and preserve its audit trail",
            inputSchema=WorkflowControl.model_json_schema(),
            outputSchema=TOOL_RESPONSE_SCHEMA,
        ),
    ]


@server.list_resources()
async def list_resources():
    return _workflow_resource_list()


@server.read_resource()
async def read_resource(uri: AnyUrl):
    resource_uri = str(uri)

    if resource_uri.startswith("workflow://") and resource_uri.endswith("/timeline"):
        workflow_id = resource_uri[len("workflow://") : -len("/timeline")]
        workflow = workflow_store.load(workflow_id)
        payload = {"workflow_id": workflow_id, "events": workflow.get("events", [])}
        return text_resource_result(resource_uri, json.dumps(payload, indent=2))

    if resource_uri.startswith("workflow://"):
        workflow_id = resource_uri[len("workflow://") :]
        workflow = workflow_store.load(workflow_id)
        return text_resource_result(resource_uri, json.dumps(workflow, indent=2))

    if resource_uri.startswith("candidate://") and resource_uri.endswith("/status"):
        candidate_id = resource_uri[len("candidate://") : -len("/status")]
        for workflow in workflow_store.list_records():
            for candidate_file, state in workflow.get("candidate_state", {}).items():
                if os.path.splitext(os.path.basename(candidate_file))[0] == candidate_id:
                    payload = {
                        "candidate_id": candidate_id,
                        "candidate_file": candidate_file,
                        "workflow_id": workflow.get("workflow_id"),
                        "state": state,
                    }
                    return text_resource_result(resource_uri, json.dumps(payload, indent=2))
        raise DomainError(RESOURCE_NOT_FOUND, f"Candidate status resource not found: {resource_uri}")

    if resource_uri.startswith("interviewkit://"):
        _, path = resource_uri.split("://", 1)
        candidate_id, workflow_id = path.split("/", 1)
        workflow = workflow_store.load(workflow_id)
        for candidate_file, state in workflow.get("candidate_state", {}).items():
            if os.path.splitext(os.path.basename(candidate_file))[0] == candidate_id:
                payload = {
                    "candidate_id": candidate_id,
                    "workflow_id": workflow_id,
                    "interview_kit": state.get("interview_kit"),
                }
                return text_resource_result(resource_uri, json.dumps(payload, indent=2))
        raise DomainError(RESOURCE_NOT_FOUND, f"Interview kit resource not found: {resource_uri}")

    raise DomainError(RESOURCE_NOT_FOUND, f"Unknown resource: {resource_uri}")


@server.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="prepare_recruiter_outreach",
            description="Guide the model to prepare candidate outreach from an orchestrated workflow",
            arguments=[
                types.PromptArgument(name="workflow_id", description="Workflow identifier", required=True),
                types.PromptArgument(name="candidate_file", description="Candidate file from the shortlist", required=True),
            ],
        ),
        types.Prompt(
            name="prepare_hiring_manager_brief",
            description="Guide the model to prepare a hiring manager brief from workflow artifacts",
            arguments=[
                types.PromptArgument(name="workflow_id", description="Workflow identifier", required=True),
                types.PromptArgument(name="candidate_file", description="Candidate file from the shortlist", required=True),
            ],
        ),
        types.Prompt(
            name="prepare_interview_loop",
            description="Guide the model to create an interview loop based on the workflow artifacts",
            arguments=[
                types.PromptArgument(name="workflow_id", description="Workflow identifier", required=True),
                types.PromptArgument(name="candidate_file", description="Candidate file from the shortlist", required=True),
            ],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None):
    arguments = arguments or {}

    if name == "prepare_recruiter_outreach":
        text = (
            "Use the `draft_candidate_outreach` tool.\n"
            f"Workflow ID: {arguments.get('workflow_id', '')}\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            "After the tool runs, tailor the outreach for recruiter tone and call out the top evidence-backed reasons for contacting the candidate."
        )
        return prompt_result("Prepare recruiter outreach from workflow state", text)

    if name == "prepare_hiring_manager_brief":
        text = (
            "Use the `generate_interview_kit` tool.\n"
            f"Workflow ID: {arguments.get('workflow_id', '')}\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            "Summarize the candidate's strengths, risks, and the interview focus areas for the hiring manager."
        )
        return prompt_result("Prepare a hiring manager brief", text)

    if name == "prepare_interview_loop":
        text = (
            "Use the `generate_interview_kit` tool first, then `schedule_interview` if scheduling inputs are available.\n"
            f"Workflow ID: {arguments.get('workflow_id', '')}\n"
            f"Candidate file: {arguments.get('candidate_file', '')}\n"
            "Construct a clear interview loop recommendation grounded in the generated kit."
        )
        return prompt_result("Prepare an interview loop recommendation", text)

    raise DomainError(PROMPT_NOT_FOUND, f"Unknown prompt: {name}")


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "start_hiring_workflow":
            args = StartHiringWorkflow(**arguments)
            workflow = create_workflow(
                store=workflow_store,
                job_description=args.job_description,
                candidate_files=_require_candidate_files(args.candidate_files),
                resume_dir=RESUME_DIR,
                embeddings=embeddings,
                llm=llm,
                role_title=args.role_title,
                recruiter_name=args.recruiter_name,
                organization_name=args.organization_name,
                top_k=args.top_k,
                scoring_profile=args.scoring_profile,
            )
            return success_tool_result(
                data=workflow,
                run_id=workflow["workflow_id"],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Started hiring workflow and created shortlist.",
            )

        if name == "draft_candidate_outreach":
            args = WorkflowCandidateAction(**arguments)
            workflow, draft_text = add_outreach_draft(workflow_store, args.workflow_id, args.candidate_file, llm)
            payload = {
                "workflow_id": workflow["workflow_id"],
                "candidate_file": args.candidate_file,
                "outreach_draft": draft_text,
            }
            return success_tool_result(
                data=payload,
                run_id=workflow["workflow_id"],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Prepared recruiter outreach for the candidate.",
            )

        if name == "generate_interview_kit":
            args = WorkflowCandidateAction(**arguments)
            workflow, interview_kit = add_interview_kit(workflow_store, args.workflow_id, args.candidate_file, llm)
            payload = {
                "workflow_id": workflow["workflow_id"],
                "candidate_file": args.candidate_file,
                "interview_kit": interview_kit,
            }
            return success_tool_result(
                data=payload,
                run_id=workflow["workflow_id"],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Generated interview kit for the candidate.",
            )

        if name == "schedule_interview":
            args = ScheduleInterview(**arguments)
            workflow, scheduling_request = add_scheduling_request(
                workflow_store,
                args.workflow_id,
                args.candidate_file,
                args.interviewer_emails,
                args.time_slots,
            )
            warnings = [
                {
                    "code": DOWNSTREAM_CONNECTOR_ERROR,
                    "message": "Scheduling payload created, but no live calendar MCP is configured in this local implementation.",
                }
            ]
            payload = {
                "workflow_id": workflow["workflow_id"],
                "candidate_file": args.candidate_file,
                "scheduling_request": scheduling_request,
            }
            return success_tool_result(
                data=payload,
                run_id=workflow["workflow_id"],
                warnings=warnings,
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Created scheduling request payload for downstream calendar automation.",
                status="degraded",
            )

        if name == "update_candidate_stage":
            args = UpdateCandidateStage(**arguments)
            workflow = update_candidate_stage(
                workflow_store,
                args.workflow_id,
                args.candidate_file,
                args.stage,
                notes=args.notes,
            )
            payload = {
                "workflow_id": workflow["workflow_id"],
                "candidate_file": args.candidate_file,
                "stage": workflow["candidate_state"][args.candidate_file]["stage"],
            }
            return success_tool_result(
                data=payload,
                run_id=workflow["workflow_id"],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Updated candidate workflow stage.",
            )

        if name == "get_workflow_status":
            args = GetWorkflowStatus(**arguments)
            payload = get_workflow_status(workflow_store, args.workflow_id)
            return success_tool_result(
                data=payload,
                run_id=payload["workflow_id"],
                trace=_trace(step_count=len(payload.get("steps", []))),
                narrative="Fetched workflow status and recent events.",
            )

        if name == "resume_workflow":
            args = WorkflowControl(**arguments)
            workflow = resume_workflow(workflow_store, args.workflow_id)
            return success_tool_result(
                data=workflow,
                run_id=workflow["workflow_id"],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Resumed workflow execution.",
            )

        if name == "cancel_workflow":
            args = WorkflowControl(**arguments)
            workflow = cancel_workflow(workflow_store, args.workflow_id, reason=args.reason)
            return success_tool_result(
                data=workflow,
                run_id=workflow["workflow_id"],
                warnings=[{"code": WORKFLOW_BLOCKED, "message": "Workflow was intentionally cancelled."}],
                trace=_trace(step_count=len(workflow.get("steps", []))),
                narrative="Cancelled workflow while preserving audit trail.",
                status="degraded",
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
    ensure_dir_exists(WORKFLOW_STATE_DIR)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="hiring_orchestrator",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
