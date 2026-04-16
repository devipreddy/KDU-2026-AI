"""
Workflow persistence and artifact generation for the hiring orchestrator MCP.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from utils.resume_utils import ensure_dir_exists
from utils.shortlisting_utils import rank_candidates


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class WorkflowStore:
    """Simple JSON-backed workflow store for local orchestration runs."""

    def __init__(self, workflow_dir):
        self.workflow_dir = os.path.abspath(workflow_dir)
        ensure_dir_exists(self.workflow_dir)

    def _workflow_path(self, workflow_id):
        return os.path.join(self.workflow_dir, f"{workflow_id}.json")

    def save(self, workflow):
        workflow["updated_at"] = _utc_now()
        with open(self._workflow_path(workflow["workflow_id"]), "w", encoding="utf-8") as handle:
            json.dump(workflow, handle, indent=2)
        return workflow

    def load(self, workflow_id):
        workflow_path = self._workflow_path(workflow_id)
        if not os.path.exists(workflow_path):
            raise ValueError(f"Workflow not found: {workflow_id}")

        with open(workflow_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_records(self):
        records = []
        for file_name in sorted(os.listdir(self.workflow_dir)):
            if not file_name.lower().endswith(".json"):
                continue
            workflow_id = os.path.splitext(file_name)[0]
            try:
                records.append(self.load(workflow_id))
            except Exception:
                continue
        return records


def _generate_outreach_text(candidate_summary, role_title, recruiter_name, organization_name, llm):
    if llm:
        prompt = f"""
        You are a recruiter writing a concise, warm outreach email.

        Candidate summary:
        {json.dumps(candidate_summary, indent=2)}

        Role title: {role_title or 'the open role'}
        Recruiter name: {recruiter_name or 'Recruiter'}
        Organization: {organization_name or 'our team'}

        Write:
        1. Subject line
        2. Email body
        3. Three personalization hooks grounded in the candidate evidence
        """
        return llm.invoke(prompt).content

    candidate_name = candidate_summary["candidate_name"]
    matched_keywords = ", ".join(candidate_summary["matched_keywords"][:5]) or "your background"
    role_fragment = role_title or "a relevant opportunity"
    sender = recruiter_name or "Recruiter"
    org = organization_name or "our team"

    return (
        f"Subject: {role_fragment} at {org}\n\n"
        f"Hi {candidate_name},\n\n"
        f"I came across your profile and was impressed by your experience with {matched_keywords}. "
        f"We are hiring for {role_fragment}, and your background appears strongly aligned.\n\n"
        f"If you're open to a quick conversation, I'd be glad to share more details.\n\n"
        f"Best,\n{sender}"
    )


def _generate_interview_kit_text(candidate_summary, job_description, llm):
    if llm:
        prompt = f"""
        Create an interview kit for a recruiter and hiring manager.

        Candidate summary:
        {json.dumps(candidate_summary, indent=2)}

        Job description:
        {job_description}

        Include:
        1. Candidate strengths
        2. Candidate risks or gaps
        3. Five interview questions
        4. Scorecard dimensions
        5. Decision guidance
        """
        return llm.invoke(prompt).content

    strengths = ", ".join(candidate_summary["matched_keywords"][:5]) or "general role alignment"
    return (
        "Interview Kit\n\n"
        f"Strengths: {strengths}\n"
        "Risks: Validate the depth of the candidate's project ownership and role-specific impact.\n"
        "Questions:\n"
        "1. Walk us through your most relevant project for this role.\n"
        "2. Which trade-offs did you make in that project and why?\n"
        "3. How have you handled ambiguity or changing requirements?\n"
        "4. Which skill in the job description are you strongest at?\n"
        "5. Which area would you want to ramp up on in the first 30 days?\n"
        "Scorecard: technical fit, communication, ownership, learning velocity, risk profile.\n"
        "Decision guidance: proceed if the candidate can demonstrate strong evidence for the matched skills."
    )


def create_workflow(
    store,
    job_description,
    candidate_files,
    resume_dir,
    embeddings,
    llm,
    role_title=None,
    recruiter_name=None,
    organization_name=None,
    top_k=3,
    scoring_profile="balanced",
):
    ranking = rank_candidates(
        job_description=job_description,
        candidate_files=candidate_files,
        resume_dir=resume_dir,
        embeddings=embeddings,
        llm=llm,
        top_k=top_k,
        scoring_profile=scoring_profile,
    )

    workflow_id = str(uuid.uuid4())
    shortlist = ranking["ranked_candidates"]
    workflow = {
        "workflow_id": workflow_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "role_title": role_title,
        "recruiter_name": recruiter_name,
        "organization_name": organization_name,
        "job_description": job_description,
        "scoring_profile": scoring_profile,
        "status": "shortlisted",
        "shortlist": shortlist,
        "candidate_state": {
            candidate["candidate_file"]: {
                "stage": "shortlisted",
                "outreach_draft": None,
                "interview_kit": None,
                "scheduling_request": None,
            }
            for candidate in shortlist
        },
        "steps": [
            {"name": "shortlist_candidates", "status": "completed"},
            {"name": "draft_candidate_outreach", "status": "pending"},
            {"name": "generate_interview_kit", "status": "pending"},
            {"name": "schedule_interview", "status": "pending"},
        ],
        "events": [
            {
                "event_type": "workflow_created",
                "timestamp": _utc_now(),
                "details": {
                    "top_candidate": shortlist[0]["candidate_file"] if shortlist else None,
                    "candidate_count": len(shortlist),
                },
            }
        ],
    }
    return store.save(workflow)


def get_candidate_summary(workflow, candidate_file):
    for candidate in workflow.get("shortlist", []):
        if candidate["candidate_file"] == candidate_file:
            return candidate

    raise ValueError(f"Candidate not part of workflow shortlist: {candidate_file}")


def add_outreach_draft(store, workflow_id, candidate_file, llm):
    workflow = store.load(workflow_id)
    candidate_summary = get_candidate_summary(workflow, candidate_file)
    draft_text = _generate_outreach_text(
        candidate_summary=candidate_summary,
        role_title=workflow.get("role_title"),
        recruiter_name=workflow.get("recruiter_name"),
        organization_name=workflow.get("organization_name"),
        llm=llm,
    )

    workflow["candidate_state"][candidate_file]["outreach_draft"] = draft_text
    workflow["candidate_state"][candidate_file]["stage"] = "outreach_prepared"
    workflow["events"].append(
        {
            "event_type": "outreach_drafted",
            "timestamp": _utc_now(),
            "details": {"candidate_file": candidate_file},
        }
    )
    workflow["steps"][1]["status"] = "completed"
    return store.save(workflow), draft_text


def add_interview_kit(store, workflow_id, candidate_file, llm):
    workflow = store.load(workflow_id)
    candidate_summary = get_candidate_summary(workflow, candidate_file)
    interview_kit = _generate_interview_kit_text(
        candidate_summary=candidate_summary,
        job_description=workflow["job_description"],
        llm=llm,
    )

    workflow["candidate_state"][candidate_file]["interview_kit"] = interview_kit
    workflow["candidate_state"][candidate_file]["stage"] = "interview_kit_ready"
    workflow["events"].append(
        {
            "event_type": "interview_kit_generated",
            "timestamp": _utc_now(),
            "details": {"candidate_file": candidate_file},
        }
    )
    workflow["steps"][2]["status"] = "completed"
    return store.save(workflow), interview_kit


def add_scheduling_request(store, workflow_id, candidate_file, interviewer_emails, time_slots):
    workflow = store.load(workflow_id)
    scheduling_request = {
        "candidate_file": candidate_file,
        "interviewer_emails": interviewer_emails,
        "time_slots": time_slots,
        "status": "pending_external_calendar_confirmation",
    }

    workflow["candidate_state"][candidate_file]["scheduling_request"] = scheduling_request
    workflow["candidate_state"][candidate_file]["stage"] = "scheduling_requested"
    workflow["events"].append(
        {
            "event_type": "scheduling_requested",
            "timestamp": _utc_now(),
            "details": {"candidate_file": candidate_file, "interviewer_count": len(interviewer_emails)},
        }
    )
    workflow["steps"][3]["status"] = "completed"
    workflow["status"] = "in_progress"
    return store.save(workflow), scheduling_request


def update_candidate_stage(store, workflow_id, candidate_file, stage, notes=None):
    workflow = store.load(workflow_id)
    if candidate_file not in workflow["candidate_state"]:
        raise ValueError(f"Candidate not part of workflow shortlist: {candidate_file}")

    workflow["candidate_state"][candidate_file]["stage"] = stage
    workflow["events"].append(
        {
            "event_type": "candidate_stage_updated",
            "timestamp": _utc_now(),
            "details": {
                "candidate_file": candidate_file,
                "stage": stage,
                "notes": notes,
            },
        }
    )
    return store.save(workflow)


def resume_workflow(store, workflow_id):
    workflow = store.load(workflow_id)
    workflow["status"] = "in_progress"
    workflow["events"].append(
        {
            "event_type": "workflow_resumed",
            "timestamp": _utc_now(),
            "details": {"workflow_id": workflow_id},
        }
    )
    return store.save(workflow)


def cancel_workflow(store, workflow_id, reason=None):
    workflow = store.load(workflow_id)
    workflow["status"] = "cancelled"
    workflow["events"].append(
        {
            "event_type": "workflow_cancelled",
            "timestamp": _utc_now(),
            "details": {"workflow_id": workflow_id, "reason": reason},
        }
    )
    return store.save(workflow)


def get_workflow_status(store, workflow_id):
    workflow = store.load(workflow_id)
    return {
        "workflow_id": workflow["workflow_id"],
        "status": workflow["status"],
        "role_title": workflow.get("role_title"),
        "shortlist_size": len(workflow.get("shortlist", [])),
        "steps": workflow["steps"],
        "candidate_state": workflow["candidate_state"],
        "events": workflow["events"][-10:],
    }
