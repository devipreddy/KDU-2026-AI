from __future__ import annotations

import json


def _session_headers(client) -> dict[str, str]:
    session = client.post("/api/session")
    assert session.status_code == 200
    token = session.json()["client_secret"]
    return {"Authorization": f"Bearer {token}"}


def _parse_sse_events(payload: str) -> list[dict]:
    events: list[dict] = []
    for chunk in payload.split("\n\n"):
        line = chunk.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def _create_thread(client, headers: dict[str, str], text: str) -> tuple[str, list[dict]]:
    response = client.post(
        "/api/chatkit",
        headers=headers,
        json={
            "type": "threads.create",
            "params": {
                "input": {
                    "content": [{"type": "input_text", "text": text}],
                    "attachments": [],
                    "inference_options": {},
                }
            },
        },
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    thread_id = events[0]["thread"]["id"]
    return thread_id, events


def test_customer_cannot_open_another_users_thread(client_factory):
    with client_factory() as owner_client:
        owner_headers = _session_headers(owner_client)
        thread_id, _ = _create_thread(
            owner_client,
            owner_headers,
            "Plan me a trip to Tokyo",
        )

    with client_factory() as attacker_client:
        attacker_headers = _session_headers(attacker_client)
        response = attacker_client.get(
            f"/api/threads/{thread_id}/summary",
            headers=attacker_headers,
        )
        assert response.status_code == 404

        stream_attempt = attacker_client.post(
            "/api/chatkit",
            headers=attacker_headers,
            json={
                "type": "threads.add_user_message",
                "params": {
                    "thread_id": thread_id,
                    "input": {
                        "content": [{"type": "input_text", "text": "Hijack this thread"}],
                        "attachments": [],
                        "inference_options": {},
                    },
                },
            },
        )
        assert stream_attempt.status_code == 404


def test_booking_action_is_idempotent(client_factory):
    with client_factory() as client:
        headers = _session_headers(client)
        thread_id, _ = _create_thread(client, headers, "Book me something in Tokyo")

        action_payload = {
            "type": "threads.custom_action",
            "params": {
                "thread_id": thread_id,
                "action": {
                    "type": "travel.book_now.confirm",
                    "payload": {
                        "offer_id": "offer_tokyo_city",
                        "idempotency_key": "duplicate-safe-key",
                    },
                },
            },
        }

        first = client.post("/api/chatkit", headers=headers, json=action_payload)
        assert first.status_code == 200
        first_events = _parse_sse_events(first.text)
        assert any(event["type"] == "progress_update" for event in first_events)
        assert any(
            event["type"] == "thread.item.done"
            and event["item"]["type"] == "widget"
            for event in first_events
        )

        second = client.post("/api/chatkit", headers=headers, json=action_payload)
        assert second.status_code == 200
        second_events = _parse_sse_events(second.text)
        assert any(
            event["type"] == "notice"
            and "already processed" in event["message"]
            for event in second_events
        )


def test_handoff_request_pauses_ai_responses(client_factory):
    with client_factory() as client:
        headers = _session_headers(client)
        thread_id, _ = _create_thread(client, headers, "I need a Rome weekend")

        handoff = client.post(
            "/api/chatkit",
            headers=headers,
            json={
                "type": "threads.custom_action",
                "params": {
                    "thread_id": thread_id,
                    "action": {
                        "type": "support.request_handoff",
                        "payload": {"reason": "Needs a live specialist"},
                    },
                },
            },
        )
        assert handoff.status_code == 200

        summary = client.get(f"/api/threads/{thread_id}/summary", headers=headers)
        assert summary.status_code == 200
        assert summary.json()["conversation_mode"] == "handoff_requested"

        follow_up = client.post(
            "/api/chatkit",
            headers=headers,
            json={
                "type": "threads.add_user_message",
                "params": {
                    "thread_id": thread_id,
                    "input": {
                        "content": [{"type": "input_text", "text": "Are you still there?"}],
                        "attachments": [],
                        "inference_options": {},
                    },
                },
            },
        )
        assert follow_up.status_code == 200
        follow_up_events = _parse_sse_events(follow_up.text)
        assert any(
            event["type"] == "notice"
            and "paused" in event["message"].lower()
            for event in follow_up_events
        )
