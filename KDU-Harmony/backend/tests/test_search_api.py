from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import DEMO_PASSWORD, seed_auth_data
from app.db.session import get_db
from app.main import create_app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with testing_session_local() as db:
        seed_auth_data(db)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session_local() as db:
            yield db

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def login(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def test_search_endpoint_returns_pipeline_shape_without_indexed_chunks(
    client: TestClient,
) -> None:
    token = login(client, "doctor@example.com")

    response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "query": "patients with cardiac issues treated in Q1 2025",
            "limit": 3,
            "candidate_limit": 5,
            "rerank_top_n": 2,
            "include_llm_answer": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hit_count"] == 0
    assert body["answer"]["status"] == "no_evidence"
    assert body["pipeline"]["bm25"]["hit_count"] == 0
    assert body["pipeline"]["dense"]["hit_count"] == 0
    assert body["pipeline"]["reranker"]["reranked_count"] == 0
    assert body["pipeline"]["llm"]["status"] == "no_evidence"


def test_stream_search_endpoint_sends_retrieval_and_answer_events(
    client: TestClient,
) -> None:
    token = login(client, "doctor@example.com")

    with client.stream(
        "POST",
        "/api/v1/search/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "query": "patients with cardiac issues treated in Q1 2025",
            "limit": 3,
            "candidate_limit": 5,
            "rerank_top_n": 2,
            "include_llm_answer": True,
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: retrieval" in body
    assert "event: answer_done" in body
    assert '"status":"no_evidence"' in body


def test_audit_events_endpoint_returns_rendered_search_audit(
    client: TestClient,
) -> None:
    doctor_token = login(client, "doctor@example.com")
    client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {doctor_token}"},
        json={"query": "patients with malaria", "include_llm_answer": False},
    )
    admin_token = login(client, "admin@example.com")

    response = client.get(
        "/api/v1/audit/events?limit=20",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
    events = response.json()
    assert any(
        event["action"] == "query_run" and event["query_text"] == "patients with malaria"
        for event in events
    )
