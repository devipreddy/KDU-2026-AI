from collections.abc import Generator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
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
    return response.json()["access_token"]


def test_password_hash_verification() -> None:
    stored_hash = hash_password("correct horse battery staple", salt=b"test-salt")

    assert verify_password("correct horse battery staple", stored_hash)
    assert not verify_password("wrong password", stored_hash)


def test_access_token_round_trip() -> None:
    token = create_access_token(
        subject="20000000-0000-4000-8000-000000000003",
        roles=["admin"],
        expires_delta=timedelta(minutes=5),
    )

    payload = decode_access_token(token)

    assert payload["sub"] == "20000000-0000-4000-8000-000000000003"
    assert payload["roles"] == ["admin"]


def test_login_returns_jwt_and_current_user(client: TestClient) -> None:
    token = login(client, "doctor@example.com")

    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "doctor@example.com"
    assert [role["name"] for role in body["roles"]] == ["doctor"]


def test_invalid_login_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.com", "password": "wrong"},
    )

    assert response.status_code == 401


def test_admin_role_can_access_admin_rbac_endpoint(client: TestClient) -> None:
    token = login(client, "admin@example.com")

    response = client.get("/api/v1/auth/rbac/admin", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["allowed"] is True


def test_non_admin_role_cannot_access_admin_rbac_endpoint(client: TestClient) -> None:
    token = login(client, "doctor@example.com")

    response = client.get("/api/v1/auth/rbac/admin", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403
