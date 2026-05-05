from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import main


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "test-chatkit.sqlite3"


@pytest.fixture
def client_factory(temp_db: Path):
    main.settings.use_mock_model = True
    main.chat_server.use_mock_model = True
    main.store.db_path = temp_db
    if temp_db.exists():
        temp_db.unlink()

    def _make_client() -> TestClient:
        return TestClient(main.app)

    return _make_client
