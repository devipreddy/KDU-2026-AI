"""
Small JSON-backed stores for persisted MCP resource state.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from utils.resume_utils import ensure_dir_exists


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class JsonRecordStore:
    """Store JSON records as one file per record."""

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        ensure_dir_exists(self.directory)

    def _record_path(self, record_id):
        return os.path.join(self.directory, f"{record_id}.json")

    def create(self, payload, record_id=None):
        record_id = record_id or str(uuid.uuid4())
        payload = dict(payload)
        payload.setdefault("record_id", record_id)
        payload.setdefault("created_at", utc_now())
        payload["updated_at"] = utc_now()
        with open(self._record_path(record_id), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def save(self, record_id, payload):
        payload = dict(payload)
        payload.setdefault("record_id", record_id)
        payload["updated_at"] = utc_now()
        with open(self._record_path(record_id), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def load(self, record_id):
        record_path = self._record_path(record_id)
        if not os.path.exists(record_path):
            raise ValueError(f"Record not found: {record_id}")

        with open(record_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_records(self):
        records = []
        for file_name in sorted(os.listdir(self.directory)):
            if not file_name.lower().endswith(".json"):
                continue

            record_id = os.path.splitext(file_name)[0]
            try:
                payload = self.load(record_id)
            except Exception:
                continue
            records.append(payload)
        return records
