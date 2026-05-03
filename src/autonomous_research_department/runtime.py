from __future__ import annotations

import os
from pathlib import Path

import appdirs


def project_root() -> Path:
    """Return the repository root for this package."""
    return Path(__file__).resolve().parents[2]


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def bootstrap_environment() -> Path:
    """Make CrewAI runtime paths workspace-local when the host path is locked down."""
    root = project_root()
    workspace_localapp = root / ".localapp"
    workspace_storage = root / ".crewai"

    desired_localapp = workspace_localapp
    localappdata = Path(os.environ.get("LOCALAPPDATA", desired_localapp))
    crewai_local_root = localappdata / "CrewAI"
    should_redirect = not _is_writable(crewai_local_root)

    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

    if should_redirect:
        workspace_localapp.mkdir(parents=True, exist_ok=True)
        os.environ["LOCALAPPDATA"] = str(workspace_localapp)
        os.environ.setdefault("APPDATA", str(workspace_localapp))

        def _workspace_win_folder(_: str) -> str:
            return str(workspace_localapp)

        appdirs._get_win_folder = _workspace_win_folder  # type: ignore[attr-defined]

    os.environ.setdefault("CREWAI_STORAGE_DIR", root.name)
    workspace_storage.mkdir(parents=True, exist_ok=True)
    return root
