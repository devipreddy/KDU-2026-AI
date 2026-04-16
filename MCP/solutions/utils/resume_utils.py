"""
Common resume utilities for the MCP tools
"""

import logging
import os
from typing import Iterable

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def resolve_resume_path(file_path, base_dir="./assets"):
    """Resolve a resume path against the configured resume directory.

    Relative paths are confined to ``base_dir`` while absolute paths are
    normalized and used as provided.
    """
    if not file_path:
        raise ValueError("A resume file path is required.")

    if os.path.isabs(file_path):
        return os.path.abspath(file_path)

    base_dir_abs = os.path.abspath(base_dir)
    resolved_path = os.path.abspath(os.path.join(base_dir_abs, file_path))

    try:
        common_root = os.path.commonpath([base_dir_abs, resolved_path])
    except ValueError as exc:
        raise ValueError(f"Invalid resume path: {file_path}") from exc

    if common_root != base_dir_abs:
        raise ValueError("Resume path escapes the configured resume directory.")

    return resolved_path

def read_resume(file_path, base_dir="./assets"):
    """Extract text from a resume PDF file.
    
    Args:
        file_path: Path to the resume PDF file
        base_dir: Base directory for relative paths
        
    Returns:
        str: The extracted text, or None if there was an error
    """
    try:
        resolved_path = resolve_resume_path(file_path, base_dir)

        doc = fitz.open(resolved_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error reading resume {file_path}: {e}")
        return None

def list_resume_files(base_dir="./assets"):
    """List all PDF resume files in the configured directory."""
    if not os.path.exists(base_dir):
        return []

    return sorted(
        file_name
        for file_name in os.listdir(base_dir)
        if file_name.lower().endswith(".pdf")
    )

def read_text_artifact(file_path, base_dir=None):
    """Read a plain-text artifact from disk."""
    resolved_path = (
        resolve_resume_path(file_path, base_dir)
        if base_dir is not None and not os.path.isabs(file_path)
        else os.path.abspath(file_path)
    )

    with open(resolved_path, "r", encoding="utf-8") as handle:
        return handle.read()

def ensure_files_exist(file_paths: Iterable[str], base_dir="./assets"):
    """Return a list of missing file paths after resolution."""
    missing = []

    for file_path in file_paths:
        try:
            resolved_path = resolve_resume_path(file_path, base_dir)
        except ValueError:
            missing.append(file_path)
            continue

        if not os.path.exists(resolved_path):
            missing.append(file_path)

    return missing

def ensure_dir_exists(directory):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
