"""
Centralized filesystem path resolution for the CAD project.

All runtime file access should be resolved relative to the project root
rather than the process current working directory. This keeps the Streamlit
dashboard, CLI demo, benchmark, and test-suite predictable regardless of
where they are launched from.
"""

import os

# ``src/paths.py`` -> project root is one directory up.
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")

# Environment variable that can override the live database location. This is
# used by the test-suite (and any deployment) to point the app at an isolated
# database without touching the developer's real records.
DB_PATH_ENV_VAR: str = "CAD_DB_PATH"

# Primary live SQLite database used by the Streamlit dashboard and memory engine.
DEFAULT_DB_PATH: str = os.path.join(OUTPUTS_DIR, "remediation_memory.db")

# Dedicated database used only by the offline benchmark so it can never
# corrupt the live application's remediation memory.
BENCHMARK_DB_PATH: str = os.path.join(OUTPUTS_DIR, "benchmark_memory.db")


def get_default_db_path() -> str:
    """
    Returns the live database path, honoring the ``CAD_DB_PATH`` override.

    Read at call time (not import time) so tests and tooling can redirect the
    database after the module has been imported.
    """
    override = os.environ.get(DB_PATH_ENV_VAR)
    return resolve_path(override) if override else DEFAULT_DB_PATH


def resolve_path(path: str) -> str:
    """
    Resolves ``path`` to an absolute path.

    Absolute paths are returned unchanged. Relative paths are interpreted
    relative to the project root, not the current working directory.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def ensure_parent_dir(path: str) -> None:
    """Creates the parent directory of ``path`` if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
