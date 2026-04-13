import os
from pathlib import Path


def _parse_env_line(line: str):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None, None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]

    return key, value


def load_project_env():
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    candidate_paths = [
        project_root / ".env",
        backend_dir / ".env",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue

        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                key, value = _parse_env_line(raw_line)
                if key:
                    os.environ.setdefault(key, value)
        except OSError:
            continue
