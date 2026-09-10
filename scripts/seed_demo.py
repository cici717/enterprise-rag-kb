import os
from pathlib import Path

import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "samples"
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def main() -> None:
    files = [
        path
        for path in SAMPLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    ]
    if not files:
        print("samples 目录中没有可导入的文档")
        return

    for path in files:
        with path.open("rb") as stream:
            response = httpx.post(
                f"{API_BASE}/documents/upload",
                files={"file": (path.name, stream)},
                timeout=180,
            )
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] {path.name} -> {data['chunk_count']} chunks")
        else:
            print(f"[FAIL] {path.name} -> {response.status_code} {response.text}")


if __name__ == "__main__":
    main()