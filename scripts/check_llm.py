import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag import OpenAICompatibleLLM, get_embedding_provider  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    print("LLM_BASE_URL:", settings.llm_base_url)
    print("LLM_MODEL:", settings.llm_model)
    print("LLM_API_KEY:", "configured" if settings.llm_api_key else "missing")
    print("EMBEDDING_PROVIDER:", settings.embedding_provider)

    if not settings.llm_api_key:
        print("")
        print("LLM_API_KEY is empty. The app will run in offline MockLLM mode.")
        print("Edit .env and set LLM_API_KEY, then rerun this script.")
        return

    try:
        llm = OpenAICompatibleLLM(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ]
        answer = llm.complete(messages)
        print("LLM_RESPONSE:", answer[:200])
    except Exception as exc:
        print("LLM_ERROR:", repr(exc))
        return

    try:
        provider = get_embedding_provider(settings)
        vectors = provider.embed(["测试文本"])
        print("EMBEDDING_PROVIDER_USED:", provider.name)
        print("EMBEDDING_DIM:", len(vectors[0]))
    except Exception as exc:
        print("EMBEDDING_ERROR:", repr(exc))


if __name__ == "__main__":
    main()