from backend.app.rag import (
    HashingEmbeddingProvider,
    chunk_text,
    cosine_similarity,
    tokenize,
)


def test_tokenize_chinese_and_english():
    tokens = tokenize("Redis 缓存 RAG")
    assert "redis" in tokens
    assert "缓" in tokens
    assert "缓存" in tokens


def test_chunk_text_keeps_reasonable_size():
    text = "第一段内容。" * 100 + "\n\n" + "第二段内容。" * 100
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)


def test_hashing_embedding_is_deterministic():
    provider = HashingEmbeddingProvider(dim=128)
    first = provider.embed(["员工年假制度"])
    second = provider.embed(["员工年假制度"])
    assert len(first[0]) == 128
    assert first == second


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0