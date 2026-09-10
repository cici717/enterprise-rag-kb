import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.set_page_config(
    page_title="企业知识库 RAG",
    page_icon="📚",
    layout="wide",
)


def api_get(path: str):
    response = httpx.get(f"{API_BASE}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = httpx.post(f"{API_BASE}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def upload_document(filename: str, content: bytes, content_type: str):
    response = httpx.post(
        f"{API_BASE}/documents/upload",
        files={"file": (filename, content, content_type)},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def stream_events(query: str, conversation_id: int | None):
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "top_k": st.session_state.top_k,
    }
    with httpx.stream(
        "POST",
        f"{API_BASE}/chat/stream",
        json=payload,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                yield json.loads(line[6:])


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"引用来源（{len(citations)} 条）"):
        for index, item in enumerate(citations, start=1):
            page = f" 第{item['page']}页" if item.get("page") else ""
            st.markdown(
                f"**[{index}] {item['filename']}{page}** · 相关度 {item['score']:.3f}"
            )
            st.caption(item.get("snippet", ""))


st.session_state.setdefault("messages", [])
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("top_k", 5)

with st.sidebar:
    st.header("知识库管理")
    try:
        health = api_get("/health")
        st.success("后端已连接")
        st.caption(
            f"LLM: {health['llm_mode']} | Embedding: {health['embedding_mode']}"
        )
        col_a, col_b = st.columns(2)
        col_a.metric("文档", health["document_count"])
        col_b.metric("Chunk", health["chunk_count"])
    except Exception as exc:
        st.error(f"后端未连接：{exc}")
        st.stop()

    st.divider()
    uploaded = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt", "md"],
        help="支持 PDF、Word、TXT、Markdown，单文件不超过 20MB",
    )
    if uploaded and st.button("上传并入库", type="primary"):
        with st.spinner("正在解析、切分和建立索引..."):
            try:
                document = upload_document(
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "",
                )
                st.success(
                    f"已导入 {document['filename']}，生成 {document['chunk_count']} 个片段"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"上传失败：{exc}")

    st.divider()
    st.slider("检索片段数 Top-K", min_value=1, max_value=10, key="top_k")
    if st.button("新建会话"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    st.divider()
    st.subheader("文档列表")
    try:
        documents = api_get("/documents")
    except Exception as exc:
        documents = []
        st.error(f"读取文档失败：{exc}")

    if not documents:
        st.caption("暂无文档，请先上传。")
    for document in documents:
        with st.container(border=True):
            st.markdown(f"**{document['filename']}**")
            st.caption(
                f"{document['chunk_count']} 个片段 · {document['status']}"
            )
            if st.button("删除", key=f"delete-{document['id']}"):
                try:
                    httpx.delete(
                        f"{API_BASE}/documents/{document['id']}",
                        timeout=30,
                    ).raise_for_status()
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败：{exc}")

st.title("企业知识库智能问答")
st.caption("基于 RAG 的文档问答演示：回答会附带检索到的引用来源。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))

prompt = st.chat_input("例如：迟到超过三十分钟怎么处理？")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        citations: list[dict] = []
        placeholder = st.empty()
        try:
            for event in stream_events(prompt, st.session_state.conversation_id):
                event_type = event.get("type")
                if event_type == "conversation":
                    st.session_state.conversation_id = event["data"]["id"]
                elif event_type == "citations":
                    citations = event.get("data", [])
                elif event_type == "token":
                    answer += event.get("data", "")
                    placeholder.markdown(answer + "▌")
                elif event_type == "error":
                    raise RuntimeError(event.get("data", {}).get("message", "未知错误"))
            placeholder.markdown(answer)
            render_citations(citations)
        except Exception as exc:
            placeholder.empty()
            st.error(f"问答失败：{exc}")
            answer = answer or "本次回答失败，请检查后端服务。"
            citations = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )