import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import rag
from .config import get_settings
from .db import (
    Chunk,
    Conversation,
    Document,
    Feedback,
    Message,
    get_db,
    init_db,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: int | None = None
    top_k: int | None = None


class FeedbackRequest(BaseModel):
    message_id: int | None = None
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


class DocumentOut(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int
    status: str
    chunk_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CitationOut(BaseModel):
    document_id: int
    filename: str
    page: int | None
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[CitationOut]


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[dict]
    created_at: datetime


class ConversationDetailOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageOut]


class FeedbackOut(BaseModel):
    id: int
    message_id: int | None
    rating: int
    comment: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
    }


@app.get(f"{settings.api_prefix}/health")
def health(db: Session = Depends(get_db)):
    document_count = db.scalar(select(func.count()).select_from(Document)) or 0
    chunk_count = db.scalar(select(func.count()).select_from(Chunk)) or 0
    embedding_mode = settings.embedding_provider
    if embedding_mode == "openai" and not (
        settings.embedding_api_key or settings.llm_api_key
    ):
        embedding_mode = "hashing"
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_mode": "openai" if settings.llm_api_key else "mock",
        "embedding_mode": embedding_mode,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }


@app.post(f"{settings.api_prefix}/documents/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件不能超过 20MB")

    try:
        document = rag.ingest_document(
            db=db,
            filename=file.filename or "document.txt",
            content_type=file.content_type or "",
            data=data,
            settings=settings,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"文档处理失败: {exc}") from exc
    return document


@app.get(f"{settings.api_prefix}/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(
        select(Document).order_by(Document.created_at.desc())
    ).all()


@app.delete(f"{settings.api_prefix}/documents/{{document_id}}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.delete(document)
    db.commit()
    return {"ok": True}


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    answer, citations, conversation_id = rag.chat(
        db=db,
        query=request.query,
        conversation_id=request.conversation_id,
        top_k=request.top_k,
        settings=settings,
    )
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        citations=[CitationOut(**item) for item in citations],
    )


@app.post(f"{settings.api_prefix}/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        rag.stream_chat(
            query=request.query,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            settings=settings,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    f"{settings.api_prefix}/conversations",
    response_model=list[ConversationOut],
)
def list_conversations(db: Session = Depends(get_db)):
    return db.scalars(
        select(Conversation).order_by(Conversation.created_at.desc())
    ).all()


@app.get(
    f"{settings.api_prefix}/conversations/{{conversation_id}}",
    response_model=ConversationDetailOut,
)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    ).all()
    return ConversationDetailOut(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            MessageOut(
                id=item.id,
                role=item.role,
                content=item.content,
                citations=json.loads(item.citations or "[]"),
                created_at=item.created_at,
            )
            for item in messages
        ],
    )


@app.post(f"{settings.api_prefix}/feedback", response_model=FeedbackOut)
def create_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    feedback = Feedback(
        message_id=request.message_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback