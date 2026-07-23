from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime


# ──────────────────────────────
# Thread Schemas
# ──────────────────────────────
class ThreadCreate(BaseModel):
    title: Optional[str] = None


class ThreadResponse(BaseModel):
    id: int
    uuid: str
    user_id: int
    title: Optional[str] = None
    last_cutoff_date: Optional[str] = None
    last_firms: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ThreadList(BaseModel):
    threads: List[ThreadResponse]

    model_config = {"from_attributes": True}


# ──────────────────────────────
# Chat Schemas
# ──────────────────────────────
class ChatCreate(BaseModel):
    message: str
    role: str = "user"
    # TemporalRAG-specific parameters
    cutoff_date: Optional[str] = "2026-06"     # YYYY-MM format
    firms: Optional[List[str]] = None           # list of tickers, None = all

class ChatResponse(BaseModel):
    id: int
    user_id: int
    thread_id: int
    role: str
    message: str
    meta_json: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatList(BaseModel):
    chats: List[ChatResponse]

    model_config = {"from_attributes": True}


class ThreadWithChats(ThreadResponse):
    chats: List[ChatResponse] = []

    model_config = {"from_attributes": True}


# ──────────────────────────────
# RAG Response Schema (rich response from TemporalRAG)
# ──────────────────────────────
class PathItem(BaseModel):
    path: List[str]          # ["NVDA", "CO_OCCURRED_WITH", "TSM"]
    weight: Optional[float] = None
    snippets: Optional[List[str]] = []


class RagChatResponse(BaseModel):
    """Full enriched response returned from the /chats/chat/{thread_uuid} endpoint."""
    thread_uuid: str
    user_message_id: int
    assistant_message_id: int
    answer: str
    mode: str                          # "pathrag" | "vector_fallback"
    cutoff_date: str
    cutoff_ts: Optional[int] = None
    firms: Optional[List[str]] = None
    paths: Optional[List[PathItem]] = []
    entities_found: Optional[List[str]] = []
    reasoning_process: Optional[str] = None   # CoT <think> content
    created_at: datetime
