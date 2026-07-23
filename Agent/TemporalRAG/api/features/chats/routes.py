"""
chats/routes.py
===============
TemporalRAG 채팅 API 엔드포인트.

PathRAG의 Thread/Chat CRUD 구조를 그대로 계승하되,
메시지 처리 로직을 TemporalRAGAgent.reason()으로 교체합니다.
인증 없이 동작 (기본 user_id=1 사용).
"""

import json
import datetime
import uuid as uuid_pkg
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db, Chat, User, Thread
from api.features.rag_manager import get_rag_agent
from .schemas import (
    ChatCreate, ChatResponse, ChatList,
    ThreadCreate, ThreadResponse, ThreadList, ThreadWithChats,
    RagChatResponse, PathItem,
)

logger = logging.getLogger("temporalrag.chats")

router = APIRouter(
    prefix="/chats",
    tags=["Chats"],
)

DEFAULT_USER_ID = 1


def get_default_user(db: Session) -> User:
    """인증 없이 기본 사용자(id=1)를 반환합니다."""
    user = db.query(User).filter(User.id == DEFAULT_USER_ID).first()
    if user is None:
        # 기본 사용자가 없으면 생성
        user = User(
            id=DEFAULT_USER_ID,
            username="admin",
            email="admin@temporalrag.local",
            hashed_password="",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


# ──────────────────────────────
# Thread CRUD
# ──────────────────────────────

@router.get("/threads", response_model=ThreadList)
async def get_threads(
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    threads = (
        db.query(Thread)
        .filter(Thread.user_id == current_user.id, Thread.is_deleted == False)
        .order_by(Thread.updated_at.desc())
        .all()
    )
    return {"threads": threads}


@router.post("/threads", response_model=ThreadResponse)
async def create_thread(
    thread: ThreadCreate = None,
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    db_thread = Thread(
        uuid=str(uuid_pkg.uuid4()),
        user_id=current_user.id,
        title=thread.title if thread and thread.title else "New Chat",
    )
    db.add(db_thread)
    db.commit()
    db.refresh(db_thread)
    return db_thread


@router.get("/threads/{thread_uuid}", response_model=ThreadWithChats)
async def get_thread(
    thread_uuid: str,
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
        Thread.is_deleted == False,
    ).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    chats = db.query(Chat).filter(Chat.thread_id == thread.id).order_by(Chat.created_at.asc()).all()
    result = ThreadWithChats.model_validate(thread)
    result.chats = chats
    return result


@router.delete("/threads/{thread_uuid}")
async def delete_thread(
    thread_uuid: str,
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
    ).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    thread.is_deleted = True
    db.commit()
    return {"success": True, "message": "Thread deleted successfully"}


@router.put("/threads/{thread_uuid}", response_model=ThreadResponse)
async def update_thread(
    thread_uuid: str,
    thread_data: ThreadCreate,
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
        Thread.is_deleted == False,
    ).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread_data.title is not None:
        thread.title = thread_data.title
    db.commit()
    db.refresh(thread)
    return thread


@router.get("/recent", response_model=ThreadList)
async def get_recent_threads(
    db: Session = Depends(get_db),
):
    current_user = get_default_user(db)
    threads = (
        db.query(Thread)
        .filter(Thread.user_id == current_user.id, Thread.is_deleted == False)
        .order_by(Thread.updated_at.desc())
        .limit(5)
        .all()
    )
    return {"threads": threads}


# ──────────────────────────────
# Core Chat Endpoint (TemporalRAG)
# ──────────────────────────────

@router.post("/chat/{thread_uuid}", response_model=RagChatResponse)
async def create_chat(
    thread_uuid: str,
    chat: ChatCreate,
    db: Session = Depends(get_db),
):
    """
    사용자 메시지를 받아 TemporalRAGAgent로 추론하고 결과를 반환합니다.

    TemporalRAG-specific payload fields:
      - cutoff_date: "YYYY-MM" 형식의 시간 컷오프 (기본: "2025-01")
      - firms: 분석할 기업 ticker 목록 (None이면 전체)
    """
    current_user = get_default_user(db)

    # ── 1. Thread 확인 ──
    thread = db.query(Thread).filter(
        Thread.uuid == thread_uuid,
        Thread.user_id == current_user.id,
        Thread.is_deleted == False,
    ).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # ── 2. 사용자 메시지 DB 저장 ──
    user_msg = Chat(
        user_id=current_user.id,
        thread_id=thread.id,
        role="user",
        message=chat.message,
    )
    db.add(user_msg)
    db.flush()  # get user_msg.id

    # ── 3. TemporalRAGAgent 호출 ──
    rag = get_rag_agent()
    cutoff_date = chat.cutoff_date or "2026-06"
    firms = chat.firms  # None means all firms

    if rag is None:
        # RAG 에이전트 미초기화 시 graceful 에러 메시지
        answer = (
            "TemporalRAG engine is not available. "
            "Please check Neo4j and vLLM server connections in your .env configuration."
        )
        mode = "error"
        paths_raw = []
        entities_found = []
        reasoning_process = None
        cutoff_ts = None
    else:
        try:
            result = rag.reason(
                query=chat.message,
                cutoff_date=cutoff_date,
                target_firms=firms,
            )
            answer = result.get("answer", "No answer generated.")
            mode = result.get("mode", "unknown")
            paths_raw = result.get("paths", [])
            entities_found = result.get("entities_found", [])
            reasoning_process = result.get("reasoning_process", None)
            cutoff_ts = result.get("cutoff_ts", None)
        except Exception as e:
            import traceback
            logger.error(f"TemporalRAGAgent.reason() error: {e}")
            logger.error(traceback.format_exc())
            answer = f"Error during RAG reasoning: {str(e)}"
            mode = "error"
            paths_raw = []
            entities_found = []
            reasoning_process = None
            cutoff_ts = None

    # ── 4. 경로 데이터 정규화 ──
    paths: List[PathItem] = []
    for p in paths_raw:
        if isinstance(p, dict):
            paths.append(PathItem(
                path=p.get("path", []),
                weight=p.get("weight"),
                snippets=p.get("snippets", []),
            ))
        elif isinstance(p, (list, tuple)):
            paths.append(PathItem(path=list(p)))

    # ── 5. 메타데이터 JSON 직렬화 ──
    meta = {
        "mode": mode,
        "cutoff_date": cutoff_date,
        "cutoff_ts": cutoff_ts,
        "firms": firms,
        "entities_found": entities_found,
        "paths": [p.model_dump() for p in paths],
        "reasoning_process": reasoning_process,
    }
    meta_json_str = json.dumps(meta, ensure_ascii=False)

    # ── 6. 어시스턴트 메시지 DB 저장 ──
    assistant_msg = Chat(
        user_id=current_user.id,
        thread_id=thread.id,
        role="assistant",
        message=answer,
        meta_json=meta_json_str,
    )
    db.add(assistant_msg)

    # ── 7. 스레드 메타 업데이트 ──
    thread.title = chat.message[:50] if len(chat.message) > 50 else chat.message
    thread.last_cutoff_date = cutoff_date
    thread.last_firms = json.dumps(firms or [])
    thread.updated_at = datetime.datetime.now(datetime.timezone.utc)

    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return RagChatResponse(
        thread_uuid=thread_uuid,
        user_message_id=user_msg.id,
        assistant_message_id=assistant_msg.id,
        answer=answer,
        mode=mode,
        cutoff_date=cutoff_date,
        cutoff_ts=cutoff_ts,
        firms=firms,
        paths=paths,
        entities_found=entities_found,
        reasoning_process=reasoning_process,
        created_at=assistant_msg.created_at,
    )
