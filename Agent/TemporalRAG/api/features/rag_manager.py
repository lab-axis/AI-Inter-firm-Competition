"""
rag_manager.py
==============
TemporalRAGAgent 싱글턴 팩토리.

FastAPI lifespan에서 초기화하고, 각 라우터가 get_rag_agent()로 참조합니다.
Neo4j 연결 및 vLLM 설정은 .env 환경변수에서 읽습니다.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("temporalrag")

_rag_agent: Optional[object] = None


def init_rag_agent() -> None:
    """애플리케이션 시작 시 1회 호출하여 TemporalRAGAgent를 초기화합니다."""
    global _rag_agent

    try:
        # Import from engine package (relative to TemporalRAG root)
        from engine.temporal_rag_agent import TemporalRAGAgent

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        neo4j_database = os.getenv("NEO4J_DATABASE", None)
        embedding_url = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8000/v1/embeddings")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        llm_url = os.getenv("VLLM_LLM_URL", "http://localhost:8001/v1/chat/completions")
        llm_model = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        top_k = int(os.getenv("TOP_K", 10))

        _rag_agent = TemporalRAGAgent(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            vllm_embed_url=embedding_url,
            embed_model=embedding_model,
            vllm_chat_url=llm_url,
            llm_model=llm_model,
            top_k=top_k,
        )
        logger.info(f"TemporalRAGAgent initialized successfully (database: {neo4j_database or 'default'})")
    except Exception as e:
        logger.error(f"Failed to initialize TemporalRAGAgent: {e}")
        logger.warning("RAG agent is not available. Chat endpoints will return errors.")
        _rag_agent = None


def get_rag_agent():
    """FastAPI dependency: 라우터에서 RAG 에이전트 인스턴스를 가져옵니다."""
    return _rag_agent
