import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ──────────────────────────────
# Logging
# ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("temporalrag")

# ──────────────────────────────
# Environment Variables
# ──────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded .env from {env_path}")
else:
    logger.warning("No .env file found. Using system environment variables.")

# ──────────────────────────────
# Imports (after env load)
# ──────────────────────────────
from models.database import create_tables, User, SessionLocal
from api.auth.jwt_handler import get_password_hash
from api.features.chats.routes import router as chats_router
from api.features.graph.routes import router as graph_router
from api.features.rag_manager import init_rag_agent


def create_default_users():
    """기본 사용자 3명 생성 (최초 실행 시)."""
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            default_users = [
                {"username": "admin", "email": "admin@temporalrag.example", "password": "Admin@123"},
                {"username": "user1", "email": "user1@temporalrag.example", "password": "Pass@123"},
            ]
            for ud in default_users:
                db.add(User(
                    username=ud["username"],
                    email=ud["email"],
                    hashed_password=get_password_hash(ud["password"]),
                ))
            db.commit()
            logger.info("Default users created: admin / user1 (password: Pass@123)")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # ── Startup ──
    logger.info("Starting TemporalRAG API...")
    create_tables()
    logger.info("SQLite tables ready")
    create_default_users()

    # Initialize TemporalRAGAgent singleton
    init_rag_agent()

    yield

    # ── Shutdown ──
    logger.info("Shutting down TemporalRAG API...")


# ──────────────────────────────
# FastAPI App
# ──────────────────────────────
app = FastAPI(
    title="TemporalRAG API",
    description=(
        "Temporal Knowledge Graph RAG API — "
        "B-MTGNN 예측 근거 설명을 위한 시간 인식 추론 엔진 (Neo4j + DeepSeek-R1 + PathRAG based TemporalRAG)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers
app.include_router(chats_router)
app.include_router(graph_router)


@app.get("/")
async def root():
    return {
        "message": "TemporalRAG API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/token, /register",
            "chats": "/chats/threads, /chats/chat/{thread_uuid}",
            "graph": "/graph/, /graph/stats, /graph/expand/{ticker}",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8002"))
    reload = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    logger.info(f"Starting server on {host}:{port} (reload={reload})")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
