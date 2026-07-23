from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class NodeResponse(BaseModel):
    id: str
    label: str
    type: str          # "Company" | "Article" | "Theme"
    ticker: Optional[str] = None
    name: Optional[str] = None
    degree: Optional[float] = None   # total edge weight sum (for node sizing)
    # Article-specific
    title: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[str] = None
    # Theme-specific
    theme_name: Optional[str] = None


class EdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    type: str          # "CO_OCCURRED_WITH" | "MENTIONED_IN" | "HAS_THEME"
    weight: Optional[float] = None
    published_ts: Optional[int] = None


class GraphResponse(BaseModel):
    nodes: List[NodeResponse]
    edges: List[EdgeResponse]
    meta: Optional[Dict[str, Any]] = None


class GraphStatsResponse(BaseModel):
    company_count: int
    article_count: int
    theme_count: int
    relation_count: int
    neo4j_connected: bool
    vllm_embedding_connected: Optional[bool] = None
    vllm_llm_connected: Optional[bool] = None
