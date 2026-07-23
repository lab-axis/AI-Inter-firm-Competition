"""
graph/routes.py
===============
Neo4j Knowledge Graph 시각화를 위한 API 엔드포인트.

PathRAG의 NetworkX 그래프와 달리, Neo4j Bolt에 직접 Cypher 쿼리를 실행하여
회사/기사/테마 노드와 관계 데이터를 반환합니다.

지원 쿼리 파라미터:
  - cutoff_date : "YYYY-MM" 형식. 해당 월 이전 데이터만 반환 (시간 격리)
  - min_weight  : 최소 에지 가중치 (노이즈 필터링)
  - ticker      : 특정 기업 중심 1-hop 확장 (기업 클릭 시 상세보기)
  - include_articles : Article 노드 포함 여부
  - include_themes   : Theme 노드 포함 여부
"""

import os
import logging
import calendar
import httpx
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import GraphDatabase

from .schemas import GraphResponse, GraphStatsResponse, NodeResponse, EdgeResponse

logger = logging.getLogger("temporalrag.graph")

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


def get_neo4j_driver():
    """Neo4j 드라이버 팩토리 (요청마다 새 세션 사용)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


def _get_database() -> Optional[str]:
    """NEO4J_DATABASE 환경변수. 없으면 None (Neo4j 기본 DB 사용)."""
    return os.getenv("NEO4J_DATABASE", None)


def _cutoff_to_timestamp(cutoff_date: str) -> int:
    """
    "YYYY-MM" 문자열을 해당 월 말일 23:59:59 UTC Unix timestamp로 변환.
    Neo4j의 r.published_ts < cutoff_ts 조건에 사용됩니다.
    """
    try:
        year, month = int(cutoff_date[:4]), int(cutoff_date[5:7])
        last_day = calendar.monthrange(year, month)[1]
        dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        # Fallback: far future
        return int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp())


# ──────────────────────────────
# GET /graph/stats — 시스템 상태 요약
# ──────────────────────────────
@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Neo4j 노드 수, 관계 수 등 통계를 반환합니다 (Dashboard용)."""
    # 백그라운드로 vLLM 헬스 체크 진행
    async def check_vllm(port: int) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"http://localhost:{port}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False

    vllm_embed_ok = await check_vllm(8000)
    vllm_llm_ok = await check_vllm(8001)

    try:
        driver = get_neo4j_driver()
        db = _get_database()
        with driver.session(database=db) as session:
            # 각 카운트를 독립적으로 조회 (MATCH 체이닝 시 빈 집합이면 전체 0 반환 버그 방지)
            company_count = session.run("MATCH (c:Company) RETURN count(c) AS cnt").single()["cnt"]
            article_count = session.run("MATCH (a:Article) RETURN count(a) AS cnt").single()["cnt"]
            theme_count   = session.run("MATCH (t:Theme)   RETURN count(t) AS cnt").single()["cnt"]
            rel_count     = session.run("MATCH ()-[r:CO_OCCURRED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]

            return GraphStatsResponse(
                company_count=company_count,
                article_count=article_count,
                theme_count=theme_count,
                relation_count=rel_count,
                neo4j_connected=True,
                vllm_embedding_connected=vllm_embed_ok,
                vllm_llm_connected=vllm_llm_ok,
            )
    except Exception as e:
        logger.error(f"Neo4j stats query failed: {e}")

    return GraphStatsResponse(
        company_count=0, article_count=0, theme_count=0,
        relation_count=0, neo4j_connected=False,
        vllm_embedding_connected=vllm_embed_ok,
        vllm_llm_connected=vllm_llm_ok,
    )



# ──────────────────────────────
# GET /graph/ — 기본 기업 관계망
# ──────────────────────────────
@router.get("/", response_model=GraphResponse)
async def get_graph(
    cutoff_date: Optional[str] = Query(None, description="YYYY-MM 형식 시간 컷오프"),
    min_weight: float = Query(1.0, description="최소 에지 가중치"),
    include_articles: bool = Query(False, description="Article 노드 포함"),
    include_themes: bool = Query(False, description="Theme 노드 포함"),
):
    """
    기업 노드(Company)와 CO_OCCURRED_WITH 에지를 반환합니다.
    Legend 토글에 따라 Article/Theme 노드도 선택적으로 포함할 수 있습니다.
    """
    try:
        driver = get_neo4j_driver()
        cutoff_ts = _cutoff_to_timestamp(cutoff_date) if cutoff_date else None
        nodes: List[NodeResponse] = []
        edges: List[EdgeResponse] = []
        seen_node_ids = set()

        db = _get_database()
        with driver.session(database=db) as session:
            if cutoff_ts:
                co_query = """
                    MATCH (c1:Company)-[r:CO_OCCURRED_WITH]->(c2:Company)
                    WHERE r.published_ts <= $cutoff_ts AND r.weight >= $min_weight
                    RETURN c1, c2, r
                """
            else:
                co_query = """
                    MATCH (c1:Company)-[r:CO_OCCURRED_WITH]->(c2:Company)
                    WHERE r.weight >= $min_weight
                    RETURN c1, c2, r
                """

            result = session.run(
                co_query,
                cutoff_ts=cutoff_ts or 9999999999,
                min_weight=min_weight,
            )

            company_degrees = {}
            raw_edges = []

            for record in result:
                c1 = record["c1"]
                c2 = record["c2"]
                r = record["r"]

                for company in [c1, c2]:
                    cid = company.get("ticker", str(company.id))
                    if cid not in seen_node_ids:
                        nodes.append(NodeResponse(
                            id=cid,
                            label=company.get("name", cid),
                            type="Company",
                            ticker=company.get("ticker"),
                            name=company.get("name"),
                            degree=company_degrees.get(cid, 0),
                        ))
                        seen_node_ids.add(cid)

                raw_edges.append((c1.get("ticker"), c2.get("ticker"), r))

            # Node degree 계산 (에지 가중치 합)
            degree_map: dict = {}
            for src, tgt, r in raw_edges:
                w = r.get("weight", 1.0) or 1.0
                degree_map[src] = degree_map.get(src, 0) + w
                degree_map[tgt] = degree_map.get(tgt, 0) + w

            # Node degree 업데이트
            nodes = [
                NodeResponse(
                    id=n.id, label=n.label, type=n.type,
                    ticker=n.ticker, name=n.name,
                    degree=degree_map.get(n.id, 0),
                )
                for n in nodes
            ]

            # 에지 추가
            for idx, (src, tgt, r) in enumerate(raw_edges):
                edges.append(EdgeResponse(
                    id=f"co_{src}_{tgt}_{idx}",
                    source=src,
                    target=tgt,
                    type="CO_OCCURRED_WITH",
                    weight=r.get("weight"),
                    published_ts=r.get("published_ts"),
                ))

            # ── 2. Article 노드 (Legend 토글 시) ──
            if include_articles:
                article_query = """
                    MATCH (c:Company)-[:MENTIONED_IN]->(a:Article)
                    WHERE c.ticker IN $tickers
                """ + (f"AND a.published_ts <= {cutoff_ts}" if cutoff_ts else "") + """
                    RETURN c.ticker AS ticker, a LIMIT 500
                """
                tickers = [n.id for n in nodes if n.type == "Company"]
                art_result = session.run(article_query, tickers=tickers)
                for art_rec in art_result:
                    a = art_rec["a"]
                    aid = f"article_{a.get('id', str(a.id))}"
                    if aid not in seen_node_ids:
                        nodes.append(NodeResponse(
                            id=aid,
                            label=a.get("title", aid)[:60],
                            type="Article",
                            title=a.get("title"),
                            url=a.get("url"),
                            published_date=a.get("date"),
                        ))
                        seen_node_ids.add(aid)
                    edges.append(EdgeResponse(
                        id=f"mi_{art_rec['ticker']}_{aid}",
                        source=art_rec["ticker"],
                        target=aid,
                        type="MENTIONED_IN",
                    ))

            # ── 3. Theme 노드 (Legend 토글 시) ──
            if include_themes and include_articles:
                theme_query = """
                    MATCH (a:Article)-[:HAS_THEME]->(t:Theme)
                    WHERE a.id IN $article_ids
                    RETURN a.id AS article_id, t LIMIT 300
                """
                article_ids = [n.id.replace("article_", "") for n in nodes if n.type == "Article"]
                if article_ids:
                    th_result = session.run(theme_query, article_ids=article_ids)
                    for th_rec in th_result:
                        t = th_rec["t"]
                        tid = f"theme_{t.get('name', str(t.id))}"
                        if tid not in seen_node_ids:
                            nodes.append(NodeResponse(
                                id=tid,
                                label=t.get("name", tid),
                                type="Theme",
                                theme_name=t.get("name"),
                            ))
                            seen_node_ids.add(tid)
                        a_id = f"article_{th_rec['article_id']}"
                        edges.append(EdgeResponse(
                            id=f"ht_{a_id}_{tid}",
                            source=a_id,
                            target=tid,
                            type="HAS_THEME",
                        ))

        driver.close()

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            meta={
                "cutoff_date": cutoff_date,
                "min_weight": min_weight,
                "include_articles": include_articles,
                "include_themes": include_themes,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )

    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j query error: {str(e)}")


# ──────────────────────────────
# GET /graph/expand/{ticker} — 기업 1-Hop 확장
# ──────────────────────────────
@router.get("/expand/{ticker}", response_model=GraphResponse)
async def expand_company(
    ticker: str,
    cutoff_date: Optional[str] = Query(None),
):
    """특정 기업 클릭 시 해당 기업과 연결된 Article/Theme 노드까지 확장합니다."""
    try:
        driver = get_neo4j_driver()
        cutoff_ts = _cutoff_to_timestamp(cutoff_date) if cutoff_date else None
        nodes: List[NodeResponse] = []
        edges: List[EdgeResponse] = []
        seen_node_ids = set()

        db = _get_database()
        with driver.session(database=db) as session:
            # 기업 노드
            comp_result = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c",
                ticker=ticker,
            )
            comp_row = comp_result.single()
            if comp_row:
                c = comp_row["c"]
                nodes.append(NodeResponse(
                    id=ticker, label=c.get("name", ticker),
                    type="Company", ticker=ticker, name=c.get("name"),
                ))
                seen_node_ids.add(ticker)

            # 연결된 기사 (최대 50개)
            art_query = """
                MATCH (c:Company {ticker: $ticker})-[:MENTIONED_IN]->(a:Article)
            """ + (f"WHERE a.published_ts <= {cutoff_ts}" if cutoff_ts else "") + """
                RETURN a ORDER BY a.published_ts DESC LIMIT 50
            """
            art_result = session.run(art_query, ticker=ticker)
            for rec in art_result:
                a = rec["a"]
                aid = f"article_{a.get('id', str(a.id))}"
                if aid not in seen_node_ids:
                    nodes.append(NodeResponse(
                        id=aid,
                        label=(a.get("title") or aid)[:60],
                        type="Article",
                        title=a.get("title"),
                        url=a.get("url"),
                        published_date=a.get("date"),
                    ))
                    seen_node_ids.add(aid)
                edges.append(EdgeResponse(
                    id=f"mi_{ticker}_{aid}",
                    source=ticker, target=aid,
                    type="MENTIONED_IN",
                ))

            # 연결된 테마 (기사 경유)
            article_ids = [n.id.replace("article_", "") for n in nodes if n.type == "Article"]
            if article_ids:
                th_result = session.run("""
                    MATCH (a:Article)-[:HAS_THEME]->(t:Theme)
                    WHERE a.id IN $article_ids
                    RETURN a.id AS article_id, t LIMIT 100
                """, article_ids=article_ids)
                for th_rec in th_result:
                    t = th_rec["t"]
                    tid = f"theme_{t.get('name', str(t.id))}"
                    if tid not in seen_node_ids:
                        nodes.append(NodeResponse(
                            id=tid, label=t.get("name", tid),
                            type="Theme", theme_name=t.get("name"),
                        ))
                        seen_node_ids.add(tid)
                    a_id = f"article_{th_rec['article_id']}"
                    edges.append(EdgeResponse(
                        id=f"ht_{a_id}_{tid}",
                        source=a_id, target=tid,
                        type="HAS_THEME",
                    ))

        driver.close()
        return GraphResponse(nodes=nodes, edges=edges)

    except Exception as e:
        logger.error(f"Graph expand failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j expand error: {str(e)}")
