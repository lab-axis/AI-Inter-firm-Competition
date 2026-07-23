"""
temporal_rag_retriever.py
=========================
Temporal PathRAG 검색 엔진 (PathRAG + Temporally-Bounded Retrieval)

[아키텍처]
  1. Query Entity Extraction  — 쿼리에서 타깃 기업명을 고속 감지
  2. Temporally-Bounded Path Retrieval — Neo4j에서 시간 격리된 인과 경로 탐색
       WHERE r.published_ts < cutoff_ts (에지 레벨 시간 필터)
       WHERE a.published_ts < cutoff_ts (기사 레벨 시간 필터)
  3. Flow-based Path Pruning (Lite) — 중복·저정보 경로 가지치기
  4. Hybrid Fallback — 경로가 없을 때 시간 제약 벡터 검색으로 전환

[학술적 근거]
  PathRAG (BUPT-GAMMA, arXiv:2502.14902, 2025):
    "We retrieve relational paths from the indexing graph rather than
     isolated chunks, using flow-based pruning to eliminate redundant
     information prior to path-based prompting."

  Temporally-Bounded Retrieval (TBR):
    Lopez de Prado (2018). Advances in Financial Machine Learning.
    Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice.
"""

import os
import math
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티: 타깃 기업명 → ticker 역매핑 (고속 정규식)
# ──────────────────────────────────────────────────────────────────────────────
import re
from gdelt_collector import TARGET_FIRMS_MAPPING

_FIRM_NAME_TO_TICKER: Dict[str, str] = {}
for _ticker, _names in TARGET_FIRMS_MAPPING.items():
    for _name in _names:
        _FIRM_NAME_TO_TICKER[_name.lower()] = _ticker


def _extract_tickers_from_query(query: str) -> List[str]:
    """
    쿼리 텍스트에서 TARGET_FIRMS에 속하는 기업 ticker를 결정론적으로 추출.

    예: "NVIDIA and TSMC partnership impact" → ["NVDA", "TSM"]

    SLM NER 없이 고속 처리:
      - 결정론적 방식이므로 결과가 완벽히 재현 가능 (논문 재현성 보장)
      - ticker 자체도 직접 검색 (예: "NVDA" 쿼리 → ["NVDA"])
    """
    query_lower = query.lower()
    found = set()

    # 1. 기업명으로 매칭
    for name_lower, ticker in _FIRM_NAME_TO_TICKER.items():
        if re.search(r'\b' + re.escape(name_lower) + r'\b', query_lower):
            found.add(ticker)

    # 2. ticker 자체로 매칭 (예: "NVDA", "AMD")
    for ticker in TARGET_FIRMS_MAPPING.keys():
        if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', query_lower):
            found.add(ticker)

    return list(found)


# ──────────────────────────────────────────────────────────────────────────────
# TemporalPathRAGRetriever
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPathRAGRetriever:
    """
    PathRAG + Temporally-Bounded Retrieval 통합 검색 엔진.

    retrieve() 메서드 호출 시 수행 단계:
      ① 쿼리 엔티티 추출 (Query Entity Extraction)
      ② 시간 격리 그래프 경로 탐색 (Temporal Graph Path Retrieval via Neo4j Cypher)
      ③ Flow-based Pruning Lite (중복·저가중치 경로 가지치기)
      ④ Path-based Context 조립
      ⑤ [Fallback] 경로 없을 시 시간 제약 VectorRAG 전환

    반환값 구조:
      {
        "paths":   [{"path_str": "NVDA -(CO_OCCURRED_WITH)→ TSM", "weight": 7,
                     "chunks": [...], "published_ts_range": (ts_min, ts_max)}, ...],
        "chunks":  [{"text": ..., "score": ..., "published_date": ...,
                     "article_url": ..., "article_title": ...}, ...],
        "mode":    "pathrag" | "vector_fallback",
        "entities_found": ["NVDA", "TSM"],
        "cutoff_date": "2025-01",
        "cutoff_ts":   1735689600,
      }
    """

    # jinaai/jina-embeddings-v3: 별도 쿼리 프리픽스 불필요 (BGE 전용 설정 제거)
    BGE_QUERY_PREFIX = ""  # Jina 모델은 prefix 없이 사용

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        vllm_embed_url: str = "http://localhost:8000/v1/embeddings",
        embed_model: str    = "jinaai/jina-embeddings-v3",  # (1024D)
        candidate_multiplier: int = 5,
        min_path_weight: int = 1,
        max_path_depth: int  = 2,
    ):
        """
        Args:
            neo4j_uri            : Neo4j Bolt URI
            neo4j_user           : Neo4j 사용자명
            neo4j_password       : Neo4j 비밀번호
            vllm_embed_url       : vLLM /v1/embeddings 엔드포인트
            embed_model          : 임베딩 모델명 — auto_rag_builder.py와 반드시 동일해야 함
                                   (기본: jinaai/jina-embeddings-v3, 1024D)
                                   ※ 모델 불일치 시 벡터 유사도 검색 정확도 급락
            candidate_multiplier : VectorRAG fallback 시 오버샘플링 배수
            min_path_weight      : PathRAG Pruning 최소 에지 가중치 임계값
                                   이 값 미만의 CO_OCCURRED_WITH 에지는 경로에서 제외
            max_path_depth       : 기업 간 최대 탐색 홉 수 (1~3 권장)
        """
        self.driver               = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.vllm_embed_url       = vllm_embed_url
        self.embed_model          = embed_model
        self.candidate_multiplier = candidate_multiplier
        self.min_path_weight      = min_path_weight
        self.max_path_depth       = max_path_depth

    def close(self):
        self.driver.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 핵심 공개 메서드
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        cutoff_date: str,
        top_k: int = 10,
        target_firms: Optional[List[str]] = None,
    ) -> Dict:
        """
        Temporal PathRAG 검색 수행.

        Args:
            query       : 자연어 쿼리
            cutoff_date : 시간 컷오프 (예: "2025-01")
                          이 날짜 이전 기사·관계만 검색 대상 → Look-ahead Bias 차단
            top_k       : 최종 반환 청크 수
            target_firms: 기업 ticker 필터 (None이면 쿼리에서 자동 추출)

        Returns:
            Dict — paths / chunks / mode / entities_found / cutoff_date / cutoff_ts
        """
        cutoff_ts = self._parse_cutoff_date(cutoff_date)
        cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"\n[TemporalRAG] 검색 시작 | cutoff: {cutoff_date} ({cutoff_dt}) | query: {query[:60]}...")

        # ① 엔티티 추출
        entities = target_firms or _extract_tickers_from_query(query)
        print(f"[TemporalRAG] 추출된 기업 엔티티: {entities}")

        # ② 시간 격리 그래프 경로 탐색
        paths = []
        if len(entities) >= 1:
            paths = self._retrieve_temporal_paths(entities, cutoff_ts, top_k)
            print(f"[TemporalRAG] 경로 인출: {len(paths)}개 (Pruning 전)")

            # ③ Flow-based Pruning Lite
            paths = self._flow_based_pruning(paths)
            print(f"[TemporalRAG] Flow Pruning 후: {len(paths)}개 경로 확정")

        # ④ 각 경로의 관련 청크 수집
        if paths:
            paths = self._attach_chunks_to_paths(paths, query, cutoff_ts, chunks_per_path=2)
            mode  = "pathrag"
        else:
            # ⑤ Fallback: 경로 없을 시 시간 제약 VectorRAG
            print(f"[TemporalRAG] 경로 없음 → VectorRAG Fallback 전환")
            mode  = "vector_fallback"

        # ⑤ 시간 제약 벡터 검색 (Fallback 또는 보완 청크)
        vector_chunks = self._vector_search_with_tbr(query, cutoff_ts, top_k, entities)

        return {
            "paths":          paths,
            "chunks":         vector_chunks,
            "mode":           mode,
            "entities_found": entities,
            "cutoff_date":    cutoff_date,
            "cutoff_ts":      cutoff_ts,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Step ②: 시간 격리 경로 탐색 (Temporal Path Retrieval)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_temporal_paths(
        self, entities: List[str], cutoff_ts: int, top_k: int
    ) -> List[Dict]:
        """
        Neo4j Cypher를 통해 기업 간 시간 격리 관계 경로 탐색.

        [핵심 설계]
          WHERE r.published_ts < $cutoff_ts
            → CO_OCCURRED_WITH 에지의 타임스탬프가 cutoff 이전인 경로만 탐색
            → 미래 관계 정보가 경로에 포함되지 않음 (Look-ahead Bias 차단)

          LIMIT $top_k * 3
            → Pruning 전 충분한 후보 확보
        """
        with self.driver.session() as session:
            # 단일 기업 쿼리: 해당 기업과 연결된 모든 기업 경로 탐색
            # 복수 기업 쿼리: 명시된 기업들 사이의 경로 탐색
            if len(entities) == 1:
                cypher = """
                    MATCH (c1:Company {ticker: $ticker})
                    MATCH (c1)-[r:CO_OCCURRED_WITH]-(c2:Company)
                    WHERE r.published_ts < $cutoff_ts
                      AND r.weight >= $min_weight
                      AND c2.ticker <> c1.ticker
                    RETURN
                        c1.ticker AS from_ticker,
                        c1.name   AS from_name,
                        c2.ticker AS to_ticker,
                        c2.name   AS to_name,
                        r.weight  AS weight,
                        r.published_ts AS rel_ts
                    ORDER BY r.weight DESC
                    LIMIT $limit
                """
                result = session.run(
                    cypher,
                    ticker=entities[0],
                    cutoff_ts=cutoff_ts,
                    min_weight=self.min_path_weight,
                    limit=top_k * 3,
                )
            else:
                # 복수 기업: 지정된 기업들 사이의 경로만 탐색
                cypher = """
                    UNWIND $entities AS e1
                    MATCH (c1:Company {ticker: e1})
                    UNWIND $entities AS e2
                    MATCH (c2:Company {ticker: e2})
                    WHERE c1.ticker < c2.ticker
                    MATCH (c1)-[r:CO_OCCURRED_WITH]-(c2)
                    WHERE r.published_ts < $cutoff_ts
                      AND r.weight >= $min_weight
                    RETURN
                        c1.ticker AS from_ticker,
                        c1.name   AS from_name,
                        c2.ticker AS to_ticker,
                        c2.name   AS to_name,
                        r.weight  AS weight,
                        r.published_ts AS rel_ts
                    ORDER BY r.weight DESC
                    LIMIT $limit
                """
                result = session.run(
                    cypher,
                    entities=entities,
                    cutoff_ts=cutoff_ts,
                    min_weight=self.min_path_weight,
                    limit=top_k * 3,
                )

            paths = []
            for rec in result:
                paths.append({
                    "from_ticker": rec["from_ticker"],
                    "from_name":   rec["from_name"],
                    "to_ticker":   rec["to_ticker"],
                    "to_name":     rec["to_name"],
                    "weight":      rec["weight"],
                    "rel_ts":      rec["rel_ts"],
                    "path_str":    (
                        f"[{rec['from_ticker']}] {rec['from_name']} "
                        f"-(CO_OCCURRED_WITH: weight={rec['weight']})→ "
                        f"[{rec['to_ticker']}] {rec['to_name']}"
                    ),
                    "chunks": [],
                })
            return paths

    # ──────────────────────────────────────────────────────────────────────────
    # Step ③: Flow-based Path Pruning (Lite)
    # ──────────────────────────────────────────────────────────────────────────

    def _flow_based_pruning(self, paths: List[Dict]) -> List[Dict]:
        """
        PathRAG Flow-based Pruning의 경량 구현.

        [원문 PathRAG 알고리즘 요약]
          "We treat the graph as a flow network and prune paths whose
           information flow (approximated by edge weight) falls below
           a threshold relative to the maximum flow path."
           (arXiv:2502.14902, Section 3.2)

        [Lite 구현 전략]
          1. weight 기준 정규화 점수(flow_score) 계산
          2. 최대 weight 대비 10% 미만 경로 제거 (저정보 경로)
          3. 동일 기업 쌍의 중복 경로 제거 (방향 무관)
          4. 상위 N개 경로만 유지

        이 방식은 원문의 네트워크 흐름 이론(Network Flow Theory)을
        단순화한 것으로, SLM의 컨텍스트 창을 압축하는 핵심 목적을 달성함.
        """
        if not paths:
            return paths

        # 1. 최대 weight 기준 flow_score 정규화
        max_weight = max(p["weight"] for p in paths)
        for p in paths:
            p["flow_score"] = p["weight"] / max_weight if max_weight > 0 else 0.0

        # 2. 저정보 경로 제거 (flow_score < 10%)
        PRUNE_THRESHOLD = 0.10
        paths = [p for p in paths if p["flow_score"] >= PRUNE_THRESHOLD]

        # 3. 중복 기업 쌍 제거 (방향 무관: A→B == B→A)
        seen_pairs = set()
        unique_paths = []
        for p in paths:
            pair = tuple(sorted([p["from_ticker"], p["to_ticker"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_paths.append(p)

        # 4. flow_score 기준 내림차순 정렬
        unique_paths.sort(key=lambda x: x["flow_score"], reverse=True)

        return unique_paths

    # ──────────────────────────────────────────────────────────────────────────
    # Step ④: 경로에 근거 청크 연결
    # ──────────────────────────────────────────────────────────────────────────

    def _attach_chunks_to_paths(
        self, paths: List[Dict], query: str, cutoff_ts: int, chunks_per_path: int = 2
    ) -> List[Dict]:
        """
        각 경로를 구성하는 기업 쌍과 연관된 실제 기사 청크를 Neo4j에서 인출.

        경로 [NVDA → TSM] 에 대해:
          - NVDA와 TSM 모두 MENTIONED_IN 관계를 가진 Article 조회
          - 그 Article의 HAS_CHUNK 청크를 반환
          - cutoff_ts 이전 청크만 포함 (TBR 보장)
        """
        query_emb = self._embed_query(query)
        if query_emb is None:
            return paths

        with self.driver.session() as session:
            for path in paths:
                tickers = [path["from_ticker"], path["to_ticker"]]
                cypher = """
                    MATCH (c1:Company {ticker: $t1})-[:MENTIONED_IN]->(a:Article)
                    MATCH (c2:Company {ticker: $t2})-[:MENTIONED_IN]->(a)
                    WHERE a.published_ts < $cutoff_ts
                    MATCH (a)-[:HAS_CHUNK]->(ch:Chunk)
                    WHERE ch.published_ts < $cutoff_ts
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           a.url           AS url,
                           a.title         AS title
                    LIMIT $limit
                """
                result = session.run(
                    cypher,
                    t1=tickers[0], t2=tickers[1],
                    cutoff_ts=cutoff_ts,
                    limit=chunks_per_path * 3,
                )
                raw_chunks = [
                    {
                        "text":           rec["text"],
                        "published_date": datetime.fromtimestamp(
                            rec["ts"], tz=timezone.utc
                        ).strftime("%Y-%m-%d") if rec["ts"] else "Unknown",
                        "article_url":    rec["url"],
                        "article_title":  rec["title"],
                        "score":          None,  # 경로 기반 청크는 별도 스코어 없음
                    }
                    for rec in result
                ]
                path["chunks"] = raw_chunks[:chunks_per_path]

        return paths

    # ──────────────────────────────────────────────────────────────────────────
    # Step ⑤: VectorRAG Fallback (TBR 포함)
    # ──────────────────────────────────────────────────────────────────────────

    def _vector_search_with_tbr(
        self,
        query: str,
        cutoff_ts: int,
        top_k: int,
        target_firms: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        시간 제약 벡터 검색 (Temporally-Bounded VectorRAG).

        PathRAG 경로가 없거나 보완 청크가 필요할 때 사용.
        candidate_k = top_k × candidate_multiplier 오버샘플링 후 시간 필터 적용.
        """
        query_emb = self._embed_query(query)
        if query_emb is None:
            print("    [경고] 쿼리 임베딩 실패 — VectorRAG Fallback 건너뜀")
            return []

        candidate_k = top_k * self.candidate_multiplier

        with self.driver.session() as session:
            if target_firms:
                cypher = """
                    CALL db.index.vector.queryNodes('article_chunks', $ck, $emb)
                    YIELD node AS ch, score
                    WHERE ch.published_ts IS NOT NULL
                      AND ch.published_ts < $cutoff_ts
                    MATCH (a:Article)-[:HAS_CHUNK]->(ch)
                    WHERE EXISTS {
                        MATCH (c:Company)-[:MENTIONED_IN]->(a)
                        WHERE c.ticker IN $firms
                    }
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           score,
                           a.url           AS url,
                           a.title         AS title
                    ORDER BY score DESC LIMIT $top_k
                """
                result = session.run(
                    cypher,
                    ck=candidate_k, emb=query_emb,
                    cutoff_ts=cutoff_ts,
                    firms=target_firms,
                    top_k=top_k,
                )
            else:
                cypher = """
                    CALL db.index.vector.queryNodes('article_chunks', $ck, $emb)
                    YIELD node AS ch, score
                    WHERE ch.published_ts IS NOT NULL
                      AND ch.published_ts < $cutoff_ts
                    MATCH (a:Article)-[:HAS_CHUNK]->(ch)
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           score,
                           a.url           AS url,
                           a.title         AS title
                    ORDER BY score DESC LIMIT $top_k
                """
                result = session.run(
                    cypher,
                    ck=candidate_k, emb=query_emb,
                    cutoff_ts=cutoff_ts,
                    top_k=top_k,
                )

            rows = []
            for rec in result:
                ts = rec["ts"]
                rows.append({
                    "text":           rec["text"],
                    "score":          round(float(rec["score"]), 4),
                    "published_date": datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).strftime("%Y-%m-%d") if ts else "Unknown",
                    "article_url":    rec["url"],
                    "article_title":  rec["title"],
                })
            return rows

    # ──────────────────────────────────────────────────────────────────────────
    # 공통 유틸리티
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """
        vLLM 임베딩 API로 쿼리 임베딩 생성.
        jinaai/jina-embeddings-v3: 별도 쿼리 프리픽스 불필요.
        (BGE 모델 사용 시에는 "Represent this sentence: " 프리픽스 필요)
        """
        prefixed = self.BGE_QUERY_PREFIX + query  # Jina 모델: prefix = "" (빈 문자열)
        try:
            payload  = {"model": self.embed_model, "input": [prefixed]}
            resp     = requests.post(self.vllm_embed_url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"    [임베딩 오류] {e}")
            return None

    @staticmethod
    def _parse_cutoff_date(cutoff_date: str) -> int:
        """
        cutoff_date 문자열 → UTC Unix Timestamp(정수).
        지원: "YYYY-MM", "YYYY-MM-DD", ISO 형식
        """
        import calendar
        cutoff_date = cutoff_date.strip()
        try:
            if len(cutoff_date) == 7:
                # "YYYY-MM" 입력 시 해당 월의 마지막 날 23:59:59로 처리 (해당 월 전체 포함)
                year, month = map(int, cutoff_date.split("-"))
                last_day = calendar.monthrange(year, month)[1]
                dt = datetime(year, month, last_day, 23, 59, 59)
            elif len(cutoff_date) == 10:
                # "YYYY-MM-DD" 입력 시 해당 일의 마지막 시간(23:59:59)으로 처리
                dt = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            else:
                dt = datetime.fromisoformat(cutoff_date)
        except ValueError as e:
            raise ValueError(
                f"cutoff_date 형식 오류: '{cutoff_date}'. "
                "지원: 'YYYY-MM', 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS'"
            ) from e
        return int(dt.replace(tzinfo=timezone.utc).timestamp())


# 하위 호환성을 위한 별칭 (기존 코드에서 TemporalRAGRetriever를 import하는 경우 대응)
TemporalRAGRetriever = TemporalPathRAGRetriever


# ──────────────────────────────────────────────────────────────────────────────
# CLI 검증 도구
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Temporal PathRAG Retriever 검증 — PathRAG + TBR"
    )
    parser.add_argument("--query",           type=str, required=True)
    parser.add_argument("--cutoff",          type=str, required=True,  help="예: '2025-01'")
    parser.add_argument("--top-k",           type=int, default=5)
    parser.add_argument("--firms",           type=str, default=None,   help="예: 'NVDA,TSM'")
    parser.add_argument("--vllm-embed-url",  type=str, default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--neo4j-uri",       type=str, default=None)
    parser.add_argument("--neo4j-user",      type=str, default=None)
    parser.add_argument("--neo4j-pass",      type=str, default=None)
    parser.add_argument("--min-path-weight", type=int, default=1)
    args = parser.parse_args()

    uri      = args.neo4j_uri  or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = args.neo4j_user or os.getenv("NEO4J_USER",     "neo4j")
    password = args.neo4j_pass or os.getenv("NEO4J_PASSWORD", "password123")
    firms    = [f.strip() for f in args.firms.split(",")] if args.firms else None

    print("=" * 65)
    print("  Temporal PathRAG Retriever 검증")
    print("=" * 65)
    print(f"  쿼리      : {args.query}")
    print(f"  컷오프    : {args.cutoff}")
    print(f"  기업 필터 : {firms or '자동 추출'}")
    print("=" * 65)

    retriever = TemporalPathRAGRetriever(
        neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
        vllm_embed_url=args.vllm_embed_url,
        min_path_weight=args.min_path_weight,
    )
    result = retriever.retrieve(
        query=args.query,
        cutoff_date=args.cutoff,
        top_k=args.top_k,
        target_firms=firms,
    )

    print(f"\n[검색 모드] {result['mode'].upper()}")
    print(f"[추출 엔티티] {result['entities_found']}")
    print(f"[컷오프 TS] {result['cutoff_ts']} (Unix UTC)")

    print(f"\n{'─'*65}")
    print(f"  인과 경로 ({len(result['paths'])}개)")
    print(f"{'─'*65}")
    for i, path in enumerate(result["paths"], 1):
        print(f"\n[경로 {i}] {path['path_str']}")
        print(f"  flow_score: {path.get('flow_score', 'N/A'):.3f} | weight: {path['weight']}")
        for j, chunk in enumerate(path.get("chunks", []), 1):
            print(f"  └─ 근거 [{j}] ({chunk['published_date']}) {chunk['article_title']}")
            print(f"     {chunk['text'][:150]}...")

    print(f"\n{'─'*65}")
    print(f"  Vector 청크 ({len(result['chunks'])}개, 시간 격리 TBR 적용)")
    print(f"{'─'*65}")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"\n[{i}] score={chunk['score']:.4f} | {chunk['published_date']}")
        print(f"     {chunk['article_title']}")
        print(f"     {chunk['text'][:200]}...")

    retriever.close()
