"""Retrieve temporally filtered firm relationships and article chunks from Neo4j.

Firm aliases identify query entities. Relationship candidates are filtered
by stored publication timestamps, ranked by time-decayed edge weights and
pruned by relative weight. Vector retrieval supplies fallback or supplementary
chunks. Co-occurrence relationships do not by themselves establish causality.

Related work: PathRAG (BUPT-GAMMA, arXiv:2502.14902, 2025). The pruning
implemented here uses relative edge weights, not a general network-flow solver."""

import os
import math
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv


# Firm alias and ticker matching
import re
from .gdelt_collector import TARGET_FIRMS_MAPPING

_FIRM_NAME_TO_TICKER: Dict[str, str] = {}
for _ticker, _names in TARGET_FIRMS_MAPPING.items():
    for _name in _names:
        _FIRM_NAME_TO_TICKER[_name.lower()] = _ticker


def _extract_tickers_from_query(query: str) -> List[str]:
    """Extract configured firm tickers from query text using alias and ticker patterns.
    
    For example, a query mentioning NVIDIA and TSMC can match NVDA and TSM.
    Matching is rule-based; no language-model entity extraction is performed."""
    query_lower = query.lower()
    found = set()

    # Match company-name aliases.
    for name_lower, ticker in _FIRM_NAME_TO_TICKER.items():
        if re.search(r'\b' + re.escape(name_lower) + r'\b', query_lower):
            found.add(ticker)

    # Also match ticker symbols directly.
    for ticker in TARGET_FIRMS_MAPPING.keys():
        if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', query_lower):
            found.add(ticker)

    return list(found)



class TemporalPathRAGRetriever:
    """Retrieve firm co-occurrence relationships and supporting article chunks.
    
    The result contains paths, chunks, retrieval mode, matched entities and cutoff
    metadata. Chunks include text, source title, URL and publication date. Vector
    retrieval is used when relationships are absent or additional context is needed."""

    # No instruction prefix is added for the configured Jina query embeddings.
    BGE_QUERY_PREFIX = ""  

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = None,
        vllm_embed_url: str = "http://localhost:8000/v1/embeddings",
        embed_model: str    = "jinaai/jina-embeddings-v3",
        candidate_multiplier: int = 5,
        min_path_weight: int = 1,
        max_path_depth: int  = 2,
    ):
        """Configure the Neo4j connection, embedding service and retrieval limits.
        
        The embedding model must match the model used to build the vector index.
        A None database name selects the Neo4j default database.
        candidate_multiplier controls vector-candidate oversampling; min_path_weight
        filters relationship weights. max_path_depth is stored for interface
        compatibility, but the current Cypher queries retrieve one-hop relationships."""
        self.driver               = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.neo4j_database       = neo4j_database
        self.vllm_embed_url       = vllm_embed_url
        self.embed_model          = embed_model
        self.candidate_multiplier = candidate_multiplier
        self.min_path_weight      = min_path_weight
        self.max_path_depth       = max_path_depth

    def close(self):
        self.driver.close()

    # Public retrieval interface

    def retrieve(
        self,
        query: str,
        cutoff_date: str,
        top_k: int = 5,
        target_firms: Optional[List[str]] = None,
    ) -> Dict:
        """Retrieve context for a query at the supplied cutoff.
        
        cutoff_date accepts a month, day or ISO timestamp. Stored publication
        timestamps are compared to that cutoff. target_firms overrides automatic
        entity extraction; top_k controls the requested context size.
        
        Return paths, chunks, mode, entities_found, cutoff_date and cutoff_ts."""
        cutoff_ts = self._parse_cutoff_date(cutoff_date)
        cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"\n[TemporalRAG] 검색 시작 | cutoff: {cutoff_date} ({cutoff_dt}) | query: {query[:60]}...")

        # Resolve query entities.
        entities = target_firms or _extract_tickers_from_query(query)
        print(f"[TemporalRAG] 추출된 기업 엔티티: {entities}")

        # Retrieve relationships subject to the publication-time filter.
        paths = []
        if len(entities) >= 1:
            paths = self._retrieve_temporal_paths(entities, cutoff_ts, top_k)
            print(f"[TemporalRAG] 경로 인출: {len(paths)}개")

            # Prune relationship candidates by relative weight.
            paths = self._flow_based_pruning(paths)
            print(f"[TemporalRAG] Flow Pruning 후: {len(paths)}개 경로 확정")

        # Attach article chunks to the retained firm pairs.
        if paths:
            paths = self._attach_chunks_to_paths(paths, query, cutoff_ts, chunks_per_path=2)
            mode  = "pathrag"
        else:
            # Use vector retrieval when no relationship candidates remain.
            print(f"[TemporalRAG] 경로 없음 → VectorRAG Fallback 전환")
            mode  = "vector_fallback"

        # Retrieve temporally filtered fallback or supplementary chunks.
        vector_chunks = self._vector_search_with_tbr(query, cutoff_ts, top_k, entities)

        return {
            "paths":          paths,
            "chunks":         vector_chunks,
            "mode":           mode,
            "entities_found": entities,
            "cutoff_date":    cutoff_date,
            "cutoff_ts":      cutoff_ts,
        }

    # Temporally filtered relationship retrieval

    def _retrieve_temporal_paths(
        self, entities: List[str], cutoff_ts: int, top_k: int
    ) -> List[Dict]:
        """Retrieve one-hop CO_OCCURRED_WITH relationships using Neo4j Cypher.
        
        Require r.published_ts <= cutoff_ts and the minimum raw edge weight.
        Rank candidates by time-decayed weight and request up to top_k * 3 rows
        before pruning. The timestamp filter depends on stored graph metadata."""
        with self.driver.session(database=self.neo4j_database) as session:
            # For one entity, search its neighboring firms.
            # For multiple entities, restrict results to pairs in that set.
            if len(entities) == 1:
                cypher = """
                    MATCH (c1:Company {ticker: $ticker})
                    MATCH (c1)-[r:CO_OCCURRED_WITH]-(c2:Company)
                    WHERE r.published_ts <= $cutoff_ts
                      AND r.weight >= $min_weight
                      AND c2.ticker <> c1.ticker
                    WITH c1, c2, r, ($cutoff_ts - r.published_ts) / 86400.0 AS days_diff
                    WITH c1, c2, r, r.weight * (0.05 + 0.95 * exp(-0.0077 * days_diff)) AS decayed_weight
                    RETURN
                        c1.ticker AS from_ticker,
                        c1.name   AS from_name,
                        c2.ticker AS to_ticker,
                        c2.name   AS to_name,
                        decayed_weight  AS weight,
                        r.published_ts AS rel_ts
                    ORDER BY weight DESC
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
                # Search only pairs among the requested firms.
                cypher = """
                    UNWIND $entities AS e1
                    MATCH (c1:Company {ticker: e1})
                    UNWIND $entities AS e2
                    MATCH (c2:Company {ticker: e2})
                    WHERE c1.ticker < c2.ticker
                    MATCH (c1)-[r:CO_OCCURRED_WITH]-(c2)
                    WHERE r.published_ts <= $cutoff_ts
                      AND r.weight >= $min_weight
                    WITH c1, c2, r, ($cutoff_ts - r.published_ts) / 86400.0 AS days_diff
                    WITH c1, c2, r, r.weight * (0.05 + 0.95 * exp(-0.0077 * days_diff)) AS decayed_weight
                    RETURN
                        c1.ticker AS from_ticker,
                        c1.name   AS from_name,
                        c2.ticker AS to_ticker,
                        c2.name   AS to_name,
                        decayed_weight  AS weight,
                        r.published_ts AS rel_ts
                    ORDER BY weight DESC
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

    # Relative-weight pruning

    def _flow_based_pruning(self, paths: List[Dict]) -> List[Dict]:
        """Rank and prune candidates using their supplied relationship weights.
        
        Normalize by the maximum weight, discard scores below 0.1, deduplicate
        unordered firm pairs, and retain the highest-scoring candidates. This is
        a relative-weight heuristic, not an implementation of network-flow optimization."""
        if not paths:
            return paths

        # Normalize candidate weights by the maximum weight.
        max_weight = max(p["weight"] for p in paths)
        for p in paths:
            p["flow_score"] = p["weight"] / max_weight if max_weight > 0 else 0.0

        # Discard candidates with a normalized weight below 0.1.
        PRUNE_THRESHOLD = 0.10
        paths = [p for p in paths if p["flow_score"] >= PRUNE_THRESHOLD]

        # Deduplicate unordered firm pairs.
        seen_pairs = set()
        unique_paths = []
        for p in paths:
            pair = tuple(sorted([p["from_ticker"], p["to_ticker"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_paths.append(p)

        # Sort by normalized weight in descending order.
        unique_paths.sort(key=lambda x: x["flow_score"], reverse=True)

        return unique_paths

    # Supporting article chunks

    def _attach_chunks_to_paths(
        self, paths: List[Dict], query: str, cutoff_ts: int, chunks_per_path: int = 2
    ) -> List[Dict]:
        """Attach chunks from articles that mention both firms in each retained pair.
        
        Article and chunk timestamps must be at or before cutoff_ts. Candidates
        are ordered by target-firm relevance and publication time."""
        query_emb = self._embed_query(query)
        if query_emb is None:
            return paths

        with self.driver.session(database=self.neo4j_database) as session:
            for path in paths:
                tickers = [path["from_ticker"], path["to_ticker"]]
                cypher = """
                    MATCH (c1:Company {ticker: $t1})-[:MENTIONED_IN]->(a:Article)
                    MATCH (c2:Company {ticker: $t2})-[:MENTIONED_IN]->(a)
                    WHERE a.published_ts <= $cutoff_ts
                    MATCH (a)-[:HAS_CHUNK]->(ch:Chunk)
                    WHERE ch.published_ts <= $cutoff_ts
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           a.url           AS url,
                           a.title         AS title,
                           CASE
                             WHEN toLower(coalesce(a.title, '')) CONTAINS toLower($t1)
                               OR toLower(coalesce(ch.text, '')) CONTAINS toLower($t1)
                               OR toLower(coalesce(a.title, '')) CONTAINS toLower($t1_name)
                               OR toLower(coalesce(ch.text, '')) CONTAINS toLower($t1_name)
                             THEN 0 ELSE 1
                           END AS target_priority
                    ORDER BY target_priority ASC, ch.published_ts DESC
                    LIMIT $limit
                """
                result = session.run(
                    cypher,
                    t1=tickers[0], t2=tickers[1],
                    t1_name=path.get("from_name", ""),
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
                        "score":          None,  # No vector similarity score is assigned to relationship-derived chunks.
                    }
                    for rec in result
                ]
                path["chunks"] = raw_chunks[:chunks_per_path]

        return paths

    # Temporally filtered vector retrieval

    def _vector_search_with_tbr(
        self,
        query: str,
        cutoff_ts: int,
        top_k: int,
        target_firms: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Retrieve vector candidates and filter them by publication time.
        
        Oversample top_k * candidate_multiplier candidates before applying the
        time filter. Use the results as fallback or supplementary context."""
        query_emb = self._embed_query(query)
        if query_emb is None:
            print("    [경고] 쿼리 임베딩 실패 - VectorRAG Fallback 건너뜀")
            return []

        candidate_k = top_k * self.candidate_multiplier

        with self.driver.session(database=self.neo4j_database) as session:
            if target_firms:
                cypher = """
                    CALL db.index.vector.queryNodes('article_chunks', $ck, $emb)
                    YIELD node AS ch, score
                    WHERE ch.published_ts IS NOT NULL
                      AND ch.published_ts <= $cutoff_ts
                    MATCH (a:Article)-[:HAS_CHUNK]->(ch)
                    WHERE EXISTS {
                        MATCH (c:Company)-[:MENTIONED_IN]->(a)
                        WHERE c.ticker IN $firms
                    }
                    WITH ch, score, a,
                         ($cutoff_ts - ch.published_ts) / 86400.0 AS days_diff
                    WITH ch, score, a, days_diff,
                         score * (0.05 + 0.95 * exp(-0.0077 * days_diff)) AS final_score
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           score           AS semantic_score,
                           final_score     AS score,
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
                      AND ch.published_ts <= $cutoff_ts
                    MATCH (a:Article)-[:HAS_CHUNK]->(ch)
                    WITH ch, score, a,
                         ($cutoff_ts - ch.published_ts) / 86400.0 AS days_diff
                    WITH ch, score, a, days_diff,
                         score * (0.05 + 0.95 * exp(-0.0077 * days_diff)) AS final_score
                    RETURN ch.text         AS text,
                           ch.published_ts AS ts,
                           score           AS semantic_score,
                           final_score     AS score,
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
                    "semantic_score": round(float(rec["semantic_score"]), 4),
                    "score":          round(float(rec["score"]), 4),
                    "published_date": datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).strftime("%Y-%m-%d") if ts else "Unknown",
                    "article_url":    rec["url"],
                    "article_title":  rec["title"],
                })
            return rows

    # Embedding and date utilities

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embed the query through the configured HTTP endpoint.
        
        The configured query prefix is prepended before the request."""
        prefixed = self.BGE_QUERY_PREFIX + query  # The default query prefix is empty.
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
        """Convert a month, day or ISO date string to a Unix timestamp.
        
        YYYY-MM resolves to the last second of that month in UTC; YYYY-MM-DD
        resolves to the last second of that day in UTC. Other supported strings
        are parsed with datetime.fromisoformat, assigned UTC and converted to a
        timestamp. Supplied timezone offsets are replaced rather than converted."""
        import calendar as _calendar
        cutoff_date = cutoff_date.strip()
        try:
            if len(cutoff_date) == 7:   # YYYY-MM
                year  = int(cutoff_date[:4])
                month = int(cutoff_date[5:7])
                last_day = _calendar.monthrange(year, month)[1]
                dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            elif len(cutoff_date) == 10:  # YYYY-MM-DD
                base = datetime.strptime(cutoff_date, "%Y-%m-%d")
                dt = base.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            else:
                dt = datetime.fromisoformat(cutoff_date).replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(
                f"cutoff_date 형식 오류: '{cutoff_date}'. "
                "지원: 'YYYY-MM', 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS'"
            ) from e
        return int(dt.timestamp())


# Retain the previous retriever name as a compatibility alias.
TemporalRAGRetriever = TemporalPathRAGRetriever


if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Temporal Retriever 검증 - PathRAG + TBR"
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
    print("  Temporal Retriever 검증")
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
