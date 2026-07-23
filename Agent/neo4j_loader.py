import os
import argparse
import calendar
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from gdelt_collector import TARGET_FIRMS_MAPPING

# ==============================================================================
# neo4j_loader.py
# ==============================================================================
# 역할: GDELT 정제 데이터(gdelt_cleaned.csv)를 로드하여 Neo4j Graph DB에
#       트랜잭션 배치 최적화(UNWIND)를 통해 구조화된 그래프망으로 적재한다.
#
# [PathRAG 아키텍처 변경 내역]
#   - 기업 간 에지 타입을 PARTNERS_WITH/COMPETES_WITH/SUPPLIES_TO 혼합 분류에서
#     단일 CO_OCCURRED_WITH 로 통합. 에지에 weight(빈도), published_ts(Unix int)를
#     함께 저장하여 PathRAG 경로 탐색 시 시간 기반 필터링 지원.
#   - 이유: 제목/URL 키워드만으로 관계 유형을 분류하면 노이즈 비율이 높아
#           PathRAG Flow-based Pruning의 정밀도를 저해함. 단일 에지 + weight 방식이
#           학술적으로 더 견고하며(재현성 ↑), 추후 관계 타입 세분화는 SLM NER을
#           통해 별도로 처리 가능.
# ==============================================================================

class Neo4jGraphLoader:
    def __init__(self, uri, user, password, database=None):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database  # None이면 Neo4j 기본 DB 사용

    def close(self):
        self.driver.close()

    def clear_database(self):
        """데이터베이스 초기화"""
        print("\n=== 0. Neo4j 데이터베이스 초기화 ===")
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"    [완료] 노드 {summary.counters.nodes_deleted}개, "
                  f"관계 {summary.counters.relationships_deleted}개 삭제 완료.\n\n")

    def create_schema_constraints(self):
        """Neo4j 인덱스 및 고유성 제약 조건 생성"""
        print("=== 1. Neo4j 인덱스 및 제약조건 생성 ===")
        with self.driver.session(database=self.database) as session:
            # 노드 고유성 제약
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE")

            # Article 인덱스
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.date)")
            # [TBR 핵심] Unix timestamp 인덱스 — WHERE a.published_ts < $cutoff_ts 쿼리 성능 보장
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.published_ts)")
            # [TBR 보조] YYYYMM 월 인덱스 — 월 단위 쿼리 보조
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.published_month)")

            # [PathRAG 핵심] CO_OCCURRED_WITH 에지 published_ts 인덱스
            # → 경로 탐색 시 "cutoff 이전 관계만" 동적 필터링에 필수
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:CO_OCCURRED_WITH]-() ON (r.published_ts)")

            print("    [완료] 제약 조건 및 인덱스 세팅 완료.")

    def load_companies(self):
        """30개 기업(TARGET_FIRMS) 마스터 노드 생성"""
        print("\n=== 2. 30개 Target 기업 마스터 노드 적재 ===")
        companies_batch = [
            {"ticker": ticker, "name": names[0]}
            for ticker, names in TARGET_FIRMS_MAPPING.items()
        ]
        with self.driver.session(database=self.database) as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (c:Company {ticker: row.ticker})
                SET c.name = row.name
            """, batch=companies_batch)
        print(f"    [완료] {len(companies_batch)}개 기업 마스터 노드 적재 완료.")

    def _reverse_organization_lookup(self, org_name):
        """GDELT 조직명 → 타깃 기업 ticker 역매핑"""
        if not isinstance(org_name, str):
            return None
        org_name_lower = org_name.lower().strip()
        for ticker, names in TARGET_FIRMS_MAPPING.items():
            for name in names:
                if name.lower() in org_name_lower or org_name_lower in name.lower():
                    return ticker
        return None

    @staticmethod
    def _parse_gdelt_date(date_str):
        """
        GDELT GKG 날짜(YYYYMMDDHHMMSS 또는 YYYYMMDD 또는 YYYY-MM-DD)를 파싱하여
        UTC 기준 Unix Timestamp(정수)와 YYYYMM 정수 월을 반환.

        반환값: (published_ts: int, published_month: int)
          - published_ts   : Unix Epoch 초 — Cypher 부등호 비교용
          - published_month: YYYYMM 정수  — 월 단위 직관 쿼리용

        [TBR 설계 근거]
          정수 비교(published_ts < cutoff_ts)는 문자열 날짜 비교보다 Neo4j에서
          3~5배 빠르며, UTC 기준으로 통일되므로 논문 재현성을 완벽히 보장.
        """
        date_str = str(date_str).strip()
        try:
            # YYYY-MM-DD (gdelt_collector preprocess 출력 형식)
            if "-" in date_str and len(date_str) >= 10:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            elif len(date_str) >= 14:   # YYYYMMDDHHMMSS
                dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:    # YYYYMMDD
                dt = datetime.strptime(date_str[:8], "%Y%m%d")
            else:
                return None, None
            dt_utc = dt.replace(tzinfo=timezone.utc)
            published_ts    = int(dt_utc.timestamp())
            published_month = int(dt_utc.strftime("%Y%m"))
            return published_ts, published_month
        except ValueError:
            return None, None

    def process_and_load_articles(self, cleaned_csv_path, batch_size=500):
        """
        정제된 GDELT CSV를 배치 처리하여 Neo4j에 적재.

        생성 노드/관계:
          - (:Article)         — 기사 노드 (published_ts, published_month 포함)
          - (:Company)         — 기업 노드 (이미 생성됨)
          - (:Theme)           — 테마 노드
          - (Company)-[:MENTIONED_IN]->(Article)          — 기사 언급 관계
          - (Article)-[:HAS_THEME]->(Theme)               — 테마 연결
          - (Company)-[:CO_OCCURRED_WITH {weight, published_ts}]->(Company)
                               — [PathRAG 핵심] 공동 출현 기반 기업 간 관계
                                 weight: 함께 등장한 기사 수 누적
                                 published_ts: 해당 기사의 Unix timestamp
                                 → PathRAG 경로 탐색 시 시간 필터 적용 가능
        """
        if not os.path.exists(cleaned_csv_path):
            print(f"[오류] 정제된 GDELT CSV 파일이 없습니다: {cleaned_csv_path}")
            return

        df = pd.read_csv(cleaned_csv_path)
        print(f"\n=== 3. GDELT 뉴스기사 & 관계 스키마 대량 적재 ({len(df)}건) ===")
        print(f"    [TemporalRAG] CO_OCCURRED_WITH 에지에 published_ts·weight 저장 활성화")

        articles_batch      = []
        mentions_batch      = []
        themes_batch        = []
        co_occurrence_batch = []

        for idx, row in df.iterrows():
            article_id      = str(row['article_id'])
            date            = str(row['date'])
            sentiment_score = float(row['sentiment_score'])
            title           = str(row['title'])
            url             = str(row['url'])
            source          = str(row['source'])

            # ── [TBR] 날짜 → Unix timestamp + 월 정수 변환 ──────────────────
            published_ts, published_month = self._parse_gdelt_date(date)

            articles_batch.append({
                "article_id":      article_id,
                "date":            date,
                "published_ts":    published_ts,
                "published_month": published_month,
                "sentiment_score": sentiment_score,
                "title":           title,
                "url":             url,
                "source":          source
            })

            # ── 기업 언급 파싱 ───────────────────────────────────────────────
            mentioned_tickers = set()
            orgs_str = row.get('organizations', '')
            if isinstance(orgs_str, str):
                for org in [o.strip() for o in orgs_str.split(';') if o.strip()]:
                    # GDELT 포맷 (OrgName,Offset)에서 이름만 추출
                    org_name_only = org.split(',')[0].strip()
                    if not org_name_only:
                        continue
                    ticker = self._reverse_organization_lookup(org_name_only)
                    if ticker:
                        mentioned_tickers.add(ticker)

            for ticker in mentioned_tickers:
                mentions_batch.append({"ticker": ticker, "article_id": article_id})

            # ── [PathRAG 핵심] CO_OCCURRED_WITH 에지 생성 ────────────────────
            # 동일 기사에 두 기업이 함께 등장 → 시간 스탬프와 가중치를 가진 단일 에지
            # weight는 나중에 MERGE 시 ON MATCH SET ... += 1 로 누적됨
            tickers_list = sorted(mentioned_tickers)
            for i in range(len(tickers_list)):
                for j in range(i + 1, len(tickers_list)):
                    co_occurrence_batch.append({
                        "c1":           tickers_list[i],
                        "c2":           tickers_list[j],
                        "article_id":   article_id,
                        "published_ts": published_ts,    # ← PathRAG 시간 필터용
                        "sentiment":    sentiment_score,
                    })

            # ── 테마 파싱 ────────────────────────────────────────────────────
            themes_str = row.get('themes', '')
            if isinstance(themes_str, str):
                unique_themes = set()
                for t in [t.strip() for t in themes_str.split(';') if t.strip()]:
                    # GDELT 포맷 (ThemeName,Offset)에서 테마명만 추출
                    theme_name_only = t.split(',')[0].strip()
                    if theme_name_only:
                        unique_themes.add(theme_name_only)
                
                # 중복 제거된 테마 중 최대 5개만 적재
                for t_name in list(unique_themes)[:5]:
                    themes_batch.append({"article_id": article_id, "theme_name": t_name})

            # 주기적 배치 플러시
            if len(articles_batch) >= batch_size or idx == len(df) - 1:
                self._execute_batch_write(
                    articles_batch, mentions_batch, themes_batch, co_occurrence_batch
                )
                print(f"    - [{idx + 1}/{len(df)}] 배치 적재 완료")
                articles_batch      = []
                mentions_batch      = []
                themes_batch        = []
                co_occurrence_batch = []

        print("\n[완료] Neo4j 지식 그래프(Knowledge Graph) 적재 완료")

    def _execute_batch_write(self, articles, mentions, themes, co_occurrences):
        """Neo4j 고속 배치 쓰기"""
        with self.driver.session(database=self.database) as session:

            # 1. Article 노드 (published_ts, published_month 포함)
            session.run("""
                UNWIND $batch AS row
                MERGE (a:Article {id: row.article_id})
                SET a.title           = row.title,
                    a.url             = row.url,
                    a.date            = row.date,
                    a.published_ts    = row.published_ts,
                    a.published_month = row.published_month,
                    a.source          = row.source,
                    a.sentiment_score = row.sentiment_score
            """, batch=articles)

            # 2. Company → Article 언급 관계
            session.run("""
                UNWIND $batch AS row
                MATCH (c:Company {ticker: row.ticker})
                MATCH (a:Article {id: row.article_id})
                MERGE (c)-[:MENTIONED_IN]->(a)
            """, batch=mentions)

            # 3. Theme 노드 및 Article 연결
            session.run("""
                UNWIND $batch AS row
                MERGE (t:Theme {name: row.theme_name})
                WITH t, row
                MATCH (a:Article {id: row.article_id})
                MERGE (a)-[:HAS_THEME]->(t)
            """, batch=themes)

            # 4. [PathRAG 핵심] CO_OCCURRED_WITH 에지
            #    동일 기업 쌍에 대해 MERGE → 이미 에지가 있으면 weight 누적, published_ts 갱신
            #    → PathRAG 경로 탐색 시: WHERE r.published_ts < $cutoff_ts 로 시간 격리 가능
            if co_occurrences:
                session.run("""
                    UNWIND $batch AS row
                    MATCH (c1:Company {ticker: row.c1})
                    MATCH (c2:Company {ticker: row.c2})
                    MERGE (c1)-[r:CO_OCCURRED_WITH]-(c2)
                    ON CREATE SET
                        r.weight       = 1,
                        r.published_ts = row.published_ts,
                        r.sentiment    = row.sentiment,
                        r.article_ids  = [row.article_id]
                    ON MATCH SET
                        r.weight       = r.weight + 1,
                        r.published_ts = CASE
                            WHEN row.published_ts > r.published_ts
                            THEN row.published_ts
                            ELSE r.published_ts
                        END,
                        r.sentiment    = (r.sentiment + row.sentiment) / 2.0
                """, batch=co_occurrences)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="GDELT Graph Loader for Neo4j (PathRAG Edition)")
    parser.add_argument("--csv-path",    type=str, default="data/gdelt_cleaned.csv")
    parser.add_argument("--neo4j-uri",   type=str, default=None)
    parser.add_argument("--neo4j-user",  type=str, default=None)
    parser.add_argument("--neo4j-pass",  type=str, default=None)
    parser.add_argument("--clear",       action="store_true", help="DB를 초기화한 뒤 재적재")
    args = parser.parse_args()

    uri      = args.neo4j_uri  or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = args.neo4j_user or os.getenv("NEO4J_USER",     "neo4j")
    password = args.neo4j_pass or os.getenv("NEO4J_PASSWORD", "neo4j")
    database = os.getenv("NEO4J_DATABASE", None)

    print(f"Neo4j 연결: {uri} (user: {user}, database: {database or 'default'})")
    try:
        loader = Neo4jGraphLoader(uri, user, password, database=database)
        if args.clear:
            loader.clear_database()
        loader.create_schema_constraints()
        loader.load_companies()
        loader.process_and_load_articles(args.csv_path)
        loader.close()
    except Exception as e:
        print(f"\n[오류] Neo4j 접속 또는 적재 실패: {e}")
