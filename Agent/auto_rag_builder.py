import os
import re
import argparse
import requests
import numpy as np
import pandas as pd
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from neo4j import GraphDatabase
from newspaper import Article, Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fetch article text, embed overlapping chunks, and populate Neo4j.
# Embedding requests use the configured HTTP service rather than a local model.
# Chunk MENTIONS edges link detected firm names to Company nodes.

# Share the firm-name mapping with GDELT preprocessing.
from TemporalRAG.engine.gdelt_collector import TARGET_FIRMS_MAPPING

# Map lowercase firm-name aliases to tickers.
_FIRM_PATTERNS = {}
for ticker, names in TARGET_FIRMS_MAPPING.items():
    for name in names:
        _FIRM_PATTERNS[name.lower()] = ticker

def _detect_firms_in_text(text: str) -> list:
    """Return firm tickers matched by word-boundary patterns in a text chunk."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    found = set()
    for name_lower, ticker in _FIRM_PATTERNS.items():
        # Match complete aliases rather than substrings within other words.
        if re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower):
            found.add(ticker)
    return list(found)


class AutomatedGraphRAGBuilder:
    """Build a Neo4j article/chunk index from GDELT URLs.
    
    Crawl article text, split it into overlapping chunks, request embeddings,
    and store publication timestamps and firm-mention relationships."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = None,
        embed_url: str = "http://localhost:8000/v1/embeddings",
        embed_model_name: str = "jinaai/jina-embeddings-v3",
        vector_dim: int = 1024,
    ):
        self.driver      = GraphDatabase.driver(uri, auth=(user, password))
        self.database    = database  # None selects the Neo4j default database.
        self.embed_url   = embed_url
        self.embed_model = embed_model_name
        self.vector_dim  = vector_dim

        print("=" * 60)
        print(" Auto RAG Builder — PathRAG Edition")
        print("=" * 60)
        print(f" vLLM 임베딩 엔드포인트: {embed_url}")
        print(f" 임베딩 모델          : {embed_model_name}")
        print(f" 벡터 차원            : {vector_dim}D")
        print("=" * 60)
        self._check_vllm_connection()

    def _check_vllm_connection(self):
        """Check the embedding service health and model endpoints."""
        try:
            # Try the health endpoint before querying the model list.
            base_url = self.embed_url.rsplit("/v1/", 1)[0]
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json().get("data", [])]
                print(f" [연결 성공] 서빙 모델: {models}")
            else:
                print(f" [경고] 서버 응답 이상 (status: {resp.status_code})")
        except Exception as e:
            print(f" [경고] 서버에 연결할 수 없습니다: {e} \n서버를 실행하세요.")
    def close(self):
        self.driver.close()


    # Embedding requests
    def _embed_via_vllm(self, texts: list) -> list:
        """Embed a batch of texts through the configured vLLM-compatible endpoint.
        
        Return vectors in input order, or a list of None values if the request fails."""
        if not texts:
            return []
        try:
            payload = {
                "model": self.embed_model,
                "input": texts,
                # Pass article text without adding an instruction prefix.
            }
            resp = requests.post(self.embed_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()["data"]
            # The API returns indexed embedding records.
            # Restore input order before extracting the vectors.
            data_sorted = sorted(data, key=lambda x: x["index"])
            embeddings  = [item["embedding"] for item in data_sorted]
            return embeddings
        except Exception as e:
            print(f"    [임베딩 오류] 호출 실패: {e}")
            return [None] * len(texts)


    # Neo4j indexes
    def create_neo4j_vector_index(self):
        """Create the vector index and the chunk publication-time index."""
        print("\n=== 1. Neo4j 네이티브 Vector Index 구성 ===")
        with self.driver.session(database=self.database) as session:
            session.run(f"""
                CREATE VECTOR INDEX `article_chunks` IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {self.vector_dim},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
            """)
            print(f"    [완료] 'article_chunks' 벡터 인덱스 ({self.vector_dim}D cosine) 생성")

            # Index publication timestamps used by temporal filters.
            session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.published_ts)")
            print("    [완료] Chunk.published_ts 인덱스 생성 — TBR Temporal Filter 준비 완료")

    # Article extraction and chunking
    @staticmethod
    def _scrape_article_body(url: str):
        """Extract article text using newspaper3k, with trafilatura as a fallback."""
        import time, random
        from newspaper import Article, Config
        import trafilatura

        # Stagger requests to reduce bursts from the same client.
        time.sleep(random.uniform(0.1, 1.0))

        # Rotate the configured User-Agent strings.
        USER_AGENTS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        ]
        ua = random.choice(USER_AGENTS)

        # Try newspaper3k first.
        try:
            config = Config()
            config.browser_user_agent = ua
            config.request_timeout = 15

            article = Article(url, config=config)
            article.download()
            article.parse()
            text = article.text
            if text and len(text) > 100:
                return text
        except Exception as e:
            err1 = str(e)

        # Fall back to trafilatura if the first extractor fails.
        try:
            settings = trafilatura.settings.use_config()
            settings.set("DEFAULT", "EXTRACTION_TIMEOUT", "15")
            downloaded = trafilatura.fetch_url(url, config=settings)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text and len(text) > 100:
                    return text
        except Exception as e:
            err2 = str(e)

        return None

    @staticmethod
    def _split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
        """Split text into overlapping character windows."""
        chunks = []
        if not text:
            return chunks
        start = 0
        while start < len(text):
            chunk = text[start: start + chunk_size].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # Batched index construction
    def process_and_build_rag(self, cleaned_csv_path: str, max_crawl_workers: int = 10, batch_size: int = 100):
        """Process article batches through crawling, chunking, embedding and storage.
        
        Previously attempted articles are skipped using the crawl_attempted flag.
        Completed writes persist; this is not an atomic transaction across all batches."""
        if not os.path.exists(cleaned_csv_path):
            print(f"[오류] 정제된 CSV 없음: {cleaned_csv_path}")
            return

        df = pd.read_csv(cleaned_csv_path)
        
        # Fetch article IDs already marked as crawl_attempted.
        print("\n=== [중복 검사] 크롤링 시도 완료된 기사 식별 ===")
        try:
            with self.driver.session(database=self.database) as session:
                res = session.run("""
                    MATCH (a:Article)
                    WHERE a.crawl_attempted = true
                    RETURN a.id AS article_id
                """)
                processed_ids = {str(row["article_id"]) for row in res}
            print(f"  → 이미 처리된 기사(crawl_attempted=true): {len(processed_ids)}건")
        except Exception as e:
            print(f"  [경고] 적재 확인 쿼리 실패: {e}")
            processed_ids = set()

        # Exclude previously attempted articles from this run.
        total_before = len(df)
        df = df[~df["article_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
        total_after = len(df)
        skipped = total_before - total_after
        if skipped > 0:
            print(f"  → 이미 적재 완료된 {skipped}건의 기사는 생략합니다. (남은 처리 대상: {total_after}건)")
        
        if len(df) == 0:
            print("  → [완료] 새로 적재할 기사가 없습니다!")
            return

        # Process the remaining articles in batches.
        for batch_start in range(0, total_after, batch_size):
            df_batch = df.iloc[batch_start:batch_start + batch_size].reset_index(drop=True)
            current_batch_len = len(df_batch)
            first_article_id = df_batch["article_id"].iloc[0] if current_batch_len > 0 else "N/A"
            last_article_id = df_batch["article_id"].iloc[-1] if current_batch_len > 0 else "N/A"
            print(f"\n=== [배치 {batch_start // batch_size + 1} / {total_after // batch_size + 1}] {batch_start + 1} ~ {batch_start + current_batch_len} / {total_after} ===")
            print(f"  → start id: {first_article_id} ~ last id: {last_article_id}")

            # Fetch article bodies concurrently within this batch.
            url_to_body = {}
            print(f"  → 크롤러 구동 (스레드: {max_crawl_workers})")
            with ThreadPoolExecutor(max_workers=max_crawl_workers) as executor:
                futures = {
                    executor.submit(self._scrape_article_body, row["url"]): row["url"]
                    for _, row in df_batch.iterrows()
                }
                for future in as_completed(futures):
                    url  = futures[future]
                    body = future.result()
                    if body:
                        url_to_body[url] = body
            print(f"    크롤링 완료: {current_batch_len}건 시도 | 성공: {len(url_to_body)}건 | 실패: {current_batch_len - len(url_to_body)}건")

            # Mark every attempted article, including failed crawls.
            attempted_batch = [
                {"id": str(row["article_id"]), "success": row["url"] in url_to_body}
                for _, row in df_batch.iterrows()
            ]
            with self.driver.session(database=self.database) as mark_session:
                mark_session.run("""
                    UNWIND $batch AS row
                    MERGE (a:Article {id: row.id})
                    SET a.crawl_attempted = true,
                        a.crawl_success   = row.success
                """, batch=attempted_batch)

            # Chunk, embed and store the current batch.
            print("  → 청킹 → 임베딩 → Neo4j 적재 진행")

            # Create Company nodes from the shared firm mapping.
            with self.driver.session(database=self.database) as session:
                for ticker, names in TARGET_FIRMS_MAPPING.items():
                    primary_name = names[0] if names else ticker
                    session.run(
                        "MERGE (c:Company {ticker: $ticker}) ON CREATE SET c.name = $name",
                        ticker=ticker, name=primary_name
                    )

            # Create Article nodes for the batch.
            from datetime import datetime, timezone as tz
            articles_batch = []
            for _, row in df_batch.iterrows():
                article_id = str(row["article_id"])
                date_str   = str(row.get("date", ""))
                pub_ts     = None
                pub_month  = None
                try:
                    date_str = str(row.get("date", "")).strip()
                    if "-" in date_str:
                        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=tz.utc)
                    else:
                        dt = datetime.strptime(date_str[:8], "%Y%m%d").replace(tzinfo=tz.utc)
                    pub_ts    = int(dt.timestamp())
                    pub_month = dt.strftime("%Y-%m")
                except Exception:
                    pass
                articles_batch.append({
                    "id":              article_id,
                    "title":           str(row.get("title", "")),
                    "url":             str(row.get("url", "")),
                    "source":          str(row.get("source", "")),
                    "published_ts":    pub_ts,
                    "published_month": pub_month,
                })
            with self.driver.session(database=self.database) as session:
                session.run("""
                    UNWIND $batch AS row
                    MERGE (a:Article {id: row.id})
                    SET a.title           = row.title,
                        a.url             = row.url,
                        a.source          = row.source,
                        a.published_ts    = row.published_ts,
                        a.published_month = row.published_month
                """, batch=articles_batch)

            # Create embedded Chunk nodes for each retrieved article.
            with self.driver.session(database=self.database) as session:
                for idx, row in df_batch.iterrows():
                    url        = row["url"]
                    article_id = str(row["article_id"])
                    title      = str(row["title"])

                    date_str  = str(row.get("date", ""))
                    pub_ts    = None
                    pub_month = None
                    try:
                        date_str = str(row.get("date", "")).strip()
                        if "-" in date_str:
                            dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=tz.utc)
                        else:
                            dt = datetime.strptime(date_str[:8], "%Y%m%d").replace(tzinfo=tz.utc)
                        pub_ts    = int(dt.timestamp())
                        pub_month = dt.strftime("%Y-%m")
                    except Exception:
                        pass

                    body_text = url_to_body.get(url)
                    if not body_text:
                        # Skip missing article bodies rather than embedding metadata as article text.
                        continue

                    chunks = self._split_text_into_chunks(body_text)
                    if not chunks:
                        continue

                    embeddings = self._embed_via_vllm(chunks)

                    chunks_batch   = []
                    mentions_batch = []

                    for c_idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                        if emb is None:
                            continue
                        chunk_id = f"{article_id}_ch{c_idx}"
                        chunks_batch.append({
                            "chunk_id":        chunk_id,
                            "text":            chunk_text,
                            "embedding":       emb,
                            "published_ts":    pub_ts,
                            "published_month": pub_month,
                        })
                        detected_tickers = _detect_firms_in_text(chunk_text)
                        for ticker in detected_tickers:
                            mentions_batch.append({"chunk_id": chunk_id, "ticker": ticker})

                    if not chunks_batch:
                        continue

                    # Link each Chunk to its parent Article through HAS_CHUNK.
                    session.run("""
                        UNWIND $batch AS row
                        MERGE (ch:Chunk {id: row.chunk_id})
                        SET ch.text             = row.text,
                            ch.embedding        = row.embedding,
                            ch.published_ts     = row.published_ts,
                            ch.published_month  = row.published_month
                        WITH ch, row
                        MATCH (a:Article {id: $article_id})
                        MERGE (a)-[:HAS_CHUNK]->(ch)
                    """, batch=chunks_batch, article_id=article_id)

                    # Link detected company mentions through MENTIONS.
                    if mentions_batch:
                        session.run("""
                            UNWIND $batch AS row
                            MATCH (ch:Chunk {id: row.chunk_id})
                            MATCH (c:Company {ticker: row.ticker})
                            MERGE (ch)-[:MENTIONS]->(c)
                        """, batch=mentions_batch)

                    local_idx = idx + 1  # Batch rows have a zero-based index after reset_index.
                    if local_idx % 25 == 0 or local_idx == current_batch_len:
                        print(
                            f"    [{local_idx}/{current_batch_len}] "
                            f"Chunk {len(chunks_batch)}개 적재 "
                            f"(Chunk->Company: {len(mentions_batch)}개)"
                        )

        print("\n[완료] RAG 인덱스 빌드 완료!")



if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG Builder")
    parser.add_argument("--csv-path", type=str, default="data/gdelt_cleaned.csv")
    parser.add_argument("--embed-url", type=str, default=None, help="임베딩 엔드포인트")
    parser.add_argument("--embed-model", type=str, default=None, help="임베딩 모델 이름")
    parser.add_argument("--vector-dim", type=int, default=1024)
    parser.add_argument("--neo4j-uri", type=str, default=None)
    parser.add_argument("--neo4j-user", type=str, default=None)
    parser.add_argument("--neo4j-pass", type=str, default=None)
    parser.add_argument("--workers", type=int, default=10, help="동시 크롤링 스레드 수(IP 차단 위험)")
    parser.add_argument("--batch-size", type=int, default=100, help="크롤링 및 적재 배치 크기")
    args = parser.parse_args()

    embed_url = args.embed_url or os.getenv("EMBED_URL", "http://localhost:8000/v1/embeddings")
    embed_model = args.embed_model or os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v3")
    uri      = args.neo4j_uri  or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user     = args.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = args.neo4j_pass or os.getenv("NEO4J_PASSWORD", "neo4j")

    database = os.getenv("NEO4J_DATABASE", None)

    try:
        builder = AutomatedGraphRAGBuilder(
            uri=uri, user=user, password=password,
            database=database,
            embed_url=embed_url, embed_model_name=embed_model,
            vector_dim=args.vector_dim,
        )
        builder.create_neo4j_vector_index()
        builder.process_and_build_rag(args.csv_path, max_crawl_workers=args.workers, batch_size=args.batch_size)
        builder.close()
    except Exception as e:
        print(f"\n[오류] RAG 빌드 실패: {e}")
