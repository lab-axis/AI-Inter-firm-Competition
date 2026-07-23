# TemporalRAG Agent Pipeline (GDELT → Neo4j → Temporally-Bounded RAG)

A pipeline that collects and preprocesses GDELT GKG news data for 30 AI/IT companies,
builds a Neo4j knowledge graph and a Temporally-Bounded (TBR) RAG layer on top of it,
and generates question-answering / analytical reports.

---

## 1. Architecture — Two-Layer Ingestion Design

Each `:Article` node is completed **in stages by two scripts with separated
responsibilities**. This is an intentional design; if the run order below is respected,
every article ends up with a complete schema.

| Layer | Script | Creates / sets |
|---|---|---|
| **① Graph layer** | `neo4j_loader.py` | `:Article` (`date`, `published_ts`, `published_month`, `title`, `url`, `source`, `sentiment_score`), `(:Company)-[:MENTIONED_IN]->(:Article)`, `(:Article)-[:HAS_THEME]->(:Theme)`, `(:Company)-[:CO_OCCURRED_WITH]->(:Company)` |
| **② RAG layer** | `auto_rag_builder.py` | Body crawl → chunking → embedding → `:Chunk` (`embedding`, `published_ts`), `(:Article)-[:HAS_CHUNK]->(:Chunk)`, `(:Chunk)-[:MENTIONS]->(:Company)`, Neo4j vector index |

### Run order (important)

```
gdelt_collector.py (collect/preprocess)  →  neo4j_loader.py (graph)  →  auto_rag_builder.py (RAG)
```

- **Always run `neo4j_loader.py` first, then `auto_rag_builder.py`.**
  The graph layer is the single source of truth for the article's temporal properties
  (`date`, etc.) and company relationships (`MENTIONED_IN`).
- If `auto_rag_builder.py` is run standalone first, the affected articles may have empty
  `date` / `MENTIONED_IN`. This is a **known constraint** of the design in which the RAG
  layer only augments body text and embeddings; it does not occur when the order above is kept.

### Temporal-filter convention

- Temporal constraints (TBR) across all layers are unified on **`published_ts`
  (integer Unix epoch)** — e.g. `WHERE a.published_ts < $cutoff_ts`,
  `WHERE ch.published_ts < $cutoff_ts`.
- `published_month` is a **human-readable convenience field** and is **not** used by any
  retrieval / filtering logic.

---

## 2. Prerequisites

| Component | Version / spec | Notes |
|---|---|---|
| Python | 3.11+ | `pandas`, `neo4j`, `newspaper3k`, `python-dotenv`, `requests` |
| Neo4j | 2026.05.x (Community) | Version with vector-index support, `bolt://localhost:7687` |
| Embedding server | vLLM `/v1/embeddings`, `jinaai/jina-embeddings-v3` (1024D) | Default `http://localhost:8000` |
| LLM server | vLLM `/v1/chat/completions` | Default `http://localhost:8001` (for the agent) |

> Embedding / LLM endpoints can be swapped for any OpenAI-compatible API via
> `--embed-url`, `--vllm-chat-url`, etc.

---

## 3. Installation

```bash
# From the repository root
cp .env.example .env          # fill in the values (API keys, Neo4j password, etc.)
pip install -r Agent/requirements.txt
```

Key `.env` variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `EMBED_URL`, `EMBED_MODEL`,
and data-collection keys such as `HF_TOKEN` / `SEC_API_KEY` / `EODHD_API_KEY` / `FINNHUB_API_KEY`.

---

## 4. Reproduction Steps

> Run the commands below **from the repository root**, with explicit paths.
> Large CSVs (`Agent/data/*.csv`) and `.env` are excluded from the repository via `.gitignore`.

### 4-1. Generate GDELT collection SQL

```bash
python Agent/gdelt_collector.py --mode sql --split-years
# → generates Agent/data/yearly_queries/gdelt_query_<year>.sql
```

Fixed methodology: collect only from the GKG 2.0 era (**from 2015-03-01**), a vetted
**whitelist of 13 outlets**, and **oversampling** to survive link rot / 404s
(`--target-per-month` × `--oversample-rate`).

Run each SQL in the Google BigQuery console, then merge all result CSVs into
`Agent/data/gdelt_orig.csv`.

### 4-2. Preprocess (cleaning)

```bash
python Agent/gdelt_collector.py --mode preprocess \
  --raw-csv Agent/data/gdelt_orig.csv \
  --out-csv Agent/data/gdelt_cleaned.csv
```

### 4-3. Load the Neo4j graph (① graph layer)

```bash
# (optional) rebuild from scratch: --clear
python Agent/neo4j_loader.py --csv-path Agent/data/gdelt_cleaned.csv
```

### 4-4. Build the RAG layer (② RAG layer — must run after 4-3)

```bash
python Agent/auto_rag_builder.py --csv-path Agent/data/gdelt_cleaned.csv
```

### 4-5. Run an agent query

```bash
python Agent/temporal_rag_agent.py \
  --query "Supply-chain collaboration trends between NVIDIA and TSMC" \
  --cutoff 2025-01 \
  --firms NVDA,TSM \
  --top-k 5
```

---

## 5. Node / Relationship Schema Reference

```
(:Company {ticker, name})
(:Article {id, date, published_ts, published_month, title, url, source, sentiment_score})
(:Theme   {name})
(:Chunk   {id, text, embedding, published_ts})

(:Company)-[:MENTIONED_IN]->(:Article)
(:Article)-[:HAS_THEME]->(:Theme)
(:Company)-[:CO_OCCURRED_WITH {weight, published_ts, sentiment}]-(:Company)
(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Company)
```

Verification queries:

```cypher
MATCH (a:Article) RETURN count(a), min(a.date), max(a.date);
MATCH (a:Article) WHERE a.date IS NULL RETURN count(a);   // 0 if 4-3 was run before 4-4
```

---

## 6. Component Map

| File / directory | Role |
|---|---|
| `gdelt_collector.py` | Generates GDELT BigQuery SQL and preprocesses collected CSVs (single source of the collection methodology) |
| `neo4j_loader.py` | Cleaned CSV → Neo4j knowledge graph (① graph layer) |
| `auto_rag_builder.py` | Body crawl / chunking / embedding → vector index (② RAG layer) |
| `temporal_rag_retriever.py` | PathRAG + temporally-bounded vector search (TBR) retriever |
| `temporal_rag_agent.py` | Retriever + LLM-based question-answering agent |
| `Multi_agent_Debate/` | Multi-agent debate-based analysis module |
| `TemporalRAG/` | FastAPI backend + UI (service layer) |
| `data/` | Collected / cleaned datasets (excluded from the repository) |
