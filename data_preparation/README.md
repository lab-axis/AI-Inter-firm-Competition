# data_preparation — Fundamental Panel & Static Graph Builder

A 5-stage pipeline that assembles a monthly, model-ready panel for 30 AI/IT companies
(2014–2025) and exports it, together with a static company graph, into the tensor format
consumed by the B-MTGNN forecasting model.

The panel's target column is **AFCI** (AI-Firm Composite Index), built from SEC-filing
sector-exposure scores plus market, news, and search-attention features.

---

## Pipeline stages

Run the scripts in numeric order. Each stage's output feeds the next.

| # | Script | Input | Output | External source |
|---|---|---|---|---|
| 1 | `1.fetch_sec_filings.py` | CIK map (30 firms) | `data/1_transcripts/<TICKER>/*.txt` | SEC (`sec-api`, `SEC_API_KEY`); 20-F via direct HTML |
| 2 | `2.topic_modeling.py` | `data/1_transcripts/` | `data/2_topic/topic_vectors.csv` | BGE embeddings `BAAI/bge-large-en-v1.5` (local) |
| 3 | `3.build_dataset.py` | `data/2_topic/topic_vectors.csv` | `data/3_dataset/data_set.csv` (+ `temp/*.csv`) | yfinance, Google Trends (`pytrends`), EODHD news (`EODHD_API_KEY`), OpenAlex, OECD |
| 4 | `4.create_static_graph.py` | `data/3_dataset/data_set.csv` | `data/4_graph/edges_static_prior.csv` | — (Pearson + NLP cosine) |
| 5 | `5.export_bmtgnn.py` | panel + edges | `data/5_bmtgnn_input/bmtgnn_data.npy`, `adj_mat.csv`, `node_ids.csv` | — |

**Stage details**

- **1 — SEC filings.** Collects 10-K / 10-Q / 20-F (6-K excluded) for 30 firms over 2014–2025
  using official CIKs. Foreign private issuers (e.g. ASML, TSM) whose 20-F the Extractor API
  does not support are downloaded as raw HTML and cleaned with BeautifulSoup.
- **2 — Topic modeling.** Computes exposure to 9 AI sectors (semiconductors, cloud, software,
  hardware, advertising, social, ecommerce, data, auto) from the filings using BGE embeddings,
  on a quarterly grid (2013-09 → 2025-12) with quarterly forward-fill.
- **3 — Panel build.** Merges topic exposures with monthly stock returns, news counts /
  sentiment / AI-exposure / impact, paper counts, and Google-Trends attention into one panel.
- **4 — Static graph (visualization grouping only; not a model/training input).** Builds a
  company adjacency from monthly return correlation (Pearson) and NLP cosine similarity. The
  trained B-MTGNN **learns its graph adaptively** (`buildA_true=True`; the predefined-graph
  blending is disabled in `net.py`), so this static prior is **not** a model input and is **not
  a claimed component in the paper**. Its only consumer is `B-MTGNN/forecast.py`, which uses it
  solely to decide *which* focal-node + neighbour forecasts are drawn together — using **edge
  presence (`weight > 0`) only**, never the weight magnitudes, and never the forecast values
  themselves. It therefore introduces **no leakage** into the forecast outputs.
- **5 — Export.** Reshapes the panel into a `[nodes, timesteps, features]` tensor for B-MTGNN.
  `adj_mat.csv` is also written from the stage-4 static prior; it is consumed **only** by
  `forecast.py` for plot grouping (see stage 4), not by model training.

---

## Prerequisites

- Python 3.11+ (Recommend 3.13)
- `pip install -r data_preparation/requirements.txt`

---

## Usage

Run from **inside the `data_preparation/` directory** (the default paths resolve relative to it):

```bash
cd data_preparation

python 1.fetch_sec_filings.py  --start-year 2014 --end-year 2025
python 2.topic_modeling.py     --model BAAI/bge-large-en-v1.5
python 3.build_dataset.py      --start-month 2014-01 --end-month 2025-12
python 4.create_static_graph.py
python 5.export_bmtgnn.py
```

---

## Panel schema (`data/3_dataset/data_set.csv`)

```
date, firm_id, stock_return,
paper_count, news_count, news_sentiment, news_ai_exposure, news_impact,
gt_firm_attention, gt_ai_momentum,
expo_topic_semiconductors, expo_topic_cloud, expo_topic_software, expo_topic_hardware,
expo_topic_advertising, expo_topic_social, expo_topic_ecommerce, expo_topic_data, expo_topic_auto,
AFCI
```

---
