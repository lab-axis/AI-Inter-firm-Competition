# AI Inter-firm Competition

Research execution path: **data preparation → B-MTGNN forecasting → evidence-grounded multi-agent interpretation**.

This directory contains one runtime implementation. Development patches, old source versions, experiment campaigns and bulky historical outputs are maintained separately, not as alternative entry points here. This is a runtime package, **not yet a complete archive reproducing every paper table**.

## Structure

```text
pipeline.py                     # one command entry point
data_preparation/
  1.fetch_sec_filings.py         # SEC API collection (credentials required)
  2.topic_modeling.py            # topic exposures from archived filings
  3.build_dataset.py             # monthly panel / AFCI construction
  4.export_bmtgnn.py             # tensor + firm IDs; no external graph
  data/                         # local inputs/intermediates
B-MTGNN/
  config.py, split_policy.py    # configuration and target-disjoint partitioning
  net.py, layer.py, util.py     # model and loader
  train_test.py, forecast.py    # training/evaluation and future forecasts
  run_profile.py               # per-checkpoint architecture and scaling metadata
  Bayesian/{6,12,24,36}mo/      # supplied checkpoints + HP + provenance
Agent/
  auto_rag_builder.py           # article extraction, embeddings, Neo4j loading
  TemporalRAG/engine/           # canonical GDELT helper and retriever
  Multi_agent_Debate/           # main, graph, nodes, prompts, state
  data/                        # local GDELT source/cleaned CSVs
```

## Installation and local check

Use a dedicated Python environment; install a PyTorch build appropriate for your hardware, then:

```sh
python -m pip install -r requirements.txt
python pipeline.py check
```

For data collection and agent/RAG execution also install `requirements-pipeline.txt`. These are dependency lists, not a validated cross-platform lockfile. The local forecast smoke test used Python 3.13, PyTorch 2.13.0+cu132 (CPU execution), NumPy 2.2.6 and pandas 2.3.1. GPU, fresh installation and live agent services were not validated in that smoke test.

Create a local `.env` from `.env.example` only if one does not already exist. Never commit credentials. Existing `.env` is preserved locally. The default chat model identifier is `casperhansen/deepseek-r1-distill-qwen-7b-awq`; the embedding builder defaults to `jinaai/jina-embeddings-v3`. Host compatible services separately and set endpoints in `.env`. Client requirements do not install or start vLLM/Neo4j.

## 1. Data preparation

The supplied tensor covers **30 firms × 144 months (2014-01–2025-12) × 17 features**. Feature order is defined in `4.export_bmtgnn.py`; `AFCI` is channel 0. Firm order is in `node_ids.csv`. The loader assumes a January 2014 start; other periods require adapting the date contract, not simply replacing the tensor.

If the prepared tensor is present, skip collection and proceed to forecasting. To rebuild from the retained monthly panel:

```sh
python pipeline.py data export
```

Only if new collection/reconstruction is intended, run these stages explicitly and in order:

```sh
python pipeline.py data collect
python pipeline.py data topics
python pipeline.py data build
python pipeline.py data export
```

`collect` requires an active SEC API subscription. `topics` may download an embedding model. `build` contacts external financial/news/attention services, including EODHD. Collection cannot be promised to reproduce an older archive byte-for-byte. Existing SEC archives are retained, so an expired subscription does not prevent use of the prepared tensor. Stage-specific arguments follow `--`, e.g. `python pipeline.py data export -- --panel PATH --out-dir OUTPUT` (relative paths are interpreted from the data-preparation directory).

**External adjacency is disabled.** The exporter writes only `bmtgnn_data.npy` and `node_ids.csv`. No `adj_mat.csv`, prior matrix or `4_graph` stage is needed. The model uses its trained graph-constructor embeddings (`buildA_true=True`); its internal adjacency operations remain necessary. This cleanup concerns the external graph input, not an audit of all feature availability dates.

## 2. Forecast or train

```sh
python pipeline.py forecast --num-runs 50 --device cpu
python pipeline.py forecast --horizon 6 --num-runs 50 --device cpu
python pipeline.py forecast --horizon 24 --num-runs 50 --device cpu
python pipeline.py forecast --horizon 36 --num-runs 50 --device cpu
```

This loads `Bayesian/12mo/o_model.safetensors` and writes `forecast/data/<FIRM>.txt` plus `forecast/gap/<FIRM>_gap.csv` under that run. No figures are rendered by default. For a distinct output directory use `--output-dir PATH`; rerunning the same destination replaces its forecast files.

The default is `12mo`; `--run 24mo` is equivalent to `--horizon 24`. Each run's `provenance.json` supplies the checkpoint-specific scaling and partition settings, which take precedence over generic forecast configuration. Existing weights and user-added results are preserved. Use a separate `--output-dir` if you want to preserve an earlier forecast in the selected run.

| Run | Forecast months | Compatibility profile |
|---|---|---|
| `6mo` | 2026-01–2026-06 | Historical `legacy` |
| `12mo` | 2026-01–2026-12 | `withheld_matched`, source `w12_cal` |
| `24mo` | 2026-01–2027-12 | Historical `legacy` |
| `36mo` | 2026-01–2028-12 | Historical `legacy` |

The added 6/24/36-month checkpoints match the archived original files. Their compatibility profiles use the archived pre-split loader's normalization boundary (`n-14`); the training split itself is not embedded in those checkpoints. Historical target overlap is reported, not treated as a new target-disjoint evaluation. Do not compare these runs as if they all used the corrected 12-month protocol.

The original `6mo/hp.txt` differs from five architectural fields in its checkpoint and has not been overwritten. Forecasting restores the checkpoint architecture directly. Its separate `training_preset_hps` in `provenance.json` comes from the unique architecture-matched configuration in the supplied search CSV, without inferring a seed or epoch identity. Other presets agree with their checkpoint architectural fields.

Each supplied model is a single stored checkpoint, not an ensemble. New Monte Carlo forecasts are **not** the archived qualitative-case outputs, and raw intervals here are **not** post-hoc calibrated intervals from the paper. Two MC draws are sufficient for a plumbing smoke test, not for scientific interpretation.

TXT files contain raw-scale AFCI predictions and uncertainty. Gap CSVs retain the original global-normalisation/exponential-smoothing calculation, now for **all ordered firm pairs** instead of pairs selected by the removed static graph. They are descriptive differences, not learned graph weights or evidence of statistically significant differences.

Optional fresh training using the selected horizon's recorded hyperparameter preset:

```sh
python pipeline.py train --run my_12mo --seed 2000 --repeats 1 --device cuda:0
python pipeline.py train --horizon 24 --run my_24mo --seed 2000 --repeats 1 --device cuda:0
python pipeline.py forecast --run my_12mo --num-runs 50
```

The wrapper refuses an existing training-run name. Five repeats reproduce the seed schedule (2000–2004), not a guaranteed identical metric across environments. Training can generate diagnostic plots and overwrite checkpoints within its new run. The original implementation's component switches remain in the model code; historical ablation campaigns are not included here.

New training defaults to `withheld_matched` with validation span `max(24, horizon)`. The wrapper checks feasibility before creating a run. At 36 months, this default leaves no training origin in the retained 144-month panel and is rejected; the existing 36-month checkpoint can still be forecast normally. A different training protocol must be chosen explicitly via `--split-policy`/`--valid-span` or a longer panel prepared. `legacy` must never be silently substituted for a disjoint split. No fresh training was performed when adding multi-horizon support.

## 3. Build evidence and run agents

Neo4j must contain the article/chunk graph and vector index built with matching embeddings. To prepare it:

```sh
python pipeline.py rag sql
# Execute the generated data/gdelt_extraction_query.sql in BigQuery yourself.
# Save its result as Agent/data/gdelt_orig.csv.
python pipeline.py rag preprocess
python pipeline.py rag build
```

`rag sql` only generates SQL; it does not download GDELT. `rag build` crawls third-party articles, calls the embedding endpoint and writes to Neo4j. Skip those steps if the compatible evidence database already exists. On Windows, `Agent/neo4j-start.bat` uses your externally configured `NEO4J_HOME`, not an author-specific installation path.

With chat/embedding servers and Neo4j ready:

```sh
python pipeline.py agents --companies MSFT AAPL --months 2026-01
python pipeline.py agents --horizon 24 --companies MSFT --months 2027-01
python pipeline.py agents --horizon 36 --companies NVDA --months 2028-12
```

For a custom forecast run use `--run my_12mo`, or supply `--forecast-dir PATH` for an explicit output directory. Generate the selected forecast before running agents. Dates are checked against the actual forecast-array length; for example, `6mo` rejects 2026-07 and `36mo` accepts through 2028-12. Agent logs are written under `Agent/Multi_agent_Debate/logs/`. The existing RAG cutoff is the selected interpretation month; future target months do not imply that future news exists, and this is not automatically a prospective, forecast-origin-frozen evaluation.

## Publication boundary

The local source archives remain available but are excluded from Git until redistribution rights are checked. `.gitignore` also excludes credentials, IDE files, databases, logs and generated runs. Review the prepared data's redistribution terms before publishing it too. No license grant, GitHub upload or DOI reservation is implied by this cleanup. Historical experiment evidence must be packaged separately for paper-level reproducibility; do not discard the recovery archive.

Use `python pipeline.py --dry-run <command> ...` to inspect execution without starting it. `check` lists missing packages and files, but does not certify live service readiness.

## Offline smoke test

On Windows, run `RUN_SMOKE.bat` from Explorer or a terminal. It uses the local
`.venv` if present, otherwise the Windows `py -3` launcher (or `python` on PATH
if the launcher is absent). To select an environment explicitly:

```bat
RUN_SMOKE.bat "C:\path\to\python.exe"
```

The batch defaults to all four horizons. To check one, use `RUN_SMOKE.bat "C:\path\to\python.exe" 24mo`.

Alternatively, run `python smoke/scripts/smoke_test.py --all` or `python smoke/scripts/smoke_test.py --run 24mo` on any supported platform.
The test checks source syntax, forecast dependencies, the supplied input and
checkpoint hashes, run-specific split metadata, two CPU Monte Carlo forward passes per horizon,
and the pure agent input-formatting helpers. Missing dependencies for other
pipeline stages are reported separately as warnings. No packages are installed.

Results use the existing fixed `smoke/` output convention (`--output PATH` overrides it). An all-horizon run writes `smoke/summary.txt` plus separate `smoke/6mo/`, `12mo/`, `24mo/` and `36mo/` reports, logs and temporary forecasts. Rerunning replaces smoke reports only, not the named runs' forecast folders, inputs or weights. Protected-file hashes are checked before and after execution. Smoke outputs are excluded from Git.

Historical profiles emit overlap warnings. PASS concerns execution and input contracts; it does not certify that historical splits are target-disjoint.

`OFFLINE SMOKE: PASS` means these offline checks passed. It does **not** validate
training, collection APIs, live Neo4j/embedding/chat services, a full agent debate
or scientific accuracy. No figures are rendered. Two Monte Carlo draws are only
for a plumbing check; do not use these outputs as paper results. CPU inference has
a 180-second timeout. The batch file pauses before closing; set `SMOKE_NO_PAUSE=1`
for unattended execution. A failed test returns a nonzero exit code.
