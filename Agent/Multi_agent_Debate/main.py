import os
import sys
import asyncio
import datetime
import time

# Suppress warnings and set logging level for cleaner output
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.ERROR)


# Add parent directory to path to allow importing TemporalRAG retriever
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dotenv import load_dotenv
from TemporalRAG.engine.temporal_rag_retriever import TemporalPathRAGRetriever
from Multi_agent_Debate.graph import create_debate_graph

load_dotenv()

# Persona-Specific RAG query templates per agent.
# Design Principles:
#   1. Company-anchored: Every query leads with the company ticker/name to maximize
#      precision. This prevents unrelated companies (e.g., Meta) from appearing in
#      company-specific regulatory or financial contexts.
#   2. Persona-differentiated: Each query targets a distinct information domain so
#      agents receive non-overlapping, role-relevant evidence.
#   3. Specific signals: Queries name exact metrics, frameworks, and event types to
#      surface high-quality chunks over generic AI industry noise.
AGENT_RAG_QUERIES = {
    "qa": (
        "{company} AI firm competitiveness quantitative context: {company} AI-related news "
        "visibility, {company} cloud and AI product traction, {company} operating margin, "
        "free cash flow, AI investment sustainability, and strategic investment relevance "
        "for {company} up to {cutoff}."
    ),
    "fa": (
        "{company} AI product-market technological research evidence: {company} product "
        "launches, {company} AI feature releases such as Copilot or Azure OpenAI, {company} "
        "foundation model or AI platform releases, {company} research publications, {company} "
        "AI papers at NeurIPS ICML ACL or related venues, {company} R&D investments, "
        "strategic partnerships, and enterprise AI adoption up to {cutoff}."
    ),
    "ms": (
        "{company} strategic investment and market environment: AI hyperscaler CapEx cycles "
        "affecting {company}, Federal Reserve interest rate decisions and tech spending, GPU "
        "and semiconductor supply chain constraints affecting {company}, global IT spending, "
        "developer ecosystem trends, and AI market adoption up to {cutoff}."
    ),
    "ba": (
        "{company} specific downside risks and negative evidence: {company} AI project failures "
        "or delays, {company} market share losses, {company} competitive threats from OpenAI "
        "AWS Google Cloud or Anthropic, {company} AI product adoption weakness, analyst "
        "downgrades of {company}, and {company} execution risk up to {cutoff}."
    ),
    "rc": (
        "{company} regulatory IP and legal risks: antitrust investigations against {company}, "
        "{company} GDPR or EU AI Act compliance issues, {company} cloud antitrust lawsuits, "
        "{company} copyright or data privacy litigation, {company} patent filings, {company} "
        "IP disputes, {company} licensing risk, government scrutiny of {company} AI products, "
        "and export control impact on {company} up to {cutoff}."
    ),
    "ar": (
        "Academic and methodological critique of AI firm competitiveness forecasting and "
        "time-series prediction models: overfitting in graph neural networks, distribution "
        "shift in AFCI or latent-index forecasting, confidence interval reliability in "
        "B-MTGNN or GNN models, Diebold-Mariano test benchmarking, and stationarity concerns "
        "in AI competitiveness index prediction up to {cutoff}."
    ),
}

async def get_temporal_rag_context(company: str, cutoff: str, query: str = None):
    """Fetch TemporalRAG context for a given company, cutoff date, and optional custom query."""
    print(f"  Loading TemporalRAG context for {company} at cutoff {cutoff}...")
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        vllm_embed_url = os.getenv("VLLM_EMBED_URL", "http://localhost:8000/v1/embeddings")
        
        retriever = TemporalPathRAGRetriever(
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password,
            vllm_embed_url=vllm_embed_url,
        )
        
        # Use custom query if provided, otherwise fallback to generic
        if query is None:
            query = f"Provide the most recent AI competitiveness news and events for {company} leading up to {cutoff}. Focus on {company}'s own recent events."
        
        result = retriever.retrieve(
            query=query,
            cutoff_date=cutoff,
            top_k=10,
            target_firms=[company]
        )
        retriever.close()
        
        # Just return the raw lists, we will format them globally later
        paths = result.get("paths", [])
        chunks = result.get("chunks", [])
        return paths, chunks
    except Exception as e:
        print(f"Warning: Could not fetch TemporalRAG context: {e}")
        return [], []

async def fetch_all_agent_rag_contexts(company: str, cutoff: str) -> tuple:
    """Sequentially fetch persona-specific RAG contexts and build a global citation registry."""
    print(f"\n[Agent-Personalized RAG] Fetching 6 persona-specific contexts for {company} at {cutoff}...")
    
    # Store raw results per agent
    agent_raw_results = {}
    
    # Global registries to ensure consistent numbering across all agents
    global_chunks_list = []
    global_chunks_map = {}  # URL or text -> global index (1-based)
    
    global_paths_list = []
    global_paths_map = {}   # path_str -> global index (1-based)
    
    for agent_key, query_template in AGENT_RAG_QUERIES.items():
        query = query_template.format(company=company, cutoff=cutoff)
        print(f"  -> [{agent_key.upper()} Agent] Query: {query[:80]}...")
        paths, chunks = await get_temporal_rag_context(company, cutoff, query)
        agent_raw_results[agent_key] = {"paths": paths, "chunks": chunks}
        
        # Register chunks globally
        for c in chunks:
            identifier = c.get("article_url") or c.get("text")[:100]
            if identifier not in global_chunks_map:
                global_chunks_list.append(c)
                global_chunks_map[identifier] = len(global_chunks_list) # 1-based index
                
        # Register paths globally
        for p in paths:
            identifier = p.get("path_str")
            if identifier not in global_paths_map:
                global_paths_list.append(p)
                global_paths_map[identifier] = len(global_paths_list)
                
    # Now format the context strings for each agent using the global indices
    agent_contexts = {}
    agent_refs_dict = {}
    from urllib.parse import urlparse
    
    for agent_key, results in agent_raw_results.items():
        context_lines = []
        ref_lines = []
        seen_ids = set()  # Track per-agent to deduplicate ref entries at source
        
        for p in results["paths"]:
            global_id = global_paths_map[p.get("path_str")]
            path_key = f"Path {global_id}"
            context_lines.append(f"[Path {global_id}]: {p['path_str']}")
            if path_key not in seen_ids:
                seen_ids.add(path_key)
                ref_lines.append(f"- Path {global_id}: {p['path_str']}")
            
        for c in results["chunks"]:
            identifier = c.get("article_url") or c.get("text")[:100]
            global_id = global_chunks_map[identifier]
            doc_key = f"Doc {global_id}"
            context_lines.append(f"[Doc {global_id}] ({c.get('published_date')}): {c.get('text', '')[:500]}")
            
            if doc_key not in seen_ids:
                seen_ids.add(doc_key)
                title = c.get('article_title', 'Unknown Title').strip()
                url = c.get('article_url', '')
                domain = urlparse(url).netloc if url else 'Unknown Journal'
                domain = domain.replace("www.", "")
                ref_lines.append(f"- Doc {global_id}: {title}, {domain} ({url})")
            
        agent_contexts[agent_key] = "\n".join(context_lines)
        agent_refs_dict[agent_key] = "\n".join(ref_lines)
        
    print(f"[Agent-Personalized RAG] Done. Total unique reference articles: {len(global_chunks_list)}")
    return agent_contexts, global_chunks_list, agent_refs_dict


def _round_list(values, digits: int = 3):
    """Round numeric lists for compact, stable LLM input."""
    return [round(float(v), digits) for v in values]


def mask_forecast_data(raw_data: str, target_month: str) -> str:
    """Mask forecast data to prevent look-ahead bias and AFCI/input-feature confusion.

    The forecast files contain:
      - Data: 144 historical observed AFCI values (2014-01 to 2025-12).
      - Forecast: 12 future AFCI values (2026-01 to 2026-12).
      - 95% Confidence: per-month +/- CI width for the AFCI forecast.
      - Variance: per-month predictive variance for the AFCI forecast.

    Agents should use Data only as historical continuity context and should never
    map arbitrary Data indices to 2026 forecast months.
    """
    try:
        import ast
        from datetime import datetime

        month_idx = int(target_month.split("-")[1]) - 1
        if month_idx < 0 or month_idx > 11:
            return raw_data

        parsed = {}
        for line in raw_data.strip().split("\n"):
            if ":" not in line:
                continue
            prefix, arr_str = line.split(":", 1)
            prefix = prefix.strip()
            if prefix in {"Data", "Forecast", "95% Confidence", "Variance"}:
                parsed[prefix] = ast.literal_eval(arr_str.strip())

        forecast = parsed.get("Forecast", [])
        ci = parsed.get("95% Confidence", [])
        variance = parsed.get("Variance", [])
        data = parsed.get("Data", [])

        forecast_val = float(forecast[month_idx])
        ci_val = float(ci[month_idx]) if month_idx < len(ci) else None
        var_val = float(variance[month_idx]) if month_idx < len(variance) else None

        # Canonical previous AFCI for MoM. For Jan 2026, use the last observed
        # 2025-12 historical AFCI endpoint as the continuity anchor and label it
        # explicitly to prevent arbitrary index/month confusion.
        if month_idx == 0:
            prev_label = "2025-12 historical observed AFCI continuity anchor"
            prev_val = float(data[-1]) if data else None
            history = []
            history_label = "No prior forecast month; target is first forecast horizon."
        else:
            prev_month = f"2026-{month_idx:02d}"
            prev_label = f"{prev_month} forecast AFCI"
            prev_val = float(forecast[month_idx - 1])
            history = forecast[:month_idx]
            history_label = f"Forecast History (2026-01 to {prev_month})"

        lower = forecast_val - ci_val if ci_val is not None else None
        upper = forecast_val + ci_val if ci_val is not None else None
        sigma = ci_val / 1.96 if ci_val is not None else None

        lines = []
        lines.append("AFCI DATA CONTRACT:")
        lines.append("- AFCI = AI Firm Competitiveness Index, a latent dimensionless competitiveness stock.")
        lines.append("- Historical Observed AFCI values cover 2014-01 to 2025-12 and are used only as continuity context.")
        lines.append("- Forecast values are the 12-month future AFCI predictions from 2026-01 to 2026-12.")
        lines.append(f"Historical Observed AFCI period: 2014-01 to 2025-12 ({len(data)} monthly values; full vector compressed to avoid arbitrary index confusion).")
        lines.append(f"Historical Observed AFCI tail (last 6 months ending 2025-12): {_round_list(data[-6:])}")
        lines.append(f"{history_label}: {_round_list(history)}")
        lines.append(f"Previous AFCI (last month / canonical MoM base: {prev_label}): {round(prev_val, 3) if prev_val is not None else 'N/A'}")
        lines.append(f"Forecast Target Value EXACT ({target_month}): {round(forecast_val, 3)}")
        lines.append(f"95% Confidence Width (+/-) for {target_month}: {round(ci_val, 3) if ci_val is not None else 'N/A'}")
        lines.append(f"Variance for {target_month}: {round(var_val, 3) if var_val is not None else 'N/A'}")
        lines.append(f"Implied sigma (authoritative; CI_width / 1.96) for {target_month}: {round(sigma, 3) if sigma is not None else 'N/A'}")
        lines.append(f"Calculated 95% Confidence Interval bounds for {target_month}: [{round(lower, 3) if lower is not None else 'N/A'}, {round(upper, 3) if upper is not None else 'N/A'}]")

        if prev_val is not None:
            mom = forecast_val - prev_val
            mom_pct = (mom / abs(prev_val) * 100.0) if prev_val != 0 else 0.0
            lines.append(f"Canonical MoM AFCI change for {target_month}: {round(mom, 3)} points ({round(mom_pct, 2)}%).")

        # Compact authorized glimpse of visible forecast values for trend context only.
        lines.append(f"Visible Forecast Values up to {target_month}: {_round_list(forecast[:month_idx + 1])}")
        lines.append(f"Visible CI Widths up to {target_month}: {_round_list(ci[:month_idx + 1])}")
        lines.append(f"Visible Variances up to {target_month}: {_round_list(variance[:month_idx + 1])}")

        return "\n".join(lines)
    except Exception as e:
        print(f"Error masking data: {e}")
        return raw_data

async def main():
    print("=== Start Monthly Rolling Validation (Walk-Forward) Debate System ===")
    start_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    log_base_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    # 돌릴 기업 리스트: INTC, TSLA, META, AAPL / MSFT run time: 233m 20.96s -> 4h
    # 🏆 1순위 (본문 메인 Table 배치): 인텔 (INTC)
    # 역할: "높은 수학적 불확실성(High Uncertainty)의 해석"
    # 어필 포인트: 직전 섹션에서 INTC의 Fan Chart가 위아래로 가장 넓게(불확실하게) 퍼진 것을 보여주었습니다. 수학 모델이 결론 내리지 못한 이 불확실성을, 에이전트들이 **"막대한 정부 보조금(긍정적 규제/자본) vs 파운드리 지연 및 AI 가속기 점유율 상실(부정적 펀더멘털)"**이라는 팽팽한 논리적 충돌로 완벽하게 해석해 내는 모습을 본문에 박제해야 합니다. 논문의 서사가 가장 완벽해집니다.
    # 🥈 2순위 (본문 서브 또는 Appendix 첫 번째): 테슬라 (TSLA)
    # 역할: "규제 및 비관적 리스크(Regulatory & Bear Risk)의 극대화"
    # 어필 포인트: 테슬라는 AI 비전(자율주행, 로보택시)으로 인한 기대감이 엄청나지만, 동시에 규제 기관(NHTSA)의 조사와 막대한 인프라 비용이라는 치명적 리스크를 안고 있습니다. 우리 시스템의 규제 에이전트(RC)와 비관적 에이전트(BA)가 가장 맹활약(Stress-test)할 수 있는 최고의 타겟입니다. "에이전트가 단순한 낙관론에 빠지지 않는다"는 것을 증명하기 좋습니다.
    # 🥉 3순위 (Appendix): 애플 (AAPL)
    # 역할: "정량적 과거 지표의 한계를 정성적 이벤트로 극복"
    # 어필 포인트: 애플은 생성형 AI 시장 진입이 늦었기 때문에, 과거 지표만 학습하는 수학 모델은 애플의 미래 격차(Gap)를 보수적으로 예측할 수 있습니다. 하지만 **펀더멘털 에이전트(FA)**가 "Apple Intelligence의 온디바이스 생태계 장악력"이라는 정성적 호재를 근거로 모델의 한계를 보완해 내는 모습을 보여주기에 완벽합니다.
    # 🏅 4순위 (Appendix): 메타 (META)
    # 역할: "비즈니스 모델(수익성 vs 오픈소스 생태계)의 구조적 토론"
    # 어필 포인트: Llama를 통한 AI 생태계 장악이라는 압도적 호재와, 수십조 원의 GPU 인프라 투자 대비 뚜렷한 수익 모델 부재라는 악재가 공존합니다. 이는 단순한 기술 평가가 아니라 재무 에이전트(QA)와 거시 에이전트(MS) 간의 고차원적인 전략 토론(Strategic Debate) 능력을 보여주는 데 적합합니다.
    target_companies = ["MU"] # "MSFT", "INTC", "TSLA", "META", "AAPL"    #, ["MU"]

    for company in target_companies:
        print(f"\n{'='*80}")
        print(f" Starting Long-Time Test for Company: {company}")
        print(f"{'='*80}")
        
        run_dir_name = f"{company}_{current_time}"
        run_dir = os.path.join(log_base_dir, run_dir_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        target_months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"] #
        
        # Load Forecast Data
        data_path = os.path.join(parent_dir, "..", "B-MTGNN", "Bayesian", "12mo", "forecast", "data", f"{company}.txt")
        
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                full_raw_data = f.read()
            print(f"Loaded Full Forecast Data from {data_path}")
        else:
            print(f"Warning: File not found {data_path}. Using dummy data.")
            full_raw_data = f"Dummy B-MTGNN Forecast Data: Upward trajectory projected over next 6 months with 95% confidence."
    
        meta_report_file = os.path.join(run_dir, f"{company}.txt")
        with open(meta_report_file, "w", encoding="utf-8") as meta_f:
            meta_f.write(f"Report for {company}. write to {current_time}\n")
            meta_f.write("=" * 80 + "\n\n")
    
        app = create_debate_graph()
    
        for target_month in target_months:
            print(f"\n{'='*80}")
            print(f" Starting Debate for Target Month: {target_month}")
            print(f"{'='*80}")
            
            # Mask future data
            masked_raw_data = mask_forecast_data(full_raw_data, target_month)
            
            forecast_data = f"""Target Company: {company}
Target Validation Month: {target_month}

[{company}.txt Data - Masked to prevent look-ahead bias]
{masked_raw_data}

**Data Format**:
- Raw Data: Historical multimodal input features for {company} from January 2014 to December 2025 (144 months). These values are NOT the AFCI target and are numerically masked to prevent confusion.
- Forecast: Future AFCI values for {company} up to {target_month}. These are the only values to interpret as AI Firm Competitiveness Index predictions.
- 95% Confidence: Authoritative +/- confidence width for the AFCI forecast.
- Variance: Authoritative AFCI forecast predictive variance.
"""
    
            # Fetch Agent-Personalized RAG contexts (6 persona-specific queries)
            agent_rag_contexts, all_chunks, agent_refs_dict = await fetch_all_agent_rag_contexts(company, target_month)
            
            # Shared fallback context (generic, used by Moderator/cross-debate nodes)
            shared_rag_context = agent_rag_contexts.get("fa", "[TemporalRAG Context Not Available]")
            
            initial_state = {
                "company_name": company,
                "target_month": target_month,
                "cutoff_date": target_month,
                "forecast_data": forecast_data,
                "rag_context": shared_rag_context,  # fallback for cross-debate/moderator
                "qa_rag_context": agent_rag_contexts.get("qa"),
                "fa_rag_context": agent_rag_contexts.get("fa"),
                "ms_rag_context": agent_rag_contexts.get("ms"),
                "ba_rag_context": agent_rag_contexts.get("ba"),
                "rc_rag_context": agent_rag_contexts.get("rc"),
                "ar_rag_context": agent_rag_contexts.get("ar"),
                "agent_refs_dict": agent_refs_dict,
                "messages": []
            }
            
            print(f"Running graph execution for {target_month}...")
            result = await app.ainvoke(initial_state, config={"recursion_limit": 50})
            
            # Save individual log: companyname_target-Year-Month_nowDate_nowTime.txt
            log_file = os.path.join(run_dir, f"{company}_{target_month}.txt")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Company: {company} | Target Month: {target_month} | Cutoff: {target_month}\n")
                f.write("=" * 80 + "\n")
                for msg in result.get("messages", []):
                    f.write(f"{msg}\n")
                    f.write("-" * 80 + "\n")
                
                f.write("\n=== References ===\n")
                from urllib.parse import urlparse
                for i, chunk in enumerate(all_chunks):
                    title = chunk.get('article_title', 'Unknown Title').strip()
                    url = chunk.get('article_url', '')
                    domain = urlparse(url).netloc if url else 'Unknown Journal'
                    domain = domain.replace("www.", "")
                    f.write(f"Doc {i+1}: {title}, {domain} ({url})\n")
                    
            print(f"-> Log saved to {log_file}")
            
            # Append to Meta Report
            mod_report = result.get("moderator_report", "No report generated.")
            with open(meta_report_file, "a", encoding="utf-8") as meta_f:
                meta_f.write(f"### [Target Month: {target_month}]\n")
                meta_f.write(f"{mod_report}\n\n")
                
                meta_f.write("=== References ===\n")
                for i, chunk in enumerate(all_chunks):
                    title = chunk.get('article_title', 'Unknown Title').strip()
                    url = chunk.get('article_url', '')
                    domain = urlparse(url).netloc if url else 'Unknown Journal'
                    domain = domain.replace("www.", "")
                    meta_f.write(f"Doc {i+1}: {title}, {domain} ({url})\n")
                meta_f.write("\n")
                
                meta_f.write("-" * 80 + "\n\n")
    
        print(f"\n{'='*80}")
        print(f"All 6 months completed. Meta Report saved to: {meta_report_file}")
    
    end_time = time.time()
    mins = int((end_time - start_time) // 60)
    secs = (end_time - start_time) % 60
    print(f"\nTotal runtime: {mins}m {secs:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
