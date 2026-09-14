import operator
from typing import TypedDict, List, Optional, Annotated

def merge_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    return {**(a or {}), **(b or {})}

class DebateState(TypedDict):
    """State Schema Definition for LangGraph Debate System"""
    # 1. Base Input & Context
    company_name: str
    target_month: str               # The specific month being evaluated (e.g., "2026-02")
    cutoff_date: str
    forecast_data: str              # B-MTGNN Forecast Data from TXT (Masked)
    rag_context: str                # Shared fallback TemporalRAG Context

    # 1-1. Agent-Personalized RAG Contexts (Persona-Specific Retrieval)
    qa_rag_context: Optional[str]   # Quantitative: revenue, metrics, growth data
    fa_rag_context: Optional[str]   # Fundamental: product launches, R&D, partnerships
    ms_rag_context: Optional[str]   # Macro: sector cycles, geopolitics, spending trends
    ba_rag_context: Optional[str]   # Bear: failures, competition threats, market share loss
    rc_rag_context: Optional[str]   # Regulatory: legal, antitrust, GDPR, compliance
    ar_rag_context: Optional[str]   # Academic: forecasting methodology, model limitations

    # 1-2. Agent Reference Lists (Formatted Strings for exact appended citations)
    agent_refs_dict: Optional[dict]

    # 2. Phase 1: Independent Statements
    qa_statement: Optional[str]
    fa_statement: Optional[str]
    ms_statement: Optional[str]
    ba_statement: Optional[str]
    rc_statement: Optional[str]
    ar_statement: Optional[str]

    # 2-1. Agent Stances (structured stance extraction per agent for agreement matrix)
    # Entries carry stance_score in [-1, 1] and associated evidence/lens fields.
    agent_stances: Annotated[Optional[dict], merge_dicts]

    # 3. Phase 2: Cross Debate
    qa_ba_debate: Optional[str]
    fa_ms_debate: Optional[str]
    rc_review: Optional[str]
    ar_review: Optional[str]
    ar_peer_review: Optional[str]
    ar_verdict: Optional[str]
    ar_confidence_score: Optional[float]
    revision_count: int
    ar_rejection_reason: Optional[str]

    # 4. Phase 3: Synthesis
    moderator_report: Optional[str]

    # 4-1. Agent Agreement Matrix & Strategic Uncertainty
    # agreement_matrix: JSON string summarizing each agent's stance for the target horizon.
    # disagreement_score: D(π) = Var(s_1, ..., s_K) across agents (excluding AR/meta-reviewer).
    agreement_matrix: Optional[str]
    disagreement_score: Optional[float]

    # Meta
    messages: Annotated[List[str], operator.add]
