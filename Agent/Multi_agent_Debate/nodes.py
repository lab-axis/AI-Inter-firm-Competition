import os
import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator

class ModeratorScore(BaseModel):
    consensus_points: str
    points_of_contention: str
    academic_peer_review_summary: str
    step_by_step_rationale: str
    score_data_grounding: float
    score_logical_coherence: float
    score_risk_robustness: float
    score_scientific_validity: float

    @field_validator('consensus_points', 'points_of_contention', 'academic_peer_review_summary', 'step_by_step_rationale', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if isinstance(v, list):
            return " | ".join(str(i) for i in v)
        return str(v) if v is not None else ""

    @field_validator('score_data_grounding', 'score_logical_coherence', 'score_risk_robustness', 'score_scientific_validity', mode='before')
    @classmethod
    def coerce_to_float(cls, v):
        if isinstance(v, str):
            # Extract first number from string
            m = re.search(r'[\d.]+', v)
            return float(m.group()) if m else 5.0
        return float(v)

class ARPeerReviewScore(BaseModel):
    formal_verdict: str
    score_logical: float
    score_quantitative: float
    score_fallacy: float
    rationale: str
    required_improvements: str = ""  # default to avoid missing field error

    @field_validator('formal_verdict', 'rationale', 'required_improvements', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if isinstance(v, list):
            return " | ".join(str(i) for i in v)
        return str(v) if v is not None else ""

    @field_validator('score_logical', 'score_quantitative', 'score_fallacy', mode='before')
    @classmethod
    def coerce_score_to_float(cls, v):
        if isinstance(v, str):
            m = re.search(r'[\d.]+', v)
            return float(m.group()) if m else 0.0
        return float(v)

ALLOWED_VALUE_LENSES = {
    "Financial", "Product-Market", "Technological", "Research",
    "IP-Risk", "Regulatory", "Strategic-Investment"
}

LENS_ALIASES = {
    "ai sector growth": "Strategic-Investment",
    "r&d investments": "Research",
    "r&d investment": "Research",
    "cloud services expansion": "Product-Market",
    "competitive landscape shift": "Product-Market",
    "cloud services": "Product-Market",
    "ai adoption": "Product-Market",
    "ai investments": "Strategic-Investment",
    "investment": "Strategic-Investment",
    "legal": "Regulatory",
    "compliance": "Regulatory",
    "ip": "IP-Risk",
    "intellectual property": "IP-Risk",
    "technology": "Technological",
    "technical": "Technological",
    "r&d": "Research",
}

class AgentStance(BaseModel):
    stance_score: float = Field(0.0, ge=-1.0, le=1.0)
    evidence_summary: str = ""
    value_lens: List[str] = Field(default_factory=list)

    @field_validator('stance_score', mode='before')
    @classmethod
    def coerce_stance_score_to_float(cls, v):
        if isinstance(v, str):
            m = re.search(r'-?\d+(?:\.\d+)?', v)
            return float(m.group()) if m else 0.0
        return float(v)

    @field_validator('value_lens', mode='before')
    @classmethod
    def normalize_lens_list(cls, v):
        if v is None:
            raw_items = []
        elif isinstance(v, str):
            raw_items = re.split(r'\s*(?:\||,|/|;)\s*', v)
        else:
            raw_items = []
            for item in list(v):
                raw_items.extend(re.split(r'\s*(?:\||,|/|;)\s*', str(item)))

        normalized = []
        for item in raw_items:
            item_str = str(item).strip()
            if not item_str:
                continue
            if item_str in ALLOWED_VALUE_LENSES:
                mapped = item_str
            else:
                mapped = LENS_ALIASES.get(item_str.lower())
            if mapped and mapped not in normalized:
                normalized.append(mapped)

        return normalized or ["Strategic-Investment"]

from .state import DebateState
from .prompts import (
    COMMON_INSTRUCTION,
    QA_SYSTEM_PROMPT, FA_SYSTEM_PROMPT, MS_SYSTEM_PROMPT,
    BA_SYSTEM_PROMPT, RC_SYSTEM_PROMPT, AR_SYSTEM_PROMPT, MJ_SYSTEM_PROMPT,
    AR_PEER_REVIEW_PROMPT, AGENT_FORMATTING_INSTRUCTION
)

# Configuration for vLLM
VLLM_URL = os.getenv("VLLM_CHAT_URL", "http://localhost:8001/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "casperhansen/deepseek-r1-distill-qwen-7b-awq")

# Limit concurrent vLLM calls to avoid overwhelming the local server
vllm_semaphore = asyncio.Semaphore(5)

async def call_vllm(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_retries: int = 5) -> str:
    """Async call to local vLLM instance with retry and concurrency limits"""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False
    }
    
    async with vllm_semaphore:
        for attempt in range(max_retries):
            try:
                # Increase timeout to 300 seconds (5 minutes) for deep reasoning models
                timeout = aiohttp.ClientTimeout(total=300)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(VLLM_URL, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            message = data["choices"][0]["message"]
                            
                            # DeepSeek-R1 splits reasoning and content. We only want the final content
                            content = message.get("content", "")
                            
                            # Basic strip of <think> tags if they bled into content
                            import re
                            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                            return content
                        else:
                            text = await response.text()
                            print(f"    [VLLM Warning] Attempt {attempt+1} failed with status {response.status}")
                            if attempt == max_retries - 1:
                                return f"[API ERROR {response.status}] {text}"
            except Exception as e:
                print(f"    [VLLM Warning] Attempt {attempt+1} connection error: {e}")
                if attempt == max_retries - 1:
                    return f"[VLLM CONNECTION ERROR] {str(e)}"
            
            # Exponential backoff before retry (1s, 2s, 4s...)
            await asyncio.sleep(2 ** attempt)

def strip_references(text: str) -> str:
    """Removes the appended RAG reference block from agent text before it is
    used as LLM context in subsequent nodes. This prevents cross-agent RAG
    block contamination and keeps each persona's reasoning independent."""
    import re
    return re.sub(r'\n\n### RAG Documents Referenced by \w+ Agent\n.*', '', text, flags=re.DOTALL).strip()


def strip_stance_json(text: str) -> str:
    """Remove model-emitted JSON blocks from user-facing agent logs."""
    text = re.sub(r'(?im)^.*(?:mandatory\s+stance\s+json|internal\s+metadata|stance\s+json|json\s+output).*$','', text)
    text = re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def strip_latex(text: str):
    """Post-processing guard: converts any LaTeX math syntax that the LLM
    may have hallucinated into plain-text equivalents. Applied only to
    QA agent outputs since other agents comply without this."""
    # Remove display math blocks: \[...\] and $$...$$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'\1', text, flags=re.DOTALL)
    # Remove inline math: \(...\) and $...$
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', r'\1', text)
    # Replace common LaTeX commands with plain text equivalents
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1 / \2)', text)
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\left\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    text = re.sub(r'\\left\[', '[', text)
    text = re.sub(r'\\right\]', ']', text)
    text = re.sub(r'\\times', 'x', text)
    text = re.sub(r'\\approx', '≈', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\sigma', 'σ', text)
    text = re.sub(r'\\mu', 'μ', text)
    text = re.sub(r'\\alpha', 'α', text)
    text = re.sub(r'\\beta', 'β', text)
    text = re.sub(r'\\sum', 'Σ', text)
    text = re.sub(r'\\rightarrow', '→', text)
    # Remove any remaining lone backslash-commands (e.g. \cdot, \hat)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Clean up extra blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def infer_stance_score_from_text(response: str) -> float:
    """Infer continuous stance intensity in [-1, +1] from free-form text.

    Negative values indicate skepticism toward the AFCI forecast, positive values
    indicate support, and values near zero indicate mixed/contested reasoning.
    """
    lower = response.lower()
    score = 0.0

    patterns = [
        (0.9, ["strongly supportive", "highly supportive", "robustly supported", "strongly bullish", "highly plausible"]),
        (-0.9, ["strongly skeptical", "not credible", "methodologically invalid", "strongly rejected", "severely overconfident"]),
        (0.45, ["modestly supportive", "cautiously optimistic", "partially supportive", "moderately supportive"]),
        (-0.45, ["partially skeptical", "somewhat skeptical", "moderately skeptical", "cautious", "qualified support"]),
        (0.0, ["overall verdict: mixed", "mixed", "neutral", "contested"]),
        (0.65, ["overall verdict: supportive", "supportive", "bullish", "supported", "plausible", "strengthening", "directionally plausible"]),
        (-0.65, ["overall verdict: skeptical", "skeptical", "bearish", "not supported", "unsupported", "implausible", "weakening", "overconfident", "contradicted"]),
    ]

    # Verdict lines get priority because they are explicit agent summaries.
    verdict = re.search(r'overall verdict\s*:\s*([^\n]+)', lower)
    search_space = verdict.group(1) if verdict else lower
    matched = False
    for val, terms in patterns:
        if any(term in search_space for term in terms):
            score = val
            matched = True
            break

    if not matched:
        pos_hits = sum(1 for term in ["support", "plausible", "bullish", "strength", "align"] if term in lower)
        neg_hits = sum(1 for term in ["skeptic", "unsupported", "risk", "weak", "overconfident", "contradict"] if term in lower)
        if pos_hits or neg_hits:
            score = max(-0.8, min(0.8, 0.18 * (pos_hits - neg_hits)))

    # Hedge/uncertainty language reduces magnitude without changing direction.
    hedge_terms = ["but", "however", "although", "uncertain", "mixed", "contested", "caution", "qualified"]
    if any(term in lower for term in hedge_terms):
        score *= 0.7

    return round(max(-1.0, min(1.0, score)), 3)


def infer_lenses_from_text(response: str, agent_key: str) -> List[str]:
    lower = response.lower()
    lenses = []
    def add(lens):
        if lens not in lenses:
            lenses.append(lens)
    if agent_key in {"qa", "qa_rebuttal"}:
        add("Financial")
    if agent_key == "fa":
        add("Product-Market"); add("Technological")
    if agent_key == "ms":
        add("Strategic-Investment")
    if agent_key == "rc":
        add("Regulatory")
    if "research" in lower or "publication" in lower or "r&d" in lower:
        add("Research")
    if "patent" in lower or "copyright" in lower or "licensing" in lower or "intellectual property" in lower:
        add("IP-Risk")
    if "regulat" in lower or "antitrust" in lower or "ftc" in lower or "eu ai act" in lower:
        add("Regulatory")
    if "product" in lower or "market" in lower or "azure" in lower or "copilot" in lower or "adoption" in lower:
        add("Product-Market")
    if "technical" in lower or "technology" in lower or "model" in lower or "infrastructure" in lower:
        add("Technological")
    if "investment" in lower or "capex" in lower or "macro" in lower or "rate" in lower:
        add("Strategic-Investment")
    return lenses or ["Strategic-Investment"]


def summarize_evidence_from_text(response: str) -> str:
    cleaned = re.sub(r'```json\s*.*?\s*```', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
    verdict_match = re.search(r'overall verdict\s*:\s*([^\n]+)', cleaned, flags=re.IGNORECASE)
    if verdict_match:
        return verdict_match.group(0).strip()[:300]
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return (sentences[-1] if sentences else cleaned)[:300]


def extract_stance(response: str, agent_key: str) -> dict:
    """Infer stance/value lenses from free-form agent text.

    JSON blocks are no longer required. If a legacy JSON stance exists, it is used;
    otherwise stance and lenses are inferred heuristically from the visible text.
    """
    try:
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, flags=re.DOTALL | re.IGNORECASE)
        if json_blocks:
            data = json.loads(json_blocks[-1].strip())
            if "stance_score" not in data and "stance" in data:
                data["stance_score"] = data.get("stance", 0)
            stance = AgentStance(**data)
            return {
                "agent": agent_key,
                "stance_score": stance.stance_score,
                "evidence_summary": stance.evidence_summary,
                "value_lens": stance.value_lens,
                "parse_ok": True,
                "inferred": False,
            }
    except Exception:
        pass

    return {
        "agent": agent_key,
        "stance_score": infer_stance_score_from_text(response),
        "evidence_summary": summarize_evidence_from_text(response),
        "value_lens": infer_lenses_from_text(response, agent_key),
        "parse_ok": True,
        "inferred": True,
    }


def merge_agent_stance(state: dict, agent_key: str, response: str) -> dict:
    stances = dict(state.get("agent_stances") or {})
    stance = extract_stance(response, agent_key)
    stance["quality_ok"] = has_substantive_analysis(response)
    if not stance["quality_ok"]:
        stance["quality_warning"] = "Response was empty, near-empty, or mostly metadata after retry."
    stances[agent_key] = stance
    return stances


def _agent_response_bundle(state: dict) -> Dict[str, str]:
    """Collect free-form agent responses for batch semantic stance scoring."""
    mapping = {
        "qa": state.get("qa_statement", ""),
        "fa": state.get("fa_statement", ""),
        "ms": state.get("ms_statement", ""),
        "ba": state.get("ba_statement", ""),
        "rc": state.get("rc_statement", ""),
        "qa_rebuttal": state.get("qa_ba_debate", ""),
        "ar": state.get("ar_statement", ""),
    }
    return {k: strip_stance_json(strip_references(v)) for k, v in mapping.items() if v}


def _fallback_all_agent_stances(state: dict) -> Dict[str, dict]:
    """Fallback heuristic stance scores if batch LLM scoring fails."""
    existing = dict(state.get("agent_stances") or {})
    for agent_key, response in _agent_response_bundle(state).items():
        stance = extract_stance(response, agent_key)
        stance["quality_ok"] = has_substantive_analysis(response)
        stance["extraction_method"] = "heuristic_fallback"
        existing[agent_key] = stance
    return existing


async def score_all_agent_stances(state: dict) -> Dict[str, dict]:
    """Use one LLM call to semantically score all agent stance intensities.

    Agents write natural analysis. This scorer reads those analyses and extracts
    continuous stance_score values in [-1,+1]. It is deliberately separated from
    agent generation so the agents are not forced to self-report JSON.
    """
    responses = _agent_response_bundle(state)
    if not responses:
        return dict(state.get("agent_stances") or {})

    response_block = "\n\n".join(
        f"### AGENT: {agent}\n{content[:3500]}" for agent, content in responses.items()
    )
    forecast_data = state.get("forecast_data", "")

    sys_prompt = """You are a semantic stance-intensity extraction module.
Your task is NOT to debate and NOT to add your own opinion. Read each agent's free-form analysis and extract how strongly that agent supports or doubts the B-MTGNN AFCI forecast.
Return ONLY valid JSON, no markdown.

Scoring scale:
-1.0 = strongly skeptical: the agent clearly argues the forecast is unsupported, implausible, overconfident, or strategically weak.
-0.5 = moderately skeptical: the agent emphasizes meaningful downside, risk, or weak support.
 0.0 = mixed / neutral / contested: the agent sees balanced support and risks or is methodologically undecided.
+0.5 = moderately supportive: the agent finds the forecast directionally plausible with caveats.
+1.0 = strongly supportive: the agent strongly supports the forecast as well-grounded.

Use any decimal value in [-1,+1] when the expressed stance is between these anchors. Score the agent's expressed view, not your own view.
Allowed value_lens entries: Financial, Product-Market, Technological, Research, IP-Risk, Regulatory, Strategic-Investment.
"""
    user_prompt = f"""Forecast Data:
{forecast_data}

Agent Responses:
{response_block}

Return this exact JSON shape as an array:
[
  {{"agent":"qa", "stance_score":0.0, "evidence_summary":"one concise sentence", "value_lens":["Financial"]}}
]
Include only agents present in the Agent Responses. stance_score must be numeric in [-1,+1].
"""

    try:
        raw = await call_vllm(sys_prompt, user_prompt, temperature=0.0, max_retries=3)
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, flags=re.DOTALL | re.IGNORECASE)
        json_str = json_match.group(1).strip() if json_match else raw[raw.find('['):raw.rfind(']') + 1]
        parsed = json.loads(json_str)
        if not isinstance(parsed, list):
            raise ValueError("stance scorer did not return a JSON array")

        scored = {}
        for item in parsed:
            agent_key = str(item.get("agent", "")).strip()
            if agent_key not in responses:
                continue
            stance = AgentStance(**item)
            scored[agent_key] = {
                "agent": agent_key,
                "stance_score": round(max(-1.0, min(1.0, float(stance.stance_score))), 3),
                "evidence_summary": stance.evidence_summary,
                "value_lens": stance.value_lens,
                "parse_ok": True,
                "inferred": True,
                "extraction_method": "llm_batch",
                "quality_ok": has_substantive_analysis(responses[agent_key]),
            }

        # Fill missing agents with fallback instead of failing the whole scoring pass.
        fallback = _fallback_all_agent_stances(state)
        for agent_key in responses:
            if agent_key not in scored:
                scored[agent_key] = fallback[agent_key]
        return scored
    except Exception as e:
        print(f"    [Stance Scorer Warning] Batch LLM scoring failed; using heuristic fallback: {e}")
        return _fallback_all_agent_stances(state)


def has_substantive_analysis(response: str, min_chars: int = 400) -> bool:
    """Detect JSON-only or near-empty responses before appending references.

    This is intentionally permissive: agents should think freely. Citation and
    grounding quality are handled later as warnings/scoring signals.
    """
    without_json = re.sub(r'```json\s*.*?\s*```', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
    without_headings = re.sub(r'[#*`\s-]+', '', without_json)
    return len(without_headings) >= min_chars


async def ensure_substantive_response(agent_label: str, response: str, sys_prompt: str, user_prompt: str) -> str:
    """One retry when the model returns only the stance JSON or an empty shell."""
    if has_substantive_analysis(response):
        return response
    retry_prompt = user_prompt + (
        "\n\n[SYSTEM RETRY - EMPTY OR JSON-ONLY RESPONSE]\n"
        "Your previous response was empty, near-empty, or mostly machine-readable metadata. "
        "Write a natural analytical response in your role. Use [Doc X] citations for important concrete evidence, "
        "but do not output JSON. End with a short verbal verdict such as 'Overall verdict: Strongly supportive/Mixed/Moderately skeptical'."
    )
    print(f"    -> [RETRY] {agent_label} response lacked substantive analysis; requesting one retry.")
    retry_response = await call_vllm(sys_prompt, retry_prompt, temperature=0.15, max_retries=3)
    if not has_substantive_analysis(retry_response):
        retry_response = retry_response.rstrip() + "\n\n[QUALITY WARNING] Response remained insufficient after one retry; downstream scores should treat this as weak agent evidence."
    return retry_response


def append_references(response: str, agent_key: str, state: dict) -> str:
    """Appends the exact RAG documents provided to this agent, with deduplication.
    Path entries (graph traversal metadata) are excluded as they are not human-readable
    references; only Doc (article chunk) entries are shown."""
    response = strip_stance_json(response)
    refs_raw = state.get("agent_refs_dict", {}).get(agent_key, "")
    if not refs_raw:
        return response
    
    # Deduplicate lines by their content while preserving order.
    # Exclude Path lines (graph traversal metadata — not relevant to readers).
    seen = set()
    unique_ref_lines = []
    for line in refs_raw.splitlines():
        if line.startswith("- Path "):
            continue  # Skip graph path metadata
        if line not in seen:
            seen.add(line)
            unique_ref_lines.append(line)
    
    if not unique_ref_lines:
        return response
    
    refs_deduped = "\n".join(unique_ref_lines)
    agent_label = agent_key.upper()
    return f"{response.strip()}\n\n### RAG Documents Referenced by {agent_label} Agent\n{refs_deduped}"

# ==========================================
# Phase 0: Data Load (Will be handled before graph entry or as first node)
# ==========================================
def load_data_node(state: DebateState) -> DebateState:
    """Pass-through node since main.py pre-loads the data into state."""
    return state

# ==========================================
# Phase 1: Independent Statements
# ==========================================
def get_common(state: DebateState) -> str:
    return COMMON_INSTRUCTION.format(
        company_name=state["company_name"],
        target_month=state["target_month"],
        cutoff_date=state["cutoff_date"]
    )

async def qa_node(state: DebateState) -> DebateState:
    # qa Agent is Quantitative Analyst.
    print(f"  [Phase 1] QA Agent is analyzing for {state['target_month']}...")
    sys_prompt = QA_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("qa_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Quantitative/Financial Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    response = await ensure_substantive_response("QA", response, sys_prompt, user_prompt)
    response = strip_latex(response)  # Guard: remove any LaTeX the 7B model hallucinated
    agent_stances = merge_agent_stance(state, "qa", response)
    response = append_references(response, "qa", state)
    return {"qa_statement": response, "agent_stances": agent_stances, "messages": [f"**QA Agent ({state['target_month']})**:\n{response}"]}

async def fa_node(state: DebateState) -> DebateState:
    # fa Agent is Fundamental Analyst.
    print(f"  [Phase 1] FA Agent is analyzing for {state['target_month']}...")
    sys_prompt = FA_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("fa_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Fundamental/Product Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    response = await ensure_substantive_response("FA", response, sys_prompt, user_prompt)
    agent_stances = merge_agent_stance(state, "fa", response)
    response = append_references(response, "fa", state)
    return {"fa_statement": response, "agent_stances": agent_stances, "messages": [f"**FA Agent ({state['target_month']})**:\n{response}"]}

async def ms_node(state: DebateState) -> DebateState:
    # ms Agent is Macro Strategist.
    print(f"  [Phase 1] MS Agent is analyzing for {state['target_month']}...")
    sys_prompt = MS_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("ms_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Macro/Sector Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    response = await ensure_substantive_response("MS", response, sys_prompt, user_prompt)
    agent_stances = merge_agent_stance(state, "ms", response)
    response = append_references(response, "ms", state)
    return {"ms_statement": response, "agent_stances": agent_stances, "messages": [f"**MS Agent ({state['target_month']})**:\n{response}"]}

async def ba_node(state: DebateState) -> DebateState:
    # ba Agent is Bear Analyst.
    print(f"  [Phase 1] BA Agent is analyzing for {state['target_month']}...")
    sys_prompt = BA_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    qa_stmt = state.get("qa_statement", "QA has not spoken yet.")
    rag = state.get("ba_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Risk/Competitive Threats Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}\n\nQA's STATEMENT:\n{qa_stmt}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.3)
    response = await ensure_substantive_response("BA", response, sys_prompt, user_prompt)
    agent_stances = merge_agent_stance(state, "ba", response)
    response = append_references(response, "ba", state)
    return {"ba_statement": response, "agent_stances": agent_stances, "messages": [f"**BA Agent ({state['target_month']})**:\n{response}"]}

async def rc_node(state: DebateState) -> DebateState:
    # rc Agent is Risk & Compliance Analyst.
    print(f"  [Phase 1] RC Agent is analyzing for {state['target_month']}...")
    sys_prompt = RC_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("rc_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Regulatory/Legal Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    response = await ensure_substantive_response("RC", response, sys_prompt, user_prompt)
    agent_stances = merge_agent_stance(state, "rc", response)
    response = append_references(response, "rc", state)
    return {"rc_statement": response, "agent_stances": agent_stances, "messages": [f"**RC Agent ({state['target_month']})**:\n{response}"]}

async def ar_node(state: DebateState) -> DebateState:
    # ar Agent is Academic Reviewer.
    print(f"  [Phase 1] AR Agent is analyzing for {state['target_month']}...")
    sys_prompt = AR_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("ar_rag_context") or state["rag_context"]
    user_prompt = f"RAG CONTEXT (Academic/Methodological Focus):\n{rag}\n\nFORECAST DATA (Masked for {state['target_month']}):\n{state['forecast_data']}"
    
    response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    response = await ensure_substantive_response("AR", response, sys_prompt, user_prompt)
    # AR is a methodological gate, not a strategic value-lens voter; keep its stance
    # for auditability but exclude it from disagreement variance downstream.
    agent_stances = merge_agent_stance(state, "ar", response)
    response = append_references(response, "ar", state)
    return {"ar_statement": response, "agent_stances": agent_stances, "messages": [f"**AR Agent ({state['target_month']})**:\n{response}"]}

# ==========================================
# Phase 2: Cross Debate (Simplified sequential for LangGraph)
# ==========================================
async def cross_debate_node(state: DebateState) -> DebateState:
    print(f"  [Phase 2] QA Agent is defending in Cross Debate for {state['target_month']}...")
    # Strip the appended RAG block from BA's statement so the QA rebuttal LLM
    # only sees BA's analytical arguments, not BA's reference list.
    ba_stmt_clean = strip_references(state.get("ba_statement", ""))
    
    sys_prompt = QA_SYSTEM_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction=AGENT_FORMATTING_INSTRUCTION)
    rag = state.get("qa_rag_context") or state["rag_context"]
    original_qa_clean = strip_references(state.get("qa_statement", ""))
    rc_stmt_clean = strip_references(state.get("rc_statement", ""))
    ba_warnings = audit_citation_grounding(ba_stmt_clean, state)
    rc_warnings = audit_citation_grounding(rc_stmt_clean, state)
    adversarial_warning_block = "\n".join(
        [f"- BA warning: {w}" for w in ba_warnings[:6]] +
        [f"- RC warning: {w}" for w in rc_warnings[:4]]
    ) or "None"
    user_prompt = f"""RAG CONTEXT (Quantitative Focus):
{rag}

FORECAST DATA (authoritative numerical source for AFCI, CI, sigma, variance, and MoM):
{state['forecast_data']}

ORIGINAL QA STATEMENT (your prior quantitative position; use it as your baseline, not BA's framing):
{original_qa_clean}

The Bear Analyst (BA) has challenged your forecast with the following points:
{ba_stmt_clean}

AUTOMATED SOFT WARNINGS ABOUT BA/RC CLAIMS (use these to avoid adopting weakly grounded adversarial claims):
{adversarial_warning_block}

Task:
1. Defend, revise, or partially concede the forecast based on the FORECAST DATA and your own RAG context.
2. Prioritize reliability and document quality: use only evidence that is clearly about the target company and directly supports the claim. If a BA/RC claim is weakly grounded, generic, or appears to rely on an unrelated company/article, explicitly say it should be treated cautiously rather than adopting it.
3. Do NOT invent probabilities, materialization rates, AFCI point impacts, or adjusted confidence intervals unless they are explicitly provided in the Forecast Data or clearly stated in a cited RAG document. If exact probabilities or impact magnitudes are unavailable, discuss the risk directionally instead of numerically.
4. Use FORECAST DATA for all model-side numerical claims. Use RAG documents only for contextual business evidence.
5. You are ABSOLUTELY REQUIRED to include at least 1 or more citations of the format 'Doc X' or '[Doc X]' in your response. A defense without any document citations is invalid and will be rejected. Cite key concrete evidence with specific document numbers (e.g., 'According to Doc 1...', 'Document 2 shows...', or '[Doc 3]') to prove your arguments are grounded. Do not write a generic defense; name the exact Doc.
6. Write a natural analytical response and end with a verbal verdict such as 'Overall verdict: Supportive/Mixed/Skeptical'. Do not output JSON.
"""
    
    rev_count = state.get("revision_count", 0)
    if rev_count > 0:
        previous_qa_rebuttal = strip_stance_json(strip_references(state.get("qa_ba_debate", "")))
        previous_ar_review = strip_references(state.get("ar_peer_review", ""))
        ar_reason = state.get("ar_rejection_reason", "")
        user_prompt += f"""

[REVISION CONTEXT - ROUND {rev_count}/5]
Your previous rebuttal did not reach the academic-review threshold. The Academic Reviewer already explained why the previous answer was insufficient. Use that feedback directly instead of rewriting from scratch.

PREVIOUS QA REBUTTAL:
{previous_qa_rebuttal[:3500]}

ACADEMIC REVIEWER'S FULL FEEDBACK:
{previous_ar_review[:3000]}

REVISION / REJECTION REASON:
{str(ar_reason)[:1500]}

REVISION TASK:
Revise your previous rebuttal by directly addressing the Academic Reviewer's feedback.
Preserve the parts of your previous rebuttal that were correct.
Fix the parts the Academic Reviewer identified as weak, missing, unsupported, or underdeveloped.
You MUST ensure that at least one document citation of the format 'Doc X' or '[Doc X]' is included in your revised response. Responses without citations will continue to be rejected.
If AR noted missing or weak citations, add relevant [Doc X] citations from your QA RAG context or remove the unsupported claim.
If AR noted weak quantitative reasoning, improve the explanation using only Forecast Data and cited RAG evidence.
Do not repeat weakly grounded BA/RC claims.
Do not invent probabilities, impact magnitudes, or adjusted forecasts unless explicitly supported by Forecast Data or cited RAG.
Do not output JSON. End with a verbal verdict.
"""
        
    qa_rebuttal = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
    qa_rebuttal = await ensure_substantive_response("QA Rebuttal", qa_rebuttal, sys_prompt, user_prompt)
    qa_rebuttal = strip_latex(qa_rebuttal)  # Guard: remove any LaTeX the 7B model hallucinated
    agent_stances = merge_agent_stance(state, "qa_rebuttal", qa_rebuttal)
    qa_rebuttal = append_references(qa_rebuttal, "qa", state)
    
    return {
        "qa_ba_debate": qa_rebuttal,
        "agent_stances": agent_stances,
        "messages": [f"**QA Agent Rebuttal ({state['target_month']})**:\n{qa_rebuttal}"]
    }

def _build_doc_index(state: dict) -> Dict[str, str]:
    """Build a global Doc-id -> text/title index from all persona RAG contexts."""
    doc_index = {}
    for key in ("qa_rag_context", "fa_rag_context", "ms_rag_context", "ba_rag_context", "rc_rag_context", "ar_rag_context", "rag_context"):
        ctx = state.get(key) or ""
        for match in re.finditer(r'\[Doc\s+(\d+)\]\s*(?:\([^)]*\))?:\s*(.*?)(?=\n\[Doc\s+\d+\]|\n\[Path\s+\d+\]|\Z)', ctx, flags=re.DOTALL):
            doc_id = match.group(1)
            text = re.sub(r'\s+', ' ', match.group(2)).strip()
            if text and doc_id not in doc_index:
                doc_index[doc_id] = text
    return doc_index


def _claim_sentences_with_docs(text: str) -> List[tuple]:
    """Return sentences/fragments containing Doc citations."""
    fragments = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [(frag.strip(), re.findall(r'\[?Doc(?:ument)?\.?\s*(\d+)\]?', frag, re.IGNORECASE)) for frag in fragments if "Doc" in frag]


def _keyword_overlap_score(claim: str, doc_text: str, company: str) -> int:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "would", "could", "should", "forecast", "afci", "company", "value", "month", "doc", "according"}
    # Do not count the target ticker/company token as grounding by itself. Otherwise
    # unrelated claims like "Fiserv cash yield drives MSFT" can pass merely because
    # the cited document mentions MSFT. Require overlap in substantive event terms.
    company_terms = {company.lower()} if company else set()
    claim_words = {w.lower() for w in re.findall(r'[A-Za-z][A-Za-z0-9_-]{3,}', claim) if w.lower() not in stop and w.lower() not in company_terms}
    doc_words = {w.lower() for w in re.findall(r'[A-Za-z][A-Za-z0-9_-]{3,}', doc_text) if w.lower() not in stop and w.lower() not in company_terms}
    return len(claim_words & doc_words)


def _is_math_or_forecast_claim(claim: str) -> bool:
    """Skip lexical RAG grounding for pure math/forecast-data statements.

    These are grounded in the Forecast Data rather than article text, so forcing
    article keyword overlap would create false positives.
    """
    lower = claim.lower()
    math_markers = ["z-score", "sigma", "variance", "confidence interval", "ci", "mom", "forecast target", "afci change", "upper bound", "lower bound"]
    has_math_marker = any(m in lower for m in math_markers)
    numbers = len(re.findall(r'[-+]?\d+(?:\.\d+)?%?', claim))
    return has_math_marker and numbers >= 1


def audit_citation_grounding(text: str, state: dict, min_overlap: int = 2) -> List[str]:
    """Verify that cited document IDs actually exist in the provided RAG contexts."""
    doc_index = _build_doc_index(state)
    failures = []
    cited_docs = set(re.findall(r'\[?Doc(?:ument)?\.?\s*(\d+)\]?', text, re.IGNORECASE))
    for doc_id in cited_docs:
        if doc_id not in doc_index:
            failures.append(f"Cited [Doc {doc_id}] does not exist in the provided RAG contexts.")
    return failures


def detect_fabricated_specifics(text: str, state: dict) -> List[str]:
    """Deprecated/no-op.

    Regex-based fabrication detection caused false hard failures for real but
    paraphrased facts and figures. Factual plausibility is now handled by AR as
    contextual review, while automated hard failure is limited to nonexistent Doc
    citations.
    """
    return []


async def ar_peer_review_node(state: DebateState) -> DebateState:
    print(f"  [Phase 2] AR Review Agent is evaluating QA's defense for {state['target_month']}...")
    ba_stmt_clean = strip_references(state.get("ba_statement", ""))
    qa_rebuttal_clean = strip_references(state.get("qa_ba_debate", ""))
    
    cited_docs = set(re.findall(r'\[?Doc(?:ument)?\.?\s*(\d+)\]?', qa_rebuttal_clean, re.IGNORECASE))
    
    if not cited_docs:
        print("    -> [WARNING] QA failed hard citation audit (no citations found). Auto-rejecting without calling LLM.")
        ar_reason = "[AUTOMATED CITATION AUDIT FAILURE] QA's defense did not contain any valid [Doc X] citations. Under strict academic review standards, all quantitative claims must be grounded in the provided RAG Context. The defense is automatically rejected."
        return {
            "ar_peer_review": ar_reason,
            "ar_verdict": "Rejected",
            "ar_confidence_score": 0.0,
            "ar_rejection_reason": ar_reason,
            "revision_count": state.get("revision_count", 0) + 1,
            "messages": [f"**AR Peer Review ({state['target_month']})**:\n{ar_reason}"]
        }

    grounding_failures = audit_citation_grounding(qa_rebuttal_clean, state)
    hard_grounding_failures = [f for f in grounding_failures if f.startswith("Cited [Doc")]
    soft_grounding_warnings = [f for f in grounding_failures if not f.startswith("Cited [Doc")]
    hard_failures = hard_grounding_failures
    if hard_failures:
        print("    -> [WARNING] QA cited nonexistent RAG document(s). Auto-rejecting without calling LLM.")
        ar_reason = "[AUTOMATED HARD CITATION AUDIT FAILURE] " + " | ".join(hard_failures[:5])
        return {
            "ar_peer_review": ar_reason,
            "ar_verdict": "Rejected",
            "ar_confidence_score": 0.0,
            "ar_rejection_reason": ar_reason,
            "revision_count": state.get("revision_count", 0) + 1,
            "messages": [f"**AR Peer Review ({state['target_month']})**:\n{ar_reason}"]
        }
    if soft_grounding_warnings:
        print("    -> [WARNING] QA has soft grounding warnings; passing to AR for scoring instead of auto-rejecting.")
    # ------------------------------------------------
    
    sys_prompt = AR_PEER_REVIEW_PROMPT.format(company_name=state["company_name"], target_month=state["target_month"], common=get_common(state), format_instruction="")
    rag = state.get("ar_rag_context") or state["rag_context"]
    soft_warning_block = "\n".join(f"- {w}" for w in soft_grounding_warnings[:10]) or "None"
    base_user_prompt = f"RAG CONTEXT:\n{rag}\n\nBA'S ATTACK:\n{ba_stmt_clean}\n\nQA'S REBUTTAL:\n{qa_rebuttal_clean}\n\nAUTOMATED SOFT GROUNDING WARNINGS (do not auto-reject solely for these; use them to adjust the score and required improvements):\n{soft_warning_block}\n\nCRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object matching the exact schema provided in the system prompt. Do NOT output any markdown headings or text outside the ```json block."
    
    ar_json_template = '''```json
{
  "formal_verdict": "EXACTLY ONE OF: Strongly Accepted | Accepted with Minor Revisions | Major Revisions Required | Rejected | Strongly Rejected - Methodological Flaw",
  "score_logical": "insert float between 0.0 and 40.0",
  "score_quantitative": "insert float between 0.0 and 40.0",
  "score_fallacy": "insert float between 0.0 and 20.0",
  "rationale": "Your 3-5 sentence justification here.",
  "required_improvements": "Specific steps QA must take to improve the defense."
}
```'''
    user_prompt = f"{base_user_prompt}\n\nYou MUST fill in ALL FOUR fields. The JSON template you MUST follow exactly:\n{ar_json_template}"
    max_retries = 3
    for attempt in range(max_retries):
        response = await call_vllm(sys_prompt, user_prompt, temperature=0.2)
        
        # Parse AR JSON
        try:
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response[response.find('{'):response.rfind('}')+1]
                
            data = json.loads(json_str)
            validated_ar = ARPeerReviewScore(**data)
            
            ar_verdict = validated_ar.formal_verdict
            ar_score = validated_ar.score_logical + validated_ar.score_quantitative + validated_ar.score_fallacy
            ar_reason = validated_ar.rationale
            
            # Build a formatted markdown string for the UI/messages
            soft_warning_text = "\n".join(f"- {w}" for w in soft_grounding_warnings[:10]) if soft_grounding_warnings else "None"
            ar_review_text = f"### Formal Verdict\n**{ar_verdict}**\n\n**Mathematical Confidence Score:** {ar_score}\n\n**Soft Grounding Warnings:**\n{soft_warning_text}\n\n**Rationale:** {ar_reason}\n\n**Required Improvements:** {validated_ar.required_improvements}"
            break # Success
            
        except Exception as e:
            print(f"    -> [WARNING] AR JSON Parsing Failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                error_prompt = f"\n\n[SYSTEM ERROR - ATTEMPT {attempt+1} FAILED]\nYour previous response failed to parse as JSON. Error details: {e}\nCRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object. Do NOT output any text before or after the ```json block."
                user_prompt = base_user_prompt + error_prompt
            else:
                ar_review_text = f"[PARSING ERROR] Raw Output:\n{response}"
                ar_verdict = "Parsing Error"
                ar_score = 0.0
                ar_reason = "Failed to output JSON format."

    ar_review_text = append_references(ar_review_text, "ar", state)
    
    return {
        "ar_peer_review": ar_review_text,
        "ar_verdict": ar_verdict,
        "ar_confidence_score": ar_score,
        "ar_rejection_reason": ar_reason if ar_score < 80.0 else None,
        "revision_count": state.get("revision_count", 0) + (1 if ar_score < 80.0 else 0),
        "messages": [f"**AR Peer Review ({state['target_month']})**:\n{ar_review_text}"]
    }

# ==========================================
# Phase 3: Synthesis
# ==========================================
def build_agreement_matrix(state: dict) -> tuple:
    """Build deterministic continuous stance-score matrix and disagreement score.

    AR is kept in the matrix for auditability but excluded from D(pi), because it is
    a methodological gate rather than a strategic value-lens evaluator.
    """
    stances = state.get("agent_stances") or {}
    ordered = ["qa", "fa", "ms", "ba", "rc", "qa_rebuttal", "ar"]
    rows = []
    strategic_values = []
    for key in ordered:
        if key not in stances:
            continue
        row = dict(stances[key])
        row.setdefault("agent", key)
        if "stance_score" not in row and "stance" in row:
            row["stance_score"] = float(row.get("stance", 0))
            row.pop("stance", None)
        row["stance_score"] = round(float(row.get("stance_score", 0.0)), 3)
        rows.append(row)
        if key != "ar":
            strategic_values.append(row["stance_score"])

    if strategic_values:
        mean = sum(strategic_values) / len(strategic_values)
        disagreement = sum((v - mean) ** 2 for v in strategic_values) / len(strategic_values)
    else:
        disagreement = 0.0

    matrix = {
        "target_month": state.get("target_month"),
        "company": state.get("company_name"),
        "stance_score_semantics": {
            "-1.0": "strong skepticism / forecast viewed as implausible or strategically weak",
            "0.0": "mixed, neutral, or contested evidence",
            "+1.0": "strong support / forecast viewed as plausible and strategically strong"
        },
        "rows": rows,
        "strategic_disagreement_variance_D_pi": round(disagreement, 4),
        "strategic_stance_score_mean": round(sum(strategic_values) / len(strategic_values), 4) if strategic_values else 0.0,
        "excluded_from_D_pi": ["ar"],
    }
    return matrix, disagreement


async def moderator_node(state: DebateState) -> DebateState:
    print(f"  [Phase 3] MJ Agent is Moderator/Judge for {state['target_month']}...")
    
    # Strip all appended RAG reference blocks from the debate history before
    # passing to the Moderator. The Moderator should evaluate analytical
    # arguments only, not raw document lists.
    history = "\n\n".join([
        f"QA: {strip_references(state.get('qa_statement', ''))}",
        f"FA: {strip_references(state.get('fa_statement', ''))}",
        f"MS: {strip_references(state.get('ms_statement', ''))}",
        f"BA: {strip_references(state.get('ba_statement', ''))}",
        f"RC: {strip_references(state.get('rc_statement', ''))}",
        f"AR: {strip_references(state.get('ar_statement', ''))}",
        f"QA Rebuttal: {strip_references(state.get('qa_ba_debate', ''))}",
        f"AR Peer Review: {strip_references(state.get('ar_peer_review', ''))}"
    ])
    
    ar_verdict = state.get("ar_verdict", "N/A")
    ar_score = state.get("ar_confidence_score", "N/A")
    scored_stances = await score_all_agent_stances(state)
    state_for_matrix = dict(state)
    state_for_matrix["agent_stances"] = scored_stances
    agreement_matrix, disagreement_score = build_agreement_matrix(state_for_matrix)
    agreement_matrix_json = json.dumps(agreement_matrix, ensure_ascii=False, indent=2)
    
    sys_prompt = MJ_SYSTEM_PROMPT.format(
        company_name=state["company_name"], 
        common=get_common(state),
        target_month=state["target_month"]
    )
    json_template = '''```json
{
  "consensus_points": "Bullet points summarizing agreements...",
  "points_of_contention": "Bullet points summarizing disagreements...",
  "academic_peer_review_summary": "Summary text here...",
  "step_by_step_rationale": "Rationale text here...",
  "score_data_grounding": "insert float between 0.0 and 10.0",
  "score_logical_coherence": "insert float between 0.0 and 10.0",
  "score_risk_robustness": "insert float between 0.0 and 10.0",
  "score_scientific_validity": "insert float between 0.0 and 10.0"
}
```'''
    base_user_prompt = f"DEBATE HISTORY:\n{history}\n\n[SYSTEM ENFORCED AR PEER REVIEW RESULT]\n- AR Verdict: {ar_verdict}\n- AR Confidence Score: {ar_score}\n\nCRITICAL INSTRUCTION: Do NOT write a text report. You MUST output ONLY a valid JSON object matching the exact schema below. The 'score_' fields MUST be numbers (floats), NOT text.\n\n{json_template}"
    user_prompt = base_user_prompt

    max_retries = 3
    for attempt in range(max_retries):
        response = await call_vllm(sys_prompt, user_prompt, temperature=0.1)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Fallback if markdown tags are missing
                json_str = response[response.find('{'):response.rfind('}')+1]
                
            data = json.loads(json_str)
            parsed_score = ModeratorScore(**data)

            # Deterministic score caps: do not rely on the Moderator LLM to obey
            # scientific-validity caps when AR rejected or automated audit failed.
            ar_review_text_for_cap = str(state.get("ar_peer_review", ""))
            ar_verdict_lower = str(ar_verdict).lower()
            try:
                ar_score_float = float(ar_score)
            except Exception:
                ar_score_float = 0.0

            if "automated hard audit failure" in ar_review_text_for_cap.lower():
                parsed_score.score_scientific_validity = min(parsed_score.score_scientific_validity, 3.0)
                parsed_score.score_data_grounding = min(parsed_score.score_data_grounding, 5.0)
            elif "automated" in ar_review_text_for_cap.lower() and "failure" in ar_review_text_for_cap.lower():
                parsed_score.score_scientific_validity = min(parsed_score.score_scientific_validity, 4.0)
                parsed_score.score_data_grounding = min(parsed_score.score_data_grounding, 6.0)

            if ar_score_float < 80.0 or ar_verdict_lower.startswith("rejected") or "strongly rejected" in ar_verdict_lower:
                parsed_score.score_scientific_validity = min(parsed_score.score_scientific_validity, 4.0)
            elif "major revisions" in ar_verdict_lower:
                parsed_score.score_scientific_validity = min(parsed_score.score_scientific_validity, 7.0)

            if "Soft Grounding Warnings:" in ar_review_text_for_cap and "None" not in ar_review_text_for_cap:
                parsed_score.score_data_grounding = min(parsed_score.score_data_grounding, 7.0)
                parsed_score.score_logical_coherence = min(parsed_score.score_logical_coherence, 8.0)

            if any(not row.get("quality_ok", True) for row in agreement_matrix.get("rows", [])):
                parsed_score.score_logical_coherence = min(parsed_score.score_logical_coherence, 7.0)
                parsed_score.score_data_grounding = min(parsed_score.score_data_grounding, 7.0)
            
            total_score = (parsed_score.score_data_grounding + parsed_score.score_logical_coherence + parsed_score.score_risk_robustness + parsed_score.score_scientific_validity) / 4.0
            
            consensus_md = parsed_score.consensus_points
            contention_md = parsed_score.points_of_contention
            
            forecast_val = "N/A"
            ci_val = "N/A"
            var_val = "N/A"
            for line in state['forecast_data'].split('\n'):
                if "Target Value EXACT" in line: forecast_val = line.split(":")[-1].strip()
                elif "Calculated 95% Confidence Interval bounds for" in line: ci_val = line.split(":")[-1].strip()
                elif "Variance for " in line: var_val = line.split(":")[-1].strip()

            agreement_table_lines = ["| Agent | Stance score | Value lenses | Evidence summary | Quality |", "|---|---:|---|---|---|"]
            for row in agreement_matrix.get("rows", []):
                agent = row.get("agent", "N/A")
                score = float(row.get("stance_score", 0.0))
                lenses = ", ".join(row.get("value_lens", []))
                evidence = str(row.get("evidence_summary", "")).replace("\n", " ")[:180]
                quality = "OK" if row.get("quality_ok", True) else "Weak"
                agreement_table_lines.append(f"| {agent} | {score:.3f} | {lenses} | {evidence} | {quality} |")
            agreement_table = "\n".join(agreement_table_lines)
            stance_mean = agreement_matrix.get("strategic_stance_score_mean", 0.0)
                
            formatted_response = f"""### 1. Executive Debate Synthesis
**Consensus Points**:
{consensus_md}

**Points of Contention**:
{contention_md}

**Academic Peer-Review Summary**:
- {parsed_score.academic_peer_review_summary}

### 2. Final Justification Score
- **Forecast Value**: {forecast_val}
- **95% Confidence Interval**: {ci_val}
- **Variance**: {var_val}

**Scores:**
- **Data Grounding & Temporal Continuity**: {parsed_score.score_data_grounding:.1f}/10.0
- **Logical Coherence**: {parsed_score.score_logical_coherence:.1f}/10.0
- **Risk Robustness**: {parsed_score.score_risk_robustness:.1f}/10.0
- **Scientific & Methodological Validity**: {parsed_score.score_scientific_validity:.1f}/10.0

**Total Justification Score**: {total_score:.2f}/10.0

**Score Rationale**:
{parsed_score.step_by_step_rationale}

### 3. Agent Agreement Matrix & Strategic Uncertainty
{agreement_table}

**Mean stance score**: {stance_mean:.4f}  
**Disagreement D(pi)**: {disagreement_score:.4f}
"""
            break # Success
        except Exception as e:
            print(f"    [Moderator Error] JSON Parsing failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                error_prompt = f"\n\n[SYSTEM ERROR - ATTEMPT {attempt+1} FAILED]\nYour previous response failed to parse as JSON. Error details: {e}\nCRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object."
                user_prompt = base_user_prompt + error_prompt
            else:
                formatted_response = f"### [PARSING ERROR] Raw Output:\n{response}"
    
    return {
        "moderator_report": formatted_response,
        "agent_stances": scored_stances,
        "agreement_matrix": agreement_matrix_json,
        "disagreement_score": float(disagreement_score),
        "messages": [f"**Moderator/Judge ({state['target_month']})**:\n{formatted_response}"]
    }
