"""
evaluate_explanation_quality.py
================================

Evaluate multi-agent explanation quality from debate logs.

Outputs two complementary metrics:
1. Reasoning Quality (parsed from AR Peer Review scores already in logs)
   - logical_alignment
   - quantitative_grounding
   - fallacy_avoidance
   - reasoning_quality

2. Task Fulfillment (LLM-as-a-judge over each agent response)
   - role-specific checklist coverage in [0, 1]
   - judged per agent response, then aggregated

Usage examples:
    python Agent/Multi_agent_Debate/metric/evaluate_explanation_quality.py \
        --log-dir Agent/Multi_agent_Debate/logs/AAPL_2026-06-28_17-59 \
        --company AAPL \
        --output-dir Agent/Multi_agent_Debate/metric/results/AAPL

    python Agent/Multi_agent_Debate/metric/evaluate_explanation_quality.py \
        --log-dir Agent/Multi_agent_Debate/logs/NVDA_2026-06-28_17-59 \
        --company NVDA

python Agent/Multi_agent_Debate/metric/evaluate_explanation_quality.py --log-dir Agent/Multi_agent_Debate/logs/NVDA_2026-06-28_17-59 --company NVDA
python Agent/Multi_agent_Debate/metric/evaluate_explanation_quality.py --log-dir Agent/Multi_agent_Debate/logs/AAPL_2026-06-28_17-59 --company AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


VLLM_URL = os.getenv("VLLM_CHAT_URL", "http://localhost:8001/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "casperhansen/deepseek-r1-distill-qwen-7b-awq")

AGENT_LABELS = ["QA", "FA", "MS", "RC", "AR", "BA", "QA Agent Rebuttal"]
AGENT_KEY_MAP = {
    "QA": "qa",
    "FA": "fa",
    "MS": "ms",
    "RC": "rc",
    "AR": "ar",
    "BA": "ba",
    "QA Agent Rebuttal": "qa_rebuttal",
}

ROLE_CHECKLISTS = {
    "qa": [
        "interprets forecast value, MoM change, confidence interval, and variance",
        "uses forecast data as the numerical source rather than stock price or revenue as AFCI",
        "links quantitative trend to business context or RAG evidence",
        "acknowledges uncertainty or limitations when appropriate",
    ],
    "fa": [
        "identifies product, partnership, market, or business-fundamental events",
        "explains product-market or technology implications for AI competitiveness",
        "connects qualitative evidence to AFCI direction",
        "states a clear fundamental verdict with caveats when needed",
    ],
    "ms": [
        "identifies a macro, sector, capex, supply-chain, or competitive-cycle driver",
        "explains how the driver affects AFCI as context rather than equating AFCI with capex",
        "includes competitor or market-structure positioning when relevant",
        "states tailwind/headwind implications with uncertainty",
    ],
    "rc": [
        "identifies regulatory, legal, compliance, or IP risks relevant to the firm",
        "explains a causal pathway from risk to AI competitiveness or AFCI",
        "distinguishes documented risks from general speculation",
        "states the severity or direction of regulatory impact",
    ],
    "ba": [
        "stress-tests the forecast with downside risks or model blind spots",
        "identifies at least one concrete risk mechanism or alternative scenario",
        "connects downside reasoning to AFCI plausibility or uncertainty",
        "avoids merely repeating bullish arguments without critique",
    ],
    "ar": [
        "evaluates methodology, stationarity, uncertainty, or validation rigor",
        "critiques confidence interval, overfitting, distribution shift, or reproducibility",
        "recommends appropriate methodological checks or baselines",
        "provides an academic-style verdict or limitation assessment",
    ],
    "qa_rebuttal": [
        "directly responds to BA/RC critique or AR feedback",
        "uses forecast data for numerical reasoning",
        "uses or discusses RAG evidence when making concrete business claims",
        "revises, defends, or concedes the forecast in a clear final position",
    ],
}


async def call_vllm(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(VLLM_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"].get("content", "")
                        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    text = await response.text()
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"vLLM HTTP {response.status}: {text[:500]}")
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def strip_references(text: str) -> str:
    return re.sub(r"\n\n### RAG Documents Referenced by .*?\n.*", "", text, flags=re.DOTALL).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_block(text: str, label: str) -> Optional[str]:
    """Extract one message block by label, e.g. '**QA Agent'."""
    if label == "QA":
        pattern = r"\*\*QA Agent \([^)]*\)\*\*:\n(.*?)(?=\n-{20,}|\n\*\*[A-Z][A-Za-z ]+Agent|\Z)"
    elif label == "QA Agent Rebuttal":
        pattern = r"\*\*QA Agent Rebuttal \([^)]*\)\*\*:\n(.*?)(?=\n-{20,}|\n\*\*[A-Z][A-Za-z ]+Agent|\Z)"
    else:
        pattern = rf"\*\*{re.escape(label)} Agent \([^)]*\)\*\*:\n(.*?)(?=\n-{{20,}}|\n\*\*[A-Z][A-Za-z ]+Agent|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return None
    # For QA rebuttal, use final one if multiple revisions exist.
    return strip_references(matches[-1])


def extract_all_agent_blocks(text: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    for label in AGENT_LABELS:
        block = extract_block(text, label)
        if block:
            blocks[AGENT_KEY_MAP[label]] = block
    return blocks


def parse_final_ar_scores(text: str) -> Dict[str, Any]:
    """Parse final AR Peer Review block and normalize component scores.

    If component scores are not present in the human-readable log, only total AR score
    is returned and component fields are left blank.
    """
    modpos = text.rfind("**Moderator/Judge")
    pre = text[:modpos] if modpos != -1 else text
    ar_blocks = re.findall(r"\*\*AR Peer Review \([^)]*\)\*\*:\n(.*?)(?=\n-{20,}|\n\*\*Moderator/Judge|\Z)", pre, flags=re.DOTALL)
    final_block = ar_blocks[-1] if ar_blocks else ""

    total_scores = [float(x) for x in re.findall(r"\*\*Mathematical Confidence Score:\*\*\s*([0-9.]+)", final_block)]
    ar_score = total_scores[-1] if total_scores else None

    # Component scores may appear only in raw JSON parse logs if preserved.
    def find_float(keys: List[str]) -> Optional[float]:
        for key in keys:
            m = re.search(rf"[\"']?{key}[\"']?\s*[:=]\s*([0-9.]+)", final_block, flags=re.I)
            if m:
                return float(m.group(1))
        return None

    score_logical = find_float(["score_logical", "logical"])
    score_quant = find_float(["score_quantitative", "quantitative"])
    score_fallacy = find_float(["score_fallacy", "fallacy"])

    if ar_score is not None and score_logical is None and score_quant is None and score_fallacy is None:
        # Components unavailable in formatted log. Preserve total as reasoning quality;
        # leave component-specific normalized scores blank to avoid fabrication.
        reasoning_quality = ar_score / 100.0
    elif None not in (score_logical, score_quant, score_fallacy):
        reasoning_quality = (score_logical + score_quant + score_fallacy) / 100.0
    else:
        reasoning_quality = None

    return {
        "ar_score": ar_score,
        "score_logical_raw": score_logical,
        "score_quantitative_raw": score_quant,
        "score_fallacy_raw": score_fallacy,
        "logical_alignment": None if score_logical is None else score_logical / 40.0,
        "quantitative_grounding": None if score_quant is None else score_quant / 40.0,
        "fallacy_avoidance": None if score_fallacy is None else score_fallacy / 20.0,
        "reasoning_quality": reasoning_quality,
        "ar_block": final_block.strip(),
    }


async def judge_task_fulfillment(agent_key: str, agent_text: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    checklist = ROLE_CHECKLISTS.get(agent_key, [])
    checklist_text = "\n".join(f"- {item}" for item in checklist)
    system_prompt = """You are a strict but fair evaluator of multi-agent analysis quality.
Your task is to judge whether one agent response fulfills its assigned analytical role.
Do not judge whether the forecast itself is true. Judge only task fulfillment.
Return ONLY valid JSON. No markdown.
"""
    user_prompt = f"""Agent role: {agent_key}

Role-specific checklist:
{checklist_text}

Agent response:
{agent_text[:5000]}

Return JSON with this schema:
{{
  "task_fulfillment": <float from 0.0 to 1.0>,
  "covered_items": ["short labels for checklist items satisfied"],
  "missing_items": ["short labels for checklist items missing or weak"],
  "rationale": "one concise sentence"
}}
"""
    async with semaphore:
        raw = await call_vllm(system_prompt, user_prompt, temperature=0.0)
    try:
        m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.I)
        js = m.group(1).strip() if m else raw[raw.find("{"):raw.rfind("}") + 1]
        data = json.loads(js)
        val = float(data.get("task_fulfillment", 0.0))
        data["task_fulfillment"] = max(0.0, min(1.0, val))
        return data
    except Exception as e:
        return {
            "task_fulfillment": 0.0,
            "covered_items": [],
            "missing_items": checklist,
            "rationale": f"judge_parse_failure: {e}; raw={raw[:300]}",
        }


async def evaluate_log_file(path: Path, company: str, semaphore: asyncio.Semaphore) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = read_text(path)
    month_match = re.search(r"Target Month:\s*(\d{4}-\d{2})", text)
    month = month_match.group(1) if month_match else path.stem[-7:]
    agent_blocks = extract_all_agent_blocks(text)
    ar = parse_final_ar_scores(text)

    async def eval_one(agent_key: str, block: str) -> Dict[str, Any]:
        judged = await judge_task_fulfillment(agent_key, block, semaphore)
        return {
            "company": company,
            "month": month,
            "agent": agent_key,
            "task_fulfillment": judged["task_fulfillment"],
            "task_rationale": judged.get("rationale", ""),
            "covered_items": "; ".join(judged.get("covered_items", [])),
            "missing_items": "; ".join(judged.get("missing_items", [])),
        }

    rows: List[Dict[str, Any]] = await asyncio.gather(
        *(eval_one(agent_key, block) for agent_key, block in agent_blocks.items())
    )

    summary = {
        "company": company,
        "month": month,
        "n_agents": len(rows),
        "avg_task_fulfillment": statistics.mean([r["task_fulfillment"] for r in rows]) if rows else None,
        "ar_score": ar["ar_score"],
        "reasoning_quality": ar["reasoning_quality"],
        "logical_alignment": ar["logical_alignment"],
        "quantitative_grounding": ar["quantitative_grounding"],
        "fallacy_avoidance": ar["fallacy_avoidance"],
    }
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def aggregate_agent_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for r in rows:
        grouped.setdefault(r["agent"], []).append(float(r["task_fulfillment"]))
    return [
        {"agent": agent, "avg_task_fulfillment": statistics.mean(vals), "n": len(vals)}
        for agent, vals in sorted(grouped.items())
    ]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, help="Directory containing COMPANY_YYYY-MM.txt logs")
    parser.add_argument("--company", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--months", nargs="*", default=None, help="Optional months, e.g. 2026-01 2026-02")
    parser.add_argument("--concurrency", type=int, default=6, help="Maximum concurrent LLM judge calls")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    company = args.company.upper()
    output_dir = Path(args.output_dir) if args.output_dir else Path("Agent/Multi_agent_Debate/metric/results") / company

    if args.months:
        files = [log_dir / f"{company}_{m}.txt" for m in args.months]
    else:
        files = sorted(log_dir.glob(f"{company}_2026-*.txt"))

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def eval_file(path: Path) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        if not path.exists():
            print(f"[skip missing] {path}")
            return None
        print(f"[eval] {path.name}")
        return await evaluate_log_file(path, company, semaphore)

    results = await asyncio.gather(*(eval_file(path) for path in files))

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for result in results:
        if result is None:
            continue
        rows, summary = result
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(output_dir / "task_fulfillment_per_agent.csv", all_rows)
    write_csv(output_dir / "explanation_quality_by_month.csv", summaries)
    write_csv(output_dir / "task_fulfillment_by_agent.csv", aggregate_agent_summary(all_rows))
    write_json(output_dir / "explanation_quality_summary.json", {
        "company": company,
        "n_months": len(summaries),
        "avg_task_fulfillment": statistics.mean([s["avg_task_fulfillment"] for s in summaries if s["avg_task_fulfillment"] is not None]) if summaries else None,
        "avg_reasoning_quality": statistics.mean([s["reasoning_quality"] for s in summaries if s["reasoning_quality"] is not None]) if summaries else None,
        "avg_ar_score": statistics.mean([s["ar_score"] for s in summaries if s["ar_score"] is not None]) if summaries else None,
        "months": summaries,
    })
    print(f"[done] wrote metrics to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
