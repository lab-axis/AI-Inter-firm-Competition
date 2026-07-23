"""
temporal_rag_agent.py
=====================
Temporal PathRAG 에이전트 — B-MTGNN 예측 근거 설명을 위한 추론 인터페이스

[아키텍처]
  TemporalPathRAGRetriever
    → 인과 경로 (paths) + 시간 격리 벡터 청크 (chunks) 수집
  Path-based Prompt 조립
    → 경로 다이어그램 + 근거 텍스트를 구조화된 프롬프트로 패킹
  vLLM Chat Completions API 호출 (OpenAI 호환)
    → /v1/chat/completions — Ollama 포맷과 완전히 다름

[LLM 서빙 변경 내역]
  - Before: Ollama /api/generate ({"model":..., "prompt":..., "stream": false})
  - After : vLLM  /v1/chat/completions (OpenAI 호환 messages 포맷)
    → 모델: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    → 논문 방법론: "언어 모델은 vLLM을 통해 DeepSeek-R1-Distill-Qwen-7B를
                   OpenAI 호환 Chat Completions API 형태로
                   로컬 서빙하였으며, reasoning 필드를 통해 CoT 사고 과정을 투명하게 기록하였다."

[vLLM 서빙 명령어]
  vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8001 --host 0.0.0.0 --gpu-memory-utilization 0.75 --trust-remote-code

[Look-ahead Bias 방지]
  System Prompt에 cutoff_date 시간 제약을 명시적으로 주입.
  vLLM SLM이 학습 데이터에서 cutoff 이후의 지식을 혼입하지 못하도록
  강력한 지시를 포함시킨다.
"""

import os
import json
import re
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv
try:
    from temporal_rag_retriever import TemporalPathRAGRetriever, TemporalRAGRetriever
except ImportError:
    from engine.temporal_rag_retriever import TemporalPathRAGRetriever, TemporalRAGRetriever


class TemporalRAGAgent:
    """
    PathRAG + vLLM 기반 시간 인식 추론 에이전트.

    reason() 호출 시 수행 단계:
      ① TemporalPathRAGRetriever.retrieve() — 경로 + 청크 인출 (TBR 보장)
      ② _build_path_based_prompt()          — PathRAG 경로 다이어그램 포함 프롬프트 조립
      ③ _call_vllm_chat()                   — vLLM Chat Completions API 호출
      ④ 결과 패키징 (재현성을 위해 cutoff_ts 포함)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # System Prompt 템플릿
    # ──────────────────────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """\
You are an expert financial analyst AI specializing in AI/IT sector corporate research.

══════════════════════════════════════════════════════
CRITICAL TEMPORAL CONSTRAINT (Look-ahead Bias Prevention)
══════════════════════════════════════════════════════
• Information cutoff : {cutoff_date}
• ALL context documents were published BEFORE {cutoff_date}
• You MUST NOT reference any events, reports, or data after {cutoff_date}
• This constraint is MANDATORY for academic causal validity and reproducibility

══════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════
Using ONLY the provided context (relational path map + supporting articles),
thoroughly answer the user's query.

RESPONSE REQUIREMENTS:
1. Ground every claim in the context (cite by [Doc N] or publication date)
2. If relational paths are provided, use them to explain causal connections
3. Explicitly state if context is insufficient — do NOT hallucinate
4. Distinguish confirmed facts (context) from analytical inferences
5. Directly address the user's query. If they ask for news articles, list them; if they ask for analysis, provide it.

══════════════════════════════════════════════════════
TEMPORAL WEIGHTING & RECENCY FOCUS
══════════════════════════════════════════════════════
• The provided context spans multiple years. While historical context is valid,
  you MUST heavily prioritize the MOST RECENT events leading up to the {cutoff_date}.
• Analyze the evolution of events and explicitly discuss the recent shifts or 
  catalysts that occurred just before the cutoff date.
• Do NOT restrict your analysis solely to older historical data (e.g., 2018-2019) 
  if more recent developments are present in the context. Focus on the timeline trajectory.

══════════════════════════════════════════════════════
CONTEXT (all published before {cutoff_date})
══════════════════════════════════════════════════════
{context_block}
"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str       = None,
        vllm_chat_url: str   = "http://localhost:8001/v1/chat/completions",
        vllm_embed_url: str  = "http://localhost:8000/v1/embeddings",
        llm_model: str       = "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        embed_model: str     = "jinaai/jina-embeddings-v3",
        top_k: int           = 5,
        temperature: float   = 0.1,
        max_tokens: int      = 2048,
    ):
        """
        Args:
            neo4j_uri      : Neo4j Bolt URI
            neo4j_user     : Neo4j 사용자명
            neo4j_password : Neo4j 비밀번호
            vllm_chat_url  : vLLM Chat Completions 엔드포인트 (기본: port 8001)
                             vllm serve casperhansen/deepseek-r1-distill-qwen-7b-awq \
                                 --port 8001
            vllm_embed_url : vLLM 임베딩 엔드포인트 (기본: port 8000)
                             vllm serve jinaai/jina-embeddings-v3 --port 8000 --trust-remote-code
            llm_model      : 추론에 사용할 모델명
                             (기본: "casperhansen/deepseek-r1-distill-qwen-7b-awq")
            embed_model    : 임베딩 모델명 (RAG 빌더와 동일 필수)
            top_k          : 최대 검색 청크 수
            temperature    : LLM 추론 온도 (0.1 = 결정론적, 학술 재현성 최대화)
            max_tokens     : 최대 출력 토큰 수
        """
        self.retriever = TemporalPathRAGRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            vllm_embed_url=vllm_embed_url,
            embed_model=embed_model,
        )
        self.vllm_chat_url = vllm_chat_url
        self.llm_model     = llm_model
        self.top_k         = top_k
        self.temperature   = temperature
        self.max_tokens    = max_tokens

    def close(self):
        self.retriever.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 핵심 공개 메서드
    # ──────────────────────────────────────────────────────────────────────────

    def reason(
        self,
        query: str,
        cutoff_date: str,
        target_firms: Optional[List[str]] = None,
        gnn_prediction: Optional[Dict]    = None,
        analysis_period: Optional[str]    = None,
    ) -> Dict:
        """
        Temporal PathRAG 기반 추론.

        Args:
            query          : 자연어 추론 쿼리
            cutoff_date    : 시간 컷오프 (예: "2025-01")
            target_firms   : 기업 ticker 필터 (None → 쿼리 자동 추출)
            gnn_prediction : B-MTGNN 예측값 딕셔너리 (프롬프트 상단 삽입)
            analysis_period: 추론 대상 기간 설명 (예: "Q1 2025")

        Returns:
            {
              "answer"           : str   — vLLM SLM 최종 분석 보고서
              "reasoning_process": str   — DeepSeek-R1 CoT 사고 과정 (reasoning 필드)
              "paths"            : list  — PathRAG 인과 경로
              "chunks"           : list  — TBR 벡터 청크
              "mode"             : str   — "pathrag" | "vector_fallback"
              "cutoff_date"      : str
              "cutoff_ts"        : int   — Unix timestamp (논문 재현성)
              "gnn_prediction"   : dict
              "entities_found"   : list
              "prompt_tokens"    : int   — 전송된 프롬프트 토큰 수 (근사)
            }
        """
        print(f"\n{'='*65}")
        print(f"  [Temporal Agent] 추론 시작")
        print(f"  cutoff: {cutoff_date} | 기업: {target_firms or '자동추출'}")
        print(f"{'='*65}")

        # ① PathRAG 검색 (경로 + 청크)
        retrieved = self.retriever.retrieve(
            query=query,
            cutoff_date=cutoff_date,
            top_k=self.top_k,
            target_firms=target_firms,
        )

        # ② Path-based Prompt 조립
        context_block = self._build_path_based_prompt(
            retrieved, gnn_prediction, cutoff_date
        )
        system_msg = self.SYSTEM_PROMPT.format(
            analysis_period=analysis_period or cutoff_date,
            cutoff_date=cutoff_date,
            context_block=context_block,
        )

        # ③ vLLM Chat Completions API 호출 (answer + reasoning_process 동시 반환)
        answer, reasoning_process = self._call_vllm_chat(system_msg, query)

        return {
            "answer":            answer,
            "reasoning_process": reasoning_process,  # DeepSeek-R1 CoT 사고 과정
            "paths":             retrieved["paths"],
            "chunks":            retrieved["chunks"],
            "mode":              retrieved["mode"],
            "entities_found":    retrieved["entities_found"],
            "cutoff_date":       cutoff_date,
            "cutoff_ts":         retrieved["cutoff_ts"],
            "gnn_prediction":    gnn_prediction,
            "query":             query,
            "prompt_tokens":     len(system_msg.split()),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Path-based Prompt 조립
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_path_based_prompt(
        retrieved: Dict,
        gnn_prediction: Optional[Dict],
        cutoff_date: str,
    ) -> str:
        """
        PathRAG의 핵심: 인과 경로 다이어그램 + 근거 문서를 구조화된 포맷으로 조립.

        [Path-based Prompting 설계 원리 — arXiv:2502.14902]
          "By presenting the LLM with explicit relational paths rather than
           flat text chunks, we guide the model's attention toward the
           causal chain of events, significantly reducing hallucination."

        출력 구조:
          [B-MTGNN 예측값]
          [기업 간 인과 관계 경로 다이어그램]
          [각 경로의 근거 텍스트]
          [보완 벡터 청크]
        """
        lines = []

        # ── B-MTGNN 예측값 블록 ─────────────────────────────────────────────
        if gnn_prediction:
            lines.append("━━━ B-MTGNN Quantitative Prediction (to be explained) ━━━")
            for k, v in gnn_prediction.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # ── PathRAG 인과 경로 다이어그램 ────────────────────────────────────
        paths = retrieved.get("paths", [])
        if paths:
            lines.append(f"━━━ Relational Path Map (all edges before {cutoff_date}) ━━━")
            lines.append("  [PathRAG: Flow-based Pruned Causal Graph]")
            lines.append("")
            for i, path in enumerate(paths, 1):
                flow = path.get("flow_score", 0.0)
                lines.append(
                    f"  PATH {i} [flow={flow:.2f}]: {path['path_str']}"
                )
                # 각 경로의 근거 청크 (2개 이하)
                for j, chunk in enumerate(path.get("chunks", []), 1):
                    lines.append(
                        f"    └─ Evidence {j} [{chunk['published_date']}] "
                        f"{chunk['article_title']}"
                    )
                    lines.append(
                        f"       {chunk['text'][:400]}"
                        f"{'...' if len(chunk['text']) > 400 else ''}"
                    )
                lines.append("")
        else:
            lines.append("  [No direct relational paths found - Vector Fallback active]")
            lines.append("")

        # ── VectorRAG 보완 청크 ──────────────────────────────────────────────
        chunks = retrieved.get("chunks", [])
        if chunks:
            lines.append(
                f"━━━ Supporting Documents ({len(chunks)} chunks, "
                f"all before {cutoff_date}) ━━━"
            )
            for i, chunk in enumerate(chunks, 1):
                score = f"{chunk['score']:.4f}" if chunk.get("score") else "N/A"
                lines.append(f"\n[Doc {i}]")
                lines.append(f"  Published  : {chunk['published_date']}")
                lines.append(f"  Relevance  : {score}")
                lines.append(f"  Source     : {chunk['article_title']}")
                lines.append(f"  URL        : {chunk['article_url']}")
                lines.append(
                    f"  Content    : {chunk['text'][:500]}"
                    f"{'...' if len(chunk['text']) > 500 else ''}"
                )
        else:
            lines.append("  [No supporting documents found within temporal constraint]")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # vLLM Chat Completions API 호출 (OpenAI 호환)
    # ──────────────────────────────────────────────────────────────────────────

    def _call_vllm_chat(self, system_prompt: str, user_query: str):
        """
        vLLM /v1/chat/completions API 호출 (OpenAI 호환 포맷).

        [DeepSeek-R1 Reasoning 파싱]
          --enable-reasoning --reasoning-parser deepseek_r1 플래그로 서빙된 경우,
          vLLM이 <think>...</think> 블록을 자동으로 파싱하여 두 필드로 분리:
            choices[0].message.reasoning → CoT 사고 과정 (투명한 추론 경로)
            choices[0].message.content   → 최종 분석 보고서

          둘 다 반환하여 논문에서 XAI 증거로 활용 가능.

        temperature=0.6 설정 이유 (DeepSeek-R1 권장):
          DeepSeek-R1 시리즈는 공식 문서상 temperature 0.5~0.7에서
          사고 과정의 다양성과 품질이 최적화됨 (0.1은 reasoning collapse 유발 가능).

        Returns:
            Tuple[str, str]: (final_answer, reasoning_process)
        """
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "stream":      False,
        }
        try:
            resp = requests.post(self.vllm_chat_url, json=payload, timeout=180)
            resp.raise_for_status()
            data    = resp.json()
            message = data["choices"][0]["message"]

            # DeepSeek-R1: reasoning 필드 (CoT 사고 과정) + content 필드 (최종 답변)
            final_answer      = message.get("content", "") or ""
            reasoning_process = message.get("reasoning", "") or ""

            # 구버전 vLLM Fallback: API 응답에 reasoning 필드가 없고 content 필드 내에 <think> 태그가 존재하는 경우 직접 파싱
            if not reasoning_process and "<think>" in final_answer:
                think_match = re.search(r"<think>(.*?)</think>", final_answer, re.DOTALL)
                if think_match:
                    reasoning_process = think_match.group(1).strip()
                    final_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
                    print(f"\n[DeepSeek-R1 Fallback] content 내 <think> 태그 파싱 성공 ({len(reasoning_process)}자)")

            if reasoning_process and not message.get("reasoning"):
                pass
            elif reasoning_process:
                print(f"\n[DeepSeek-R1 CoT] 사고 과정 {len(reasoning_process)}자 수신됨")

            return final_answer, reasoning_process

        except requests.exceptions.ConnectionError:
            err = (
                "[vLLM_UNAVAILABLE] vLLM 서버에 연결할 수 없습니다.\n"
                f"서버 주소: {self.vllm_chat_url}\n\n"
                "vLLM 서버 실행 명령어:\n"
                f"  vllm serve {self.llm_model} \\\ \n"
                "      --port 8001 --host 0.0.0.0 \\\ \n"
                "      --enable-reasoning --reasoning-parser deepseek_r1 \\\ \n"
                "      --gpu-memory-utilization 0.45\n\n"
                "[Context Only Mode]\n"
                "  result['paths']와 result['chunks']를 직접 참조하여 분석하세요."
            )
            return err, ""
        except requests.exceptions.HTTPError as e:
            return f"[HTTP 오류] {e}\n응답: {resp.text[:300]}", ""
        except KeyError as e:
            return f"[응답 파싱 오류] {e}\n전체 응답: {resp.text[:300]}", ""
        except Exception as e:
            return f"[vLLM 오류] {e}", ""


# ──────────────────────────────────────────────────────────────────────────────
# CLI 실행 예시
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Temporal Agent — B-MTGNN 예측 근거 설명 (vLLM 버전)"
    )
    parser.add_argument("--query",          type=str, required=True)
    parser.add_argument("--cutoff",         type=str, required=True,  help="예: '2025-01'")
    parser.add_argument("--firms",          type=str, default=None,   help="예: 'NVDA,TSM'")
    parser.add_argument("--period",         type=str, default=None,   help="예: 'Q1 2025'")
    parser.add_argument("--llm-model",      type=str, default="casperhansen/deepseek-r1-distill-qwen-7b-awq")
    parser.add_argument("--vllm-chat-url",  type=str, default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vllm-embed-url", type=str, default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--top-k",          type=int, default=5)
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--neo4j-uri",      type=str, default=None)
    parser.add_argument("--neo4j-user",     type=str, default=None)
    parser.add_argument("--neo4j-pass",     type=str, default=None)
    parser.add_argument("--no-llm",         action="store_true", help="LLM 없이 RAG 검색 결과만 출력")
    args = parser.parse_args()

    uri      = args.neo4j_uri  or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = args.neo4j_user or os.getenv("NEO4J_USER",     "neo4j")
    password = args.neo4j_pass or os.getenv("NEO4J_PASSWORD", "password123")
    firms    = [f.strip() for f in args.firms.split(",")] if args.firms else None

    # B-MTGNN 예측 샘플 (실제 연구에서는 모델 출력값으로 교체)
    sample_gnn_prediction = {
        "target_firms":   firms or ["ALL"],
        "analysis_period": args.period or args.cutoff,
        "model":           "Bayesian-MTGNN (B-MTGNN)",
        "note":            "Replace with actual B-MTGNN output in production",
    }

    agent = TemporalRAGAgent(
        neo4j_uri=uri, neo4j_user=user, neo4j_password=password,
        vllm_chat_url=args.vllm_chat_url,
        vllm_embed_url=args.vllm_embed_url,
        llm_model=args.llm_model,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    result = agent.reason(
        query=args.query,
        cutoff_date=args.cutoff,
        target_firms=firms,
        gnn_prediction=sample_gnn_prediction,
        analysis_period=args.period,
    )

    print(f"\n{'='*65}")
    print("  [Temporal Agent] 추론 완료")
    print(f"{'='*65}")
    print(f"  검색 모드   : {result['mode'].upper()}")
    print(f"  컷오프      : {result['cutoff_date']} (ts: {result['cutoff_ts']})")
    print(f"  엔티티      : {result['entities_found']}")
    print(f"  경로 수     : {len(result['paths'])}개")
    print(f"  벡터 청크 수: {len(result['chunks'])}개")

    if result.get("reasoning_process"):
        print(f"\n[DeepSeek-R1 CoT 사고 과정]\n{result['reasoning_process'][:800]}...")
        print("  (전체 사고 과정은 result['reasoning_process']에서 확인 가능)")
    print(f"\n[LLM 최종 분석 보고서]\n{result['answer']}")

    print(f"\n[참조 경로 목록 — {result['cutoff_date']} 이전]")
    for i, path in enumerate(result["paths"], 1):
        print(f"  {i}. {path['path_str']}")

    print(f"\n[참조 문서 목록 — {result['cutoff_date']} 이전]")
    for i, ctx in enumerate(result["chunks"], 1):
        score = f"{ctx['score']:.4f}" if ctx.get("score") else "N/A"
        print(f"  {i}. [{ctx['published_date']}] (score={score}) {ctx['article_title']}")

    agent.close()
