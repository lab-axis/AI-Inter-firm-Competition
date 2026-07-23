import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import argparse
from dotenv import load_dotenv

# 역할: 수집된 SEC 보고서(Business, Risk, MD&A 통합본)를 분석하여 9가지 핵심 AI 섹터 노출도를 계산한다.
#       BGE(Big-data General Embedding) 모델을 사용하여 문맥적 의미를 파악한다.
#       각 기업별로 2013-09부터 2025-12까지 분기별(Quarterly: 03, 06, 09, 12월) 시계열 그리드를 구축하고
#       결측치는 분기 단위 전진채움(Forward-Fill)을 적용하여 완전한 스코어링 테이블을 완성한다.
# 결과: data/2_topic/topic_vectors.csv

TARGET_FIRMS = [
    'MSFT', 'GOOGL', 'AMZN', 'META', 'AAPL',
    'NVDA', 'AMD', 'AVGO', 'QCOM', 'INTC', 'ADI',
    'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'TXN',
    'ADBE', 'CRM', 'ORCL', 'IBM', 'SAP', 'NOW', 'INTU', 'PANW', 'ADSK',
    'CSCO', 'STX', 'TSLA'
]

# 9대 AI 비즈니스 섹터 정의 (OECD AI Taxonomy 반영)
SECTORS = {
    "expo_topic_semiconductors": "Semiconductors, AI hardware accelerators, GPUs, TPUs, NPUs, and manufacturing hardware (OECD D5).",
    "expo_topic_cloud": "Cloud computing infrastructure, IaaS, distributed computing, and AI training clusters (OECD D5).",
    "expo_topic_software": "Enterprise software, SaaS, B2B applications, LLM APIs, and Foundation models (OECD D4).",
    "expo_topic_hardware": "Consumer electronics, smartphones, PCs, wearables, and Edge AI devices (OECD D5).",
    "expo_topic_advertising": "Digital advertising, ad-tech, targeted marketing, and recommendation engines (OECD D4).",
    "expo_topic_social": "Social media networks, user engagement platforms, and social graph analysis.",
    "expo_topic_ecommerce": "E-commerce, online retail, and automated supply chain optimization (OECD D4).",
    "expo_topic_data": "Data analytics, big data integration, and predictive intelligence software (OECD D3/D4).",
    "expo_topic_auto": "Autonomous vehicles, computer vision, robotics, and industrial automation (OECD D4)."
}

def refine_sec_text(text: str) -> str:
    """SEC 보고서 특유의 노이즈와 법률적 상용구(Boilerplate)를 정밀하게 제거."""
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    
    legal_boilerplate = [
        'million', 'billion', 'quarter', 'fiscal', 'ended', 'months', 'company', 
        'approximately', 'including', 'primarily', 'operations', 'financial',
        'result', 'results', 'certain', 'various', 'subject', 'pursuant', 'provisions',
        'risks', 'risk', 'factors', 'uncertainties', 'statements', 'forward-looking',
        'believe', 'expect', 'anticipate', 'estimate', 'plan', 'potential', 'significant',
        'adversely', 'impact', 'material', 'could', 'may', 'should', 'would', 'might'
    ]
    pattern = re.compile(r'\b(' + '|'.join(legal_boilerplate) + r')\b', re.IGNORECASE)
    text = pattern.sub(' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_transcripts(data_dir: str) -> pd.DataFrame:
    """수집된 텍스트 파일을 로드하고 메타데이터(티커, 날짜)를 추출함."""
    search_pattern = os.path.join(data_dir, "**/*.txt")
    files = glob.glob(search_pattern, recursive=True)
    records = []

    print(f"\nLoading and Preprocessing SEC Transcripts…")
    print(f"  Target Directory: {data_dir}")
    print(f"  Found {len(files)} files to process.")

    for i, f in enumerate(files):
        basename = os.path.basename(f)
        name_parts = basename.replace(".txt", "").split("_")

        if len(name_parts) >= 3:
            firm_id = name_parts[0]
            date_ym = name_parts[2][:7] # YYYY-MM

            with open(f, 'r', encoding='utf-8') as file:
                raw_text = file.read()

            if not raw_text.strip(): continue

            clean_txt = refine_sec_text(raw_text)
            records.append({"firm_id": firm_id, "date": date_ym, "text": clean_txt})
            
        if (i+1) % 100 == 0:
            print(f"    Processed {i+1}/{len(files)} documents...")

    df = pd.DataFrame(records)
    print(f"  Successfully loaded {len(df)} documents.")
    return df

def run_zero_shot_nlp(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """임베딩 모델을 통해 AI 섹터별 노출도를 계산함."""
    print(f"\nInitializing NLP Model ({model_name})…")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    model = SentenceTransformer(model_name, device=device)

    # 섹터 정의 인코딩
    sector_keys = list(SECTORS.keys())
    sector_embeddings = model.encode(list(SECTORS.values()), convert_to_tensor=True, show_progress_bar=False)

    print(f"  Analysing documents (Batch processing on {device.upper()})…")
    BATCH_SIZE = 8
    doc_embeddings_list = []
    num_docs = len(df)

    for i in range(0, num_docs, BATCH_SIZE):
        batch_texts = df['text'].tolist()[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings_list.append(batch_embeddings)
        
        if (i // BATCH_SIZE + 1) % 20 == 0:
            print(f"    Progress: {i + len(batch_texts)}/{num_docs} documents encoded.")
        
        if device == 'cuda': torch.cuda.empty_cache()

    doc_embeddings = torch.cat(doc_embeddings_list, dim=0)

    # 코사인 유사도 계산 및 정규화
    print("  Finalizing exposure vectors…")
    scores = util.cos_sim(doc_embeddings, sector_embeddings).cpu().numpy()
    scores = np.clip(scores, 0, None)
    
    row_sums = scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_scores = scores / row_sums

    for i, key in enumerate(sector_keys):
        df[key] = norm_scores[:, i].round(6)

    return df

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data_preparation/data/1_transcripts")
    parser.add_argument("--out-csv", type=str, default="../data_preparation/data/2_topic/topic_vectors.csv")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--start-month", type=str, default="2013-09")
    parser.add_argument("--end-month", type=str, default="2025-12")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    df_raw = load_transcripts(args.data_dir)
    
    if not df_raw.empty:
        # 1. 문서별 Exposure 점수 도출
        df_final = run_zero_shot_nlp(df_raw, args.model)
        df_final.drop(columns=['text'], inplace=True)
        
        # 1.5 비표준 회계연도(Fiscal Year)를 가진 기업들(예: NVDA, AVGO 등)의 공시 일자를 표준 분기 종료월(03, 06, 09, 12월)로 캘린더 매핑
        def map_date_to_quarter_end(date_str):
            y, m = map(int, date_str.split('-'))
            if m in [1, 2, 3]:
                return f"{y:04d}-03"
            elif m in [4, 5, 6]:
                return f"{y:04d}-06"
            elif m in [7, 8, 9]:
                return f"{y:04d}-09"
            else:
                return f"{y:04d}-12"
        
        df_final['q_date'] = df_final['date'].apply(map_date_to_quarter_end)
        
        # 동일 분기 내 중복 공시 데이터가 매핑될 경우 평균 노출도로 정규화
        t_cols = list(SECTORS.keys())
        df_final_grouped = df_final.groupby(['firm_id', 'q_date'])[t_cols].mean().reset_index()
        df_final_grouped.rename(columns={'q_date': 'date'}, inplace=True)
        
        # 2. 2013-09부터 2025-12까지 분기별(Quarterly: 03, 06, 09, 12월) 시계열 그리드 생성
        # freq='Q'는 3, 6, 9, 12월의 말일을 반환하며, 포맷팅 시 정확히 YYYY-MM이 됨
        target_dates = pd.date_range(start=f'{args.start_month}-01', end=f'{args.end_month}-31', freq='QE').strftime('%Y-%m').tolist()
        
        grid = []
        for firm in TARGET_FIRMS:
            for d in target_dates:
                grid.append({'firm_id': firm, 'date': d})
        grid_df = pd.DataFrame(grid)
        
        # 3. 분기 그리드에 스코어링 결과 Outer Join (역사적/이전 공시의 ffill 연산을 위해 전체 기간 유지)
        print("\nMerging sparse NLP scores into the quarterly timeline...")
        df_merged = pd.merge(grid_df, df_final_grouped, on=['firm_id', 'date'], how='outer')
        
        # 4. 시간순 정렬 및 분기 단위 전진채움(Forward-Fill) 처리
        print("Applying quarterly forward-fill (ffill) grouping by firm...")
        df_merged.sort_values(by=['firm_id', 'date'], inplace=True)
        t_cols = list(SECTORS.keys())
        df_merged[t_cols] = df_merged.groupby('firm_id')[t_cols].ffill().fillna(0.0)
        
        # 4.5 그리드 범위(2013-09 ~ 2025-12) 이외의 역사적 임시 행들 필터링하여 제외
        df_merged = df_merged[df_merged['date'].isin(target_dates)]
        
        # 5. 최종 데이터 정렬 저장
        df_merged.sort_values(by=['firm_id', 'date'], inplace=True)
        df_merged.to_csv(args.out_csv, index=False)
        print(f"\nQuarterly topic vectors sorted and saved to {args.out_csv}")
        print(f"  Shape: {df_merged.shape} (Expected: {len(TARGET_FIRMS) * len(target_dates)} rows)")
    else:
        print("No transcripts found. Please check data/1_transcripts folder.")
