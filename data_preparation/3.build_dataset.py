import os
import time
import calendar
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pytrends.request import TrendReq
from dotenv import load_dotenv

# 역할: 2014-2025년 (12년) 동안의 30개 정예 AI 핵심 기업 데이터를 수집한다.
# 결과: data/3_dataset/model_ready_panel.csv & temp/*.csv

load_dotenv()

# 0. AI value chain Company list
TARGET_FIRMS = [
    'MSFT', 'GOOGL', 'AMZN', 'META', 'AAPL',
    'NVDA', 'AMD', 'AVGO', 'QCOM', 'INTC', 'ADI',
    'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'TXN',
    'ADBE', 'CRM', 'ORCL', 'IBM', 'SAP', 'NOW', 'INTU', 'PANW', 'ADSK',
    'CSCO', 'STX', 'TSLA'
]

OPENALEX_INSTITUTIONS = {
    "MSFT": "ror:00d0nc645", "GOOGL": "ror:02e9yx751", "AMZN": "ror:04mv4n011",
    "META": "ror:01zbnvs85", "AAPL": "ror:059hsda18", "NVDA": "ror:03jdj4y14",
    "AMD": "ror:04kd6c783", "AVGO": "ror:035gt5s03", "QCOM": "ror:002zrf773",
    "INTC": "ror:01ek73717", "ADI": "ror:01545pm61", "TSM": "ror:02wx79d08",
    "ASML": "ror:01vxknj13", "AMAT": "ror:04h1q4c89", "LRCX": "ror:04gecbm52",
    "KLAC": "ror:02rqhpa98", "MU": "ror:02fv52296", "TXN": "ror:03vsmv677",
    "ADBE": "ror:059tvcg64", "CRM": "ror:057315g56", "ORCL": "ror:006c77m33",
    "IBM": "ror:05hh8d621", "SAP": "ror:03dsc8d33", "NOW": "ror:05xr0bc04",
    "INTU": "ror:049mrbr98", "PANW": "ror:01rn6rn86", "ADSK": "ror:00pkt4594",
    "CSCO": "ror:03yt1ez60", "STX": "ror:04p1xtv71", "TSLA": "ror:02kpcqm42"
}

EODHD_TICKERS = {f: [f"{f}.US"] for f in TARGET_FIRMS}
EODHD_TICKERS['GOOGL'] = ['GOOGL.US', 'GOOG.US']

OECD_AI_TAXONOMY = {
    'D3_Model': ['machine learning', 'deep learning', 'neural network', 'artificial intelligence'],
    'D4_Task': ['nlp', 'computer vision', 'generative ai', 'llm', 'gpt', 'transformer'],
    'D5_Implementation': ['gpu', 'tpu', 'ai chip', 'data center']
}
_ALL_AI_KEYWORDS = [kw for sublist in OECD_AI_TAXONOMY.values() for kw in sublist]

# EODHD 단어 가중치(Unigram) 매칭용 AI 키워드 세트 (다중 단어 구문 차단 및 매칭 누락 해결)
_EODHD_AI_UNIGRAMS = {
    'ai', 'openai', 'intelligence', 'neural', 'network', 'learning',
    'nlp', 'vision', 'generative', 'llm', 'gpt', 'transformer',
    'gpu', 'tpu', 'chip'
}

# 1. API Functions (OpenAlex, EODHD, Google Trends)
def fetch_openalex_paper_count(institution_id: str, year: int, month: int) -> int:
    if not institution_id: return 0
    # ROR ID를 OpenAlex 규격에 맞는 full URL 형식(https://ror.org/...)으로 변환
    ror_id = institution_id.replace("ror:", "https://ror.org/")
    last_day = calendar.monthrange(year, month)[1]
    pub_range = f"from_publication_date:{year}-{month:02d}-01,to_publication_date:{year}-{month:02d}-{last_day}"
    
    # lineage 대신 institutions.ror 필터를 사용하여 정밀 조회
    params = {
        'filter': f'institutions.ror:{ror_id},{pub_range}', 
        'per-page': 1, 
        'mailto': 'research@ai-company.com'
    }
    try:
        res = requests.get('https://api.openalex.org/works', params=params, timeout=15)
        if res.status_code == 200: 
            return res.json().get('meta', {}).get('count', 0)
    except: 
        pass
    return 0

def fetch_eodhd_news(ticker_list: list, year: int, month: int, api_key: str) -> dict:
    """EODHD의 공식 Financial News API(/api/news)를 조회하여 
       뉴스 개수, 감성 분석 점수를 가져온다."""
    fallback = {'news_count': 0, 'news_sentiment': 0.0}
    if not api_key or api_key == 'your_eodhd_api_key_here': return fallback
    last_day = calendar.monthrange(year, month)[1]
    from_date = f'{year}-{month:02d}-01'
    to_date = f'{year}-{month:02d}-{last_day}'
    
    for ticker in ticker_list:
        try:
            url = 'https://eodhd.com/api/news'
            params = {
                's': ticker,
                'from': from_date,
                'to': to_date,
                'limit': 1000,
                'api_token': api_key,
                'fmt': 'json'
            }
            res = requests.get(url, params=params, timeout=20)
            if res.status_code == 200:
                articles = res.json()
                if isinstance(articles, list) and len(articles) > 0:
                    polarities = []
                    for a in articles:
                        sent = a.get('sentiment')
                        if isinstance(sent, dict):
                            polarities.append(sent.get('polarity', 0.0))
                        elif isinstance(sent, (int, float)):
                            polarities.append(float(sent))
                            
                    news_count = len(articles)
                    avg_sentiment = round(float(np.mean(polarities)), 4) if polarities else 0.0
                    return {'news_count': news_count, 'news_sentiment': avg_sentiment}
        except: 
            pass
    return fallback

def fetch_eodhd_news_ai_exposure(ticker_list: list, year: int, month: int, api_key: str) -> float:
    """EODHD의 공식 단어 가중치 API(/api/news-word-weights)를 조회하여 AI 테마 노출 비중을 계산한다.
       All World 요금제 계정이므로 브래킷([]) 문자열 인코딩 문제 차단을 위해 쿼리 스트링 직접 빌드 방식을 적용한다."""
    if not api_key or api_key == 'your_eodhd_api_key_here': return 0.0
    last_day = calendar.monthrange(year, month)[1]
    from_date = f'{year}-{month:02d}-01'
    to_date = f'{year}-{month:02d}-{last_day}'
    
    for ticker in ticker_list:
        try:
            # 브래킷 URL 인코딩 버그 차단용 수동 쿼리 빌드
            url = 'https://eodhd.com/api/news-word-weights'
            query_str = f"s={ticker}&api_token={api_key}&fmt=json&filter[date_from]={from_date}&filter[date_to]={to_date}"
            full_url = f"{url}?{query_str}"
            
            res = requests.get(full_url, timeout=30)
            if res.status_code == 200:
                resp = res.json()
                if isinstance(resp, dict):
                    data = resp.get('data', {})
                    if isinstance(data, dict) and data:
                        total = sum(data.values())
                        if total > 0:
                            ai_w = sum(val for word, val in data.items() if word.lower() in _EODHD_AI_UNIGRAMS)
                            return round(ai_w / total, 4)
                        return 0.0
        except: 
            pass
    return 0.0

def fetch_google_trends_raw(keyword: str, timeframe: str) -> pd.Series:
    pytrends = TrendReq(hl='en-US', tz=360)
    for attempt in range(5):
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='')
            df = pytrends.interest_over_time()
            if not df.empty and keyword in df.columns:
                return df[keyword].resample('MS').mean().round(0).astype(int)
            break
        except Exception as e:
            if "429" in str(e):
                wait = 20 * (attempt + 1)
                print(f"    [Google Trends 429 Error] Rate limit hit for '{keyword}'. Waiting {wait}s and retrying ({attempt+1}/5)...")
                time.sleep(wait)
            else:
                print(f"    [Google Trends Error] Unexpected error for '{keyword}': {e}")
                break
    return pd.Series(dtype=float)

# 2. OECD Momentum Index (Cached in temp)
def get_oecd_momentum(start_month: str, end_month: str, cache_path: str) -> pd.Series:
    if os.path.exists(cache_path):
        print(f"  > Loading cached OECD AI Momentum from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.iloc[:, 0]
        
    print(f"\n[Global] ── Generating OECD AI Momentum Index…")
    s_dt = f"{start_month}-01"
    e_y, e_m = map(int, end_month.split('-'))
    e_dt = f"{end_month}-{calendar.monthrange(e_y, e_m)[1]}"
    
    results = []
    core_kws = _ALL_AI_KEYWORDS
    for kw in core_kws:
        print(f"    Fetching: {kw}")
        s = fetch_google_trends_raw(kw, f"{s_dt} {e_dt}")
        if not s.empty: results.append(s)
        time.sleep(3)
        
    if not results: return pd.Series(0, index=pd.date_range(s_dt, e_dt, freq='MS'))
    combined = pd.concat(results, axis=1).mean(axis=1).round(2)
    
    # temp 디렉토리 내부이므로 안심하고 저장하여 429 에러 방지
    combined.to_csv(cache_path)
    print(f"  > Saved OECD AI Momentum to cache: {cache_path}")
    return combined

# 3. Main Logic with Checkpoints
def collect_firm_data(firm: str, target_dates: list, eodhd_key: str, oecd_momentum: pd.Series, temp_dir: str, current_idx: int, total_count: int):
    temp_path = os.path.join(temp_dir, f"{firm}.csv")
    if os.path.exists(temp_path):
        print(f"  [{current_idx}/{total_count}] {firm} ── Already exists in temp. Skipping.")
        return
    
    print(f"  [{current_idx}/{total_count}] {firm} ── Starting collection…")
    y_start, y_end = int(target_dates[0][:4]), int(target_dates[-1][:4])
    s_dt, e_dt = f"{target_dates[0]}-01", f"{target_dates[-1]}-{calendar.monthrange(y_end, int(target_dates[-1][5:7]))[1]}"
    
    print(f"    - Fetching Google Trends (Firm Attention)…")
    gt_series = fetch_google_trends_raw(firm, f"{s_dt} {e_dt}")
    
    print(f"    - Fetching yfinance (Stock Price & Returns)…")
    hist_dict = {}
    try:
        hist = yf.Ticker(firm).history(start=f'{y_start-1}-12-01', end=f'{y_end+1}-01-31', interval='1mo')
        if not hist.empty:
            if hist.index.tz: hist.index = hist.index.tz_convert(None)
            hist['stock_return'] = hist['Close'].pct_change()
            hist['date_ym'] = hist.index.strftime('%Y-%m')
            hist_dict = hist.set_index('date_ym').to_dict('index')
            print(f"      - yfinance: Found {len(hist)} months of data.")
    except Exception as e:
        print(f"      - yfinance: Error fetching data - {e}")

    inst = OPENALEX_INSTITUTIONS.get(firm, '')
    tickers = EODHD_TICKERS.get(firm, [f'{firm}.US'])

    print(f"    - Fetching monthly API metrics (OpenAlex & EODHD News)…")
    recs = []
    for i, ym in enumerate(target_dates):
        if (i+1) % 48 == 0:
            print(f"      Progress: {ym} ({i+1}/{len(target_dates)} months done)")
            
        y, mo = int(ym[:4]), int(ym[5:7])
        stock = hist_dict.get(ym, {})
        dt_idx = pd.to_datetime(f"{ym}-01")
        
        # EODHD API 호출: 일반 뉴스 데이터(/api/news) 및 프리미엄 단어 가중치 데이터(/api/news-word-weights) 연동
        news_data = fetch_eodhd_news(tickers, y, mo, eodhd_key)
        ai_exposure = fetch_eodhd_news_ai_exposure(tickers, y, mo, eodhd_key)
        
        recs.append({
            'date': ym, 'firm_id': firm,
            'stock_return': stock.get('stock_return', np.nan),
            'paper_count': fetch_openalex_paper_count(inst, y, mo),
            'news_count': news_data['news_count'],
            'news_sentiment': news_data['news_sentiment'],
            'news_ai_exposure': ai_exposure,
            'gt_firm_attention': gt_series.get(dt_idx, np.nan),
            'gt_ai_momentum': oecd_momentum.get(dt_idx, np.nan)
        })
    
    firm_df = pd.DataFrame(recs)
    firm_df.to_csv(temp_path, index=False)
    print(f"    - {firm} data saved to '{temp_path}'.\n")
    time.sleep(1.5)

def calculate_afci(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCalculating Academic-Grade PIM-AFCI (Exponential Decay with 12M Half-Life)…")
    df['news_impact'] = df['news_count'] * df['news_sentiment']
    vars_to_scale = ['paper_count', 'news_impact', 'news_ai_exposure', 'gt_firm_attention', 'gt_ai_momentum']
    df[vars_to_scale] = df[vars_to_scale].fillna(0.0)
    
    # 1. 월별 횡단면 Min-Max 스케일링 수행
    df_scaled = df.copy()
    for col in vars_to_scale:
        min_series = df.groupby('date')[col].transform('min')
        max_series = df.groupby('date')[col].transform('max')
        range_series = max_series - min_series
        df_scaled[col] = np.where(range_series > 1e-8, (df[col] - min_series) / range_series, 0.0)
        
    # 2. 당월 AFCI 복합 지수 계산 (민맥스 스케일 피처 평균)
    df_scaled['AFCI_MinMax'] = df_scaled[vars_to_scale].mean(axis=1)
    
    # 3. 날짜와 기업 기준으로 피벗하여 지수적 감가상각(PIM) 재귀 연산 수행
    df_scaled.sort_values(by=['date', 'firm_id'], inplace=True)
    df_pivot = df_scaled.pivot(index='date', columns='firm_id', values='AFCI_MinMax').sort_index()
    
    # 반감기(Half-Life) 기반 월별 감가상각률(delta) 동적 계산 (예측 타겟 12개월 매칭)
    half_life_months = 12.0
    delta = 1.0 - (0.5 ** (1.0 / half_life_months)) # delta ≈ 0.056126 (5.61% decay/month)
    
    df_pim = pd.DataFrame(index=df_pivot.index, columns=df_pivot.columns, dtype=float)
    
    # 첫 달은 그대로 대입
    df_pim.iloc[0] = df_pivot.iloc[0]
    for t in range(1, len(df_pivot)):
        df_pim.iloc[t] = (1.0 - delta) * df_pim.iloc[t - 1] + df_pivot.iloc[t]
        
    # 4. 피벗된 PIM 값을 다시 기존 long-format dataframe인 df에 매핑 후 저장
    pim_series = df_pim.stack()
    pim_series.index.names = ['date', 'firm_id']
    pim_series.name = 'AFCI'

    if 'AFCI' in df.columns:
        df = df.drop(columns=['AFCI'])

    pim_df = pim_series.reset_index()
    pim_df['date'] = pd.to_datetime(pim_df['date']).dt.strftime('%Y-%m')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    df = pd.merge(df, pim_df, on=['date', 'firm_id'], how='left')
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-month', type=str, default='2014-01')
    parser.add_argument('--end-month', type=str, default='2025-12')
    parser.add_argument('--topic-csv', type=str, default='../data_preparation/data/2_topic/topic_vectors.csv')
    parser.add_argument('--out-csv', type=str, default='../data_preparation/data/3_dataset/data_set.csv')
    args = parser.parse_args()

    eodhd_key = os.getenv('EODHD_API_KEY', '')
    target_dates = pd.date_range(start=f'{args.start_month}-01', end=f'{args.end_month}-28', freq='MS').strftime('%Y-%m').tolist()
    
    temp_dir = '../data_preparation/data/3_dataset/temp'
    os.makedirs(temp_dir, exist_ok=True)
    oecd_cache_path = os.path.join(temp_dir, 'oecd_momentum.csv')
    
    print(f"=== Pipeline | {args.start_month} ~ {args.end_month} | Checkpoint Enabled ===")
    
    # 1. OECD 지수
    oecd_momentum = get_oecd_momentum(args.start_month, args.end_month, oecd_cache_path)
    
    # 2. 기업별 개별 수집 및 개별 CSV 저장
    total_firms = len(TARGET_FIRMS)
    for i, firm in enumerate(TARGET_FIRMS):
        collect_firm_data(firm, target_dates, eodhd_key, oecd_momentum, temp_dir, i+1, total_firms)

    # 3. 모든 개별 CSV 병합
    print("\nMerging all collected firm data…")
    all_files = [os.path.join(temp_dir, f"{f}.csv") for f in TARGET_FIRMS if os.path.exists(os.path.join(temp_dir, f"{f}.csv"))]
    if not all_files:
        print("No data collected. Exit.")
        exit()
        
    df = pd.concat([pd.read_csv(f) for f in all_files])
    
    # 4. SEC 토픽 통합
    if os.path.exists(args.topic_csv):
        topic_df = pd.read_csv(args.topic_csv)
        # 2013-09, 2013-12 등의 과거 분기 공시 데이터를 살려서 2014년 초반으로 전진 채우기 위해 outer join 사용
        df = pd.merge(df, topic_df, on=['firm_id', 'date'], how='outer')
        df.sort_values(by=['firm_id', 'date'], inplace=True)
        t_cols = [c for c in df.columns if c.startswith('expo_topic_')]
        if t_cols: 
            df[t_cols] = df.groupby('firm_id')[t_cols].ffill().fillna(0.0)
            
        # 전진 채움이 완료된 후 분석 대상 타겟 범위(2014-01 ~ 2025-12)만 남기기
        df = df[(df['date'] >= args.start_month) & (df['date'] <= args.end_month)]
        df.sort_values(by=['firm_id', 'date'], inplace=True)
    
    # 5. AFCI 및 최종 저장
    df = calculate_afci(df)
    
    # 컬럼 이름 및 최종 순서 정렬
    if 'stock_return_1m' in df.columns:
        df.rename(columns={'stock_return_1m': 'stock_return'}, inplace=True)
        
    final_cols = [
        'date', 'firm_id', 'stock_return', 'paper_count',
        'news_count', 'news_sentiment', 'news_ai_exposure', 'news_impact',
        'gt_firm_attention', 'gt_ai_momentum', 'expo_topic_semiconductors',
        'expo_topic_cloud', 'expo_topic_software', 'expo_topic_hardware',
        'expo_topic_advertising', 'expo_topic_social', 'expo_topic_ecommerce',
        'expo_topic_data', 'expo_topic_auto', 'AFCI'
    ]
    df = df[final_cols]
    df.to_csv(args.out_csv, index=False)
    print(f"\nFinal dataset saved ({len(df)} rows) → {args.out_csv}")
