import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# ==============================================================================
# GDELT Data Collector (SCI Q1 Methodology)
# 
# [방법론 핵심 설계]
# 1. 일관성 확보: GKG 2.0 출범 이후인 2015년 3월 1일부터 수집 (2014년 혼합 배제)
# 2. 화이트리스트 도입: 검증된 1~3 Tier 매체 도메인만 수집하여 스팸 및 노이즈 원천 차단
# 3. 오버샘플링(Oversampling) 전략: 크롤링 시 발생하는 404 에러(링크 부패)를 대비하여, 
#    목표 수량(예: 50개)의 N배수(예: 200개)를 BigQuery에서 1차 추출.
# ==============================================================================

TARGET_FIRMS_MAPPING = {
    "MSFT": ["Microsoft"],
    "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon"],
    "META": ["Meta Platforms", "Facebook"],
    "AAPL": ["Apple"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "AMD": ["Advanced Micro Devices", "AMD"],
    "AVGO": ["Broadcom"],
    "QCOM": ["Qualcomm"],
    "INTC": ["Intel"],
    "ADI": ["Analog Devices"],
    "TSM": ["Taiwan Semiconductor", "TSMC"],
    "ASML": ["ASML"],
    "AMAT": ["Applied Materials"],
    "LRCX": ["Lam Research"],
    "KLAC": ["KLA Corporation", "KLA"],
    "MU": ["Micron Technology", "Micron"],
    "TXN": ["Texas Instruments"],
    "ADBE": ["Adobe"],
    "CRM": ["Salesforce"],
    "ORCL": ["Oracle"],
    "IBM": ["IBM", "International Business Machines"],
    "SAP": ["SAP"],
    "NOW": ["ServiceNow"],
    "INTU": ["Intuit"],
    "PANW": ["Palo Alto Networks"],
    "ADSK": ["Autodesk"],
    "CSCO": ["Cisco Systems", "Cisco"],
    "STX": ["Seagate Technology", "Seagate"],
    "TSLA": ["Tesla"]
}

# 고신뢰성 매체 화이트리스트 도메인 (SourceCommonName 실측값 기반)
WHITELIST_DOMAINS = [
    # Tier 1: Industry Analysis & Business Insight
    'venturebeat.com', 
    'forbes.com', 
    'bloomberg.com', 
    'cnbc.com', 
    'businessinsider.com',
    # Tier 2: Authoritative Tech & Industry News
    'reuters.com', 
    'apnews.com', 
    'theverge.com', 
    'wired.com', 
    'techcrunch.com',
    # Tier 3: Reliable Market & Financial Data
    'wsj.com', 
    'ft.com', 
    'nasdaq.com'
]

def generate_bigquery_sql(start_date: str, end_date: str, target_per_month: int = 50, oversample_rate: int = 4, sample_strategy: str = "random", by_company: bool = True) -> str:
    """
    BigQuery 추출 SQL 쿼리 생성.
    오버샘플링을 적용하여 월별/기업별 (target_per_month * oversample_rate) 개수만큼 추출합니다.
    """
    from datetime import datetime
    
    # GDELT DATE 포맷: YYYYMMDDHHMMSS (integer)
    start_fmt = start_date.replace("-", "") + "000000"
    end_fmt = end_date.replace("-", "") + "235959"
    
    # 30개 기업에 대한 매칭 조건
    like_conditions = []
    for ticker, names in TARGET_FIRMS_MAPPING.items():
        for name in names:
            escaped_name = name.replace("'", "''")
            like_conditions.append(f"V2Organizations LIKE '%{escaped_name}%'")
            
    or_clause = "\n      OR ".join(like_conditions)
    
    # 화이트리스트 도메인 매칭 조건 생성
    domain_list_str = ", ".join([f"'{d}'" for d in WHITELIST_DOMAINS])
    whitelist_clause = f"SourceCommonName IN ({domain_list_str})"
    
    # 최종 추출할 월별 버킷 사이즈 계산 (목표치 x 오버샘플링 배수)
    bucket_limit = target_per_month * oversample_rate
    
    order_by_clause = "date_raw DESC" if sample_strategy == "latest" else "RAND()"
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        months_count = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
        if months_count <= 0: months_count = 1
    except Exception:
        months_count = 130 # 2015-03 to 2025-12
        
    if by_company:
        # 기업별 매핑 서브쿼리 생성
        union_parts = []
        for ticker, names in TARGET_FIRMS_MAPPING.items():
            conditions = []
            for name in names:
                escaped_name = name.replace("'", "''")
                conditions.append(f"organizations LIKE '%{escaped_name}%'")
            or_cond = " OR ".join(conditions)
            union_parts.append(f"  SELECT *, '{ticker}' AS matched_company FROM filtered_gkg WHERE {or_cond}")
            
        union_clause = "\n  UNION ALL\n".join(union_parts)
        
        sql = f"""-- =========================================================================
-- GDELT BigQuery Extraction SQL (SCI Q1 Methodology)
-- Period: {start_date} ~ {end_date} (Total {months_count} months)
-- Target: {target_per_month} valid rows/month/company -> Oversampled to {bucket_limit} rows
-- Strategy: {sample_strategy} (by_company = True)
-- =========================================================================

WITH filtered_gkg AS (
  SELECT
    GKGRECORDID AS article_id,
    DATE AS date_raw,
    SourceCommonName AS source,
    DocumentIdentifier AS url,
    V2Organizations AS organizations,
    V2Themes AS themes,
    V2Tone AS tone_raw,
    SUBSTR(CAST(DATE AS STRING), 1, 6) AS year_month
  FROM
    `gdelt-bq.gdeltv2.gkg`
  WHERE
    DATE >= {start_fmt}
    AND DATE <= {end_fmt}
    -- [품질 확보] 화이트리스트 저널만 허용 (스팸 봇 완전 차단)
    AND {whitelist_clause}
    AND (
      {or_clause}
    )
    -- [공통 다층 필터 1] 불필요 노이즈 배제
    AND NOT (
      V2Themes LIKE '%AGR_AGRICULTURE%'
      OR V2Themes LIKE '%FARMING%'
      OR V2Themes LIKE '%CROPS%'
      OR V2Themes LIKE '%FOOD_AND_BEVERAGE%'
      OR V2Themes LIKE '%HARVEST%'
      OR V2Themes LIKE '%SUPERMARKET%'
      OR V2Themes LIKE '%SPORTS%'
      OR V2Themes LIKE '%LEISURE%'
      OR V2Themes LIKE '%ENTERTAINMENT%'
    )
    -- [공통 다층 필터 2] 비즈니스 및 IT 연관 기사만 통과
    AND (
      V2Themes LIKE '%TECHNOLOGY%'
      OR V2Themes LIKE '%COMPUTERS%'
      OR V2Themes LIKE '%TELECOMMUNICATIONS%'
      OR V2Themes LIKE '%ECON_%'
      OR V2Themes LIKE '%BUSINESS%'
      OR V2Themes LIKE '%PATENT%'
      OR V2Themes LIKE '%SCIENCE%'
      OR V2Themes LIKE '%MANUFACTURING%'
    )
    AND DocumentIdentifier LIKE 'http%'
),
company_mapped AS (
{union_clause}
),
ranked_gkg AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY year_month, matched_company ORDER BY {order_by_clause}) AS row_num
  FROM
    company_mapped
),
selected_articles AS (
  SELECT
    article_id,
    date_raw,
    source,
    url,
    organizations,
    themes,
    tone_raw,
    matched_company
  FROM
    ranked_gkg
  WHERE
    row_num <= {bucket_limit}
)
SELECT DISTINCT
  *
FROM
  selected_articles
ORDER BY
  date_raw DESC;
"""
    else:
        # 기업 구분 없는 단순 월별 수집
        sql = f"""-- =========================================================================
-- GDELT BigQuery Extraction SQL (Uniform Monthly Oversampling)
-- Period: {start_date} ~ {end_date}
-- Target: {target_per_month} valid rows/month -> Oversampled to {bucket_limit} rows
-- Strategy: {sample_strategy} (by_company = False)
-- =========================================================================

WITH filtered_gkg AS (
  SELECT
    GKGRECORDID AS article_id,
    DATE AS date_raw,
    SourceCommonName AS source,
    DocumentIdentifier AS url,
    V2Organizations AS organizations,
    V2Themes AS themes,
    V2Tone AS tone_raw,
    SUBSTR(CAST(DATE AS STRING), 1, 6) AS year_month
  FROM
    `gdelt-bq.gdeltv2.gkg`
  WHERE
    DATE >= {start_fmt}
    AND DATE <= {end_fmt}
    AND {whitelist_clause}
    AND (
      {or_clause}
    )
    AND NOT (
      V2Themes LIKE '%AGR_AGRICULTURE%'
      OR V2Themes LIKE '%FARMING%'
      OR V2Themes LIKE '%CROPS%'
      OR V2Themes LIKE '%FOOD_AND_BEVERAGE%'
      OR V2Themes LIKE '%HARVEST%'
      OR V2Themes LIKE '%SUPERMARKET%'
      OR V2Themes LIKE '%SPORTS%'
      OR V2Themes LIKE '%LEISURE%'
      OR V2Themes LIKE '%ENTERTAINMENT%'
    )
    AND (
      V2Themes LIKE '%TECHNOLOGY%'
      OR V2Themes LIKE '%COMPUTERS%'
      OR V2Themes LIKE '%TELECOMMUNICATIONS%'
      OR V2Themes LIKE '%ECON_%'
      OR V2Themes LIKE '%BUSINESS%'
      OR V2Themes LIKE '%PATENT%'
      OR V2Themes LIKE '%SCIENCE%'
      OR V2Themes LIKE '%MANUFACTURING%'
    )
    AND DocumentIdentifier LIKE 'http%'
),
ranked_gkg AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY year_month ORDER BY {order_by_clause}) AS row_num
  FROM
    filtered_gkg
)
SELECT
  article_id,
  date_raw,
  source,
  url,
  organizations,
  themes,
  tone_raw
FROM
  ranked_gkg
WHERE
  row_num <= {bucket_limit}
ORDER BY
  date_raw DESC;
"""
    return sql

def preprocess_gdelt_csv(raw_csv_path: str, output_csv_path: str):
    """
    BigQuery에서 내려받은 오버샘플링된 CSV 데이터를 1차 정제합니다.
    (실제 404 체크 및 본문 추출은 별도의 크롤러 봇 스크립트에서 수행합니다.)
    """
    if not os.path.exists(raw_csv_path):
        print(f"[오류] 원본 CSV 파일이 없습니다: {raw_csv_path}")
        return False
        
    print(f"=== GDELT 1차 전처리 가동: {raw_csv_path} ===")
    df = pd.read_csv(raw_csv_path)
    print(f"    - 로드된 총 로우 수 (오버샘플링 포함): {len(df)}")
    
    df['date'] = df['date_raw'].astype(str).str[:8]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date'])
    
    def parse_tone(x):
        if pd.isna(x) or not isinstance(x, str): return 0.0
        try: return float(x.split(',')[0])
        except: return 0.0
            
    df['sentiment_score'] = df['tone_raw'].apply(parse_tone)
    
    def extract_title_from_url(url):
        if pd.isna(url) or not isinstance(url, str): return "News"
        try:
            parts = url.rstrip('/').split('/')
            last_part = parts[-1].replace('-', ' ').replace('_', ' ')
            if len(last_part) > 15 and '.html' not in last_part:
                return last_part.split('.')[0].capitalize()
        except: pass
        return "Tech News"
        
    df['title'] = df['url'].apply(extract_title_from_url)
    
    # 'matched_company' 필드가 있으면 유지, 없으면 None
    cols_to_keep = ['article_id', 'date', 'source', 'url', 'title', 'organizations', 'themes', 'sentiment_score']
    if 'matched_company' in df.columns:
        cols_to_keep.append('matched_company')
        
    cleaned_df = df[cols_to_keep]
    cleaned_df = cleaned_df.drop_duplicates(subset=['url'])
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cleaned_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[완료] 1차 정제본 저장 (중복제거 후 {len(cleaned_df)}건) -> {output_csv_path}")
    return True

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="GDELT Project Data Collector")
    parser.add_argument("--mode", type=str, choices=["sql", "preprocess"], default="sql")
    parser.add_argument("--raw-csv", type=str, default="data/gdelt_orig.csv")
    parser.add_argument("--out-csv", type=str, default="data/gdelt_cleaned.csv")
    # 2014년 배제 (GKG 2.0 공식 시작일 이후로 고정)
    parser.add_argument("--start-date", type=str, default="2015-03-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    # 월별 목표 개수와 오버샘플링 비율로 로직 변경
    parser.add_argument("--target-per-month", type=int, default=50,
                        help="기업별/월별 최종 확보할 유효 기사 수")
    parser.add_argument("--oversample-rate", type=int, default=2,
                        help="초과 추출 배수 (404 에러 대비, 기본값 2배수 = SQL에서는 100개 추출)")
    parser.add_argument("--sample-strategy", type=str, choices=["latest", "random"], default="random")
    parser.add_argument("--by-company", action="store_true", default=True)
    parser.add_argument("--no-by-company", action="store_false", dest="by_company")
    parser.add_argument("--split-years", action="store_true", default=False)
    
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    
    if args.mode == "sql":
        if args.split_years:
            from datetime import datetime
            try:
                start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            except:
                start_dt = datetime(2015, 1, 1)
                end_dt = datetime(2026, 5, 31)
                
            output_dir = "Agent/data/yearly_queries"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n🚀 연도별 분할 SQL 생성 (오버샘플링 및 화이트리스트 적용)")
            
            for year in range(start_dt.year, end_dt.year + 1):
                # 2015년은 3월 1일부터 시작하도록 예외 처리
                year_start = f"{year}-03-01" if year == 2015 else f"{year}-01-01"
                year_end = f"{year}-12-31"
                
                sql_query = generate_bigquery_sql(
                    year_start, year_end,
                    target_per_month=args.target_per_month,
                    oversample_rate=args.oversample_rate,
                    sample_strategy=args.sample_strategy,
                    by_company=args.by_company
                )
                
                filepath = os.path.join(output_dir, f"gdelt_query_{year}.sql")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(sql_query)
                    
            print(f"✅ 연도별 SQL 파일 생성 완료! ({output_dir}/)")
            print(f"   - 타겟: 최종 {args.target_per_month}개 달성을 위해 {args.target_per_month * args.oversample_rate}개씩 오버샘플링 추출")
            print(f"   - 필터: 지정된 {len(WHITELIST_DOMAINS)}개 고품질 도메인 화이트리스트 적용")
            
        else:
            sql_query = generate_bigquery_sql(
                args.start_date, args.end_date,
                target_per_month=args.target_per_month,
                oversample_rate=args.oversample_rate,
                sample_strategy=args.sample_strategy,
                by_company=args.by_company
            )
            filepath = "Agent/data/gdelt_extraction_query.sql"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(sql_query)
            print(f"\n🚀 단일 SQL 생성 완료: {filepath}")
            print(f"   - 기간: {args.start_date} ~ {args.end_date}")
            print(f"   - 타겟: 최종 {args.target_per_month}개 달성을 위해 {args.target_per_month * args.oversample_rate}개씩 오버샘플링 추출")
            
    elif args.mode == "preprocess":
        preprocess_gdelt_csv(args.raw_csv, args.out_csv)
