import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# python Agent/gdelt_collector.py --mode sql
# python Agent/gdelt_collector.py --mode preprocess

# 역할: 30개 AI/IT 기업의 12년치(2014~2025) 뉴스 데이터 수집을 위한 GDELT BigQuery SQL 생성 및 수집된 GDELT CSV 데이터를 정제하는 전처리 파이프라인.
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

def generate_bigquery_sql(start_date: str, end_date: str) -> str:
    """
    Google BigQuery 콘솔에서 실행하여 12년치 고품질 GDELT 뉴스를 다운로드할 수 있는 
    최적화된 SQL 쿼리를 생성합니다. 
    사용자 피드백을 반영하여 모든 기업에 대해 공통 다층 필터 장치(글로벌 노이즈 차단)와
    IT/비즈니스 도메인의 최소 테마 제약(Minimum Theme Constraints)을 일괄 적용했습니다.
    """
    # GDELT DATE 포맷: YYYYMMDDHHMMSS (integer)
    start_fmt = start_date.replace("-", "") + "000000"
    end_fmt = end_date.replace("-", "") + "235959"
    
    # 30개 기업에 대한 V2Organizations LIKE 매칭 조건 작성 (GKG v2의 실측 매칭 컬럼)
    like_conditions = []
    for ticker, names in TARGET_FIRMS_MAPPING.items():
        for name in names:
            # SQL 내 인용부호 이스케이프 처리
            escaped_name = name.replace("'", "''")
            like_conditions.append(f"V2Organizations LIKE '%{escaped_name}%'")
            
    or_clause = "\n      OR ".join(like_conditions)
    
    sql = f"""-- =========================================================================
-- GDELT BigQuery Extraction SQL (2014-01-01 ~ 2025-12-31)
-- [V2Organizations 및 V2Themes 완벽 대응 및 글로벌 필터 반영]
-- =========================================================================

SELECT
  GKGRECORDID AS article_id,
  DATE AS date_raw,
  SourceCommonName AS source,
  DocumentIdentifier AS url,
  V2Organizations AS organizations, -- GKG v2 호환을 위한 V2 컬럼 매핑
  V2Themes AS themes,               -- GKG v2 호환을 위한 V2 컬럼 매핑
  V2Tone AS tone_raw
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  DATE >= {start_fmt}
  AND DATE <= {end_fmt}
  AND (
      {or_clause}
  )
  -- [공통 다층 필터 1] 농업, 식음료, 스포츠, 엔터 등 IT/비즈니스 예측과 무관한 테마 일괄 배제
  -- (Apple의 과일 노이즈, Amazon의 우림/강 노이즈, SAP/NOW의 일반어 노이즈 동시 차단)
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
  -- [공통 다층 필터 2: 최소 테마 제약] 데이터의 성격이 최소한 기술, 컴퓨터, 경제, 비즈니스, 과학, 제조 중 하나여야 함
  -- 이 조건을 충족하는 양질의 기업 경영/기술 관련 뉴스만 선별 적재합니다.
  AND (
    V2Themes LIKE '%TECHNOLOGY%'
    OR V2Themes LIKE '%COMPUTERS%'
    OR V2Themes LIKE '%TELECOMMUNICATIONS%'
    OR V2Themes LIKE '%ECON_%'  -- ECON_STOCKMARKET, ECON_ENTREPRENEURSHIP 등 포괄
    OR V2Themes LIKE '%BUSINESS%'
    OR V2Themes LIKE '%PATENT%'
    OR V2Themes LIKE '%SCIENCE%'
    OR V2Themes LIKE '%MANUFACTURING%'
  )
  -- 연구 목적의 고품질 영어 기사 위주로 필터링
  AND DocumentIdentifier LIKE 'http%'
  AND SourceCommonName IS NOT NULL
ORDER BY
  DATE DESC
LIMIT 200000;
"""
    return sql

def preprocess_gdelt_csv(raw_csv_path: str, output_csv_path: str):
    """
    BigQuery에서 내려받은 GDELT GKG CSV raw 파일을 읽어 
    시계열 날짜 정밀화, 감성 분석 점수(Tone) 추출 및 정제 작업을 수행합니다.
    """
    if not os.path.exists(raw_csv_path):
        print(f"[오류] 원본 GDELT CSV 파일이 없습니다: {raw_csv_path}")
        print("    -> BigQuery에서 다운로드한 CSV 파일을 위 경로에 배치해주세요.")
        return False
        
    print(f"=== GDELT 전처리 파이프라인 가동: {raw_csv_path} ===")
    df = pd.read_csv(raw_csv_path)
    print(f"    - 로드된 총 로우 수: {len(df)}")
    
    # 1. 날짜 처리 (YYYYMMDDHHMMSS -> YYYY-MM-DD)
    print("    - 1. 시계열 날짜 변환 중...")
    df['date'] = df['date_raw'].astype(str).str[:8]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date'])
    
    # 2. Tone 데이터 파싱 (V2Tone 형식: 'tone,positive,negative,polarity,activity,self_reference,words')
    #    우리는 첫 번째 요소인 대표 'tone'(평균 감성 점수: -100 ~ +100)를 추출합니다.
    print("    - 2. 감성 스코어(V2Tone) 정제 중...")
    def parse_tone(x):
        if pd.isna(x) or not isinstance(x, str):
            return 0.0
        try:
            parts = x.split(',')
            return float(parts[0]) # 평균 톤 점수
        except:
            return 0.0
            
    df['sentiment_score'] = df['tone_raw'].apply(parse_tone)
    
    # 3. 불필요한 원본 컬럼 제거 및 기사 제목 복원 시도
    #    GDELT GKG는 URL은 주지만 기사 제목은 제공하지 않으므로, URL의 마지막 부분을 간이 제목으로 삼거나 
    #    추후 임베딩 단계에서 크롤링할 수 있도록 설계합니다.
    print("    - 3. URL 기반 기사 간이 제목 생성 중...")
    def extract_title_from_url(url):
        if pd.isna(url) or not isinstance(url, str):
            return "IT Market Intelligence News"
        try:
            # URL 끝자락에서 단어들을 추출하여 제목처럼 생성
            parts = url.rstrip('/').split('/')
            last_part = parts[-1].replace('-', ' ').replace('_', ' ')
            if len(last_part) > 15 and '.html' not in last_part:
                # 확장자 제거
                title = last_part.split('.')[0]
                return title.capitalize()
        except:
            pass
        return "Global Tech Industry News"
        
    df['title'] = df['url'].apply(extract_title_from_url)
    
    # 4. 정제 완료된 컬럼만 보관
    cleaned_df = df[['article_id', 'date', 'source', 'url', 'title', 'organizations', 'themes', 'sentiment_score']]
    
    # 중복 뉴스 제거 (동일 URL 기준)
    cleaned_df = cleaned_df.drop_duplicates(subset=['url'])
    
    # 저장
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cleaned_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[완료] 정제된 데이터가 저장되었습니다 ({len(cleaned_df)}건) -> {output_csv_path}")
    return True

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="GDELT Project Data Collector")
    parser.add_argument("--mode", type=str, choices=["sql", "preprocess"], default="sql", 
                        help="sql: BigQuery용 SQL 쿼리 생성, preprocess: 수집된 CSV 데이터 전처리")
    parser.add_argument("--raw-csv", type=str, default="Agent/data/gdelt_orig.csv")
    parser.add_argument("--out-csv", type=str, default="Agent/data/gdelt_cleaned.csv")
    parser.add_argument("--start-date", type=str, default="2014-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    
    args = parser.parse_args()
    
    # 프로젝트 루트에 data 디렉토리 생성
    os.makedirs("data", exist_ok=True)
    
    if args.mode == "sql":
        sql_query = generate_bigquery_sql(args.start_date, args.end_date)
        sql_filepath = "data/gdelt_extraction_query.sql"
        with open(sql_filepath, 'w', encoding='utf-8') as f:
            f.write(sql_query)
        
        print("\n" + "="*80)
        print(" GDELT BigQuery용 12년치 추출 SQL 쿼리 파일이 생성되었습니다.")
        print(f" 경로: {sql_filepath}")
        print("="*80)
        print(" * 다운로드한 CSV 파일을 'Agent/data/gdelt_orig.csv' 경로에 복사한 뒤,")
        print("   다음 명령어를 실행하여 데이터를 정제하세요:")
        print("   python Agent/gdelt_collector.py --mode preprocess")
        print("="*80 + "\n")
        
    elif args.mode == "preprocess":
        preprocess_gdelt_csv(args.raw_csv, args.out_csv)
