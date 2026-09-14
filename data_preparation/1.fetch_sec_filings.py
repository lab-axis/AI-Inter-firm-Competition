import os
import argparse
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi

# Collect configured 10-K, 10-Q and 20-F filings by company CIK.
# Use section extraction for domestic forms and HTML text extraction for 20-F.
# 6-K filings are excluded by the configured form filter.
# Write per-firm TXT archives under the configured output directory.

CIK_MAPPING = {
    "MSFT": ["0000789019"],
    "GOOGL": ["0001652044", "0001288776"],  # Alphabet Inc. & Google Inc. (pre-2015)
    "AMZN": ["0001018724"],
    "META": ["0001326801"],
    "AAPL": ["0000320193"],
    "NVDA": ["0001045810"],
    "AMD": ["0000002488"],
    "AVGO": ["0001730168", "0001441634"],  # Broadcom Inc. & Avago Technologies (pre-2018)
    "QCOM": ["0000804328"],
    "INTC": ["0000050863"],
    "ADI": ["0000006281"],
    "TSM": ["0001046179"],
    "ASML": ["0000937966"],
    "AMAT": ["0000006951"],
    "LRCX": ["0000707549"],
    "KLAC": ["0000319201"],
    "MU": ["0000723125"],
    "TXN": ["0000097476"],
    "ADBE": ["0000796343"],
    "CRM": ["0001108524"],
    "ORCL": ["0001341439"],
    "IBM": ["0000051143"],
    "SAP": ["0001000184"],
    "NOW": ["0001373715"],
    "INTU": ["0000896878"],
    "PANW": ["0001327567"],
    "ADSK": ["0000769397"],
    "CSCO": ["0000858877"],
    "STX": ["0001137789"],
    "TSLA": ["0001318605"]
}

TARGET_FIRMS = list(CIK_MAPPING.keys())

def fetch_sec_filings(api_key: str, start_year: int, end_year: int, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    query_api = QueryApi(api_key=api_key)
    extractor_api = ExtractorApi(api_key=api_key)
    
    total_firms = len(TARGET_FIRMS)
    print(f"=== SEC Filings Collection (CIK-based) | {start_year} ~ {end_year} ===")
    print("Collecting: 10-K, 10-Q, 20-F (Excluding 6-K)")
    
    total_fetched = 0
    
    for idx, ticker in enumerate(TARGET_FIRMS):
        ciks = CIK_MAPPING[ticker]
        ciks_str = ", ".join(ciks)
        # Remove leading zeros from the CIK used in the search query.
        cik_queries = [cik.lstrip('0') for cik in ciks]
        cik_query_str = " OR ".join([f"cik:{cq}" for cq in cik_queries])
        
        print(f"\n[{idx + 1}/{total_firms}] {ticker} (CIKs: {ciks_str}) ── Checking filings…")
        ticker_dir = os.path.join(out_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        all_filings = []
        for cik in ciks:
            cik_query = cik.lstrip('0')
            from_idx = 0
            while True:
                query = {
                    "query": {
                        "query_string": {
                            "query": f"cik:{cik_query} AND formType:(\"10-K\" OR \"10-Q\" OR \"10-K/A\" OR \"10-Q/A\" OR \"20-F\" OR \"20-F/A\") AND filedAt:[{start_year - 1}-01-01 TO {end_year + 1}-03-31]"
                        }
                    },
                    "from": str(from_idx),
                    "size": "50",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
                try:
                    response = query_api.get_filings(query)
                    filings = response.get('filings', [])
                    print(f"    - CIK {cik} (from {from_idx}): Found {len(filings)} filings.")
                    all_filings.extend(filings)
                    if len(filings) < 50:
                        break
                    from_idx += 50
                except Exception as e:
                    print(f"    [ERROR] CIK {cik} query failed (from {from_idx}): {e}")
                    break
                
        # Now process all collected filings across all CIKs
        print(f"    - Total corporate filings across all CIKs: {len(all_filings)}")
        
        filings_by_period = {}
        for filing in all_filings:
            period = filing.get('periodOfReport')
            if not period: continue
            if period not in filings_by_period:
                filings_by_period[period] = []
            filings_by_period[period].append(filing)
            
        for period, group in filings_by_period.items():
            group_sorted = sorted(group, key=lambda x: x['formType'].endswith('/A'), reverse=True)
            
            period_success = False
            for filing in group_sorted:
                form_type = filing['formType']
                period_of_report = filing.get('periodOfReport', filing['filedAt'][:10])
                safe_form_type = form_type.replace('/', '')
                filename = f"{ticker}_{safe_form_type}_{period_of_report}.txt"
                filepath = os.path.join(ticker_dir, filename)
                
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    period_success = True
                    break
                    
                url = filing['linkToFilingDetails']
                try:
                    text_content = ""
                    # Extract the configured sections from domestic 10-K and 10-Q forms.
                    if form_type.startswith("10-K"):
                        text_content += extractor_api.get_section(url, "1", "text") or ""
                        text_content += "\n\n" + (extractor_api.get_section(url, "1A", "text") or "")
                        text_content += "\n\n" + (extractor_api.get_section(url, "7", "text") or "")
                    elif form_type.startswith("10-Q"):
                        text_content += extractor_api.get_section(url, "part1item2", "text") or ""
                        text_content += "\n\n" + (extractor_api.get_section(url, "part2item1a", "text") or "")
                    
                    # Download and parse HTML for foreign-issuer 20-F forms.
                    elif form_type.startswith("20-F"):
                        headers = {'User-Agent': 'AI Research Project research@ai-company.com'}
                        res = requests.get(url, headers=headers, timeout=20)
                        if res.status_code == 200:
                            soup = BeautifulSoup(res.text, 'html.parser')
                            text_content = soup.get_text(separator=' ', strip=True)
                        else:
                            print(f"      [!] 20-F Download failed with status {res.status_code}")
                        
                    if text_content.strip():
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        print(f"      Saved: {filename}")
                        total_fetched += 1
                        period_success = True
                        break
                except Exception as e:
                    print(f"    [!] Error ({filename}): {e}")
                
                time.sleep(0.5)
            
    print(f"\n[SUCCESS] Total filings saved: {total_fetched}")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", type=str, default="../data_preparation/data/1_transcripts")
    
    args = parser.parse_args()
    api_key = args.api_key or os.getenv("SEC_API_KEY")
    if not api_key: exit(1)
        
    fetch_sec_filings(api_key, args.start_year, args.end_year, args.out_dir)
