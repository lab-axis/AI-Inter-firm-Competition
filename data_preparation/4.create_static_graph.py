import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

# 역할: Step 3의 패널 데이터를 바탕으로 [주가 상관계수 50% + NLP 코사인 유사도 50%](단일 가중치 처리도 선택 가능)의 가중치를 주어 하이브리드 수학 모델을 돌린다.
# 결과: n개 노드 연결망 구조인 data/edges_static_prior.csv로 만들어 진다.

def calculate_pearson_edges(panel_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Method 1: Empirical Market Co-movement (Pearson Correlation)
    Calculates the correlation matrix of monthly stock returns.
    """
    print("Calculating Market Pearson Correlation (A_market)...")
    # Pivot to get firms as columns and dates as rows for stock_return
    returns_df = panel_df.pivot(index='date', columns='firm_id', values='stock_return')
    corr_matrix = returns_df.corr(method='pearson')
    
    edges = []
    firms = corr_matrix.columns
    for i in range(len(firms)):
        for j in range(len(firms)):
            if i != j:
                weight = corr_matrix.iloc[i, j]
                if weight > threshold:
                    edges.append({
                        "source_id": firms[i],
                        "target_id": firms[j],
                        "weight": weight,
                        "relation_type": "market"
                    })
    return pd.DataFrame(edges)

def calculate_cosine_edges(panel_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Method 2: Strategic Business Overlap (NLP Topic Cosine Similarity)
    Calculates the cosine similarity of the firm's average NLP topic vectors.
    """
    print("Calculating NLP Cosine Similarity (A_nlp)...")
    topic_cols = [c for c in panel_df.columns if c.startswith('expo_topic_')]
    
    if not topic_cols:
        print("Warning: No NLP topic columns found. Cannot calculate Cosine Similarity.")
        return pd.DataFrame(columns=["source_id", "target_id", "weight", "relation_type"])
        
    # Get the average topic exposure vector for each firm across all time
    avg_topics = panel_df.groupby('firm_id')[topic_cols].mean()
    firms = avg_topics.index
    
    # Calculate pairwise cosine similarity
    sim_matrix = cosine_similarity(avg_topics.values)
    
    edges = []
    for i in range(len(firms)):
        for j in range(len(firms)):
            if i != j:
                weight = sim_matrix[i, j]
                if weight > threshold:
                    edges.append({
                        "source_id": firms[i],
                        "target_id": firms[j],
                        "weight": weight,
                        "relation_type": "nlp"
                    })
    return pd.DataFrame(edges)

def calculate_hybrid_edges(panel_df: pd.DataFrame, alpha: float = 0.5, threshold: float = 0.5) -> pd.DataFrame:
    """
    Method 3: Multiplex/Hybrid Graph
    Combines Pearson (market dynamics) and Cosine Similarity (fundamental strategy)
    A_hybrid = alpha * A_market + (1 - alpha) * A_nlp
    """
    print(f"Calculating Hybrid Multiplex Graph (alpha={alpha})...")
    
    # 1. Market Matrix
    returns_df = panel_df.pivot(index='date', columns='firm_id', values='stock_return')
    A_market = returns_df.corr(method='pearson').fillna(0)
    
    # 2. NLP Matrix
    topic_cols = [c for c in panel_df.columns if c.startswith('expo_topic_')]
    if not topic_cols:
        print("Warning: No NLP columns found. Falling back to pure Pearson (alpha forced to 1.0).")
        print("         Run 2.topic_modeling.py + 3.build_dataset.py to enable NLP layer.")
        A_hybrid = A_market  # NLP 없을 때 가중치의 사라짐 방지
    else:
        avg_topics = panel_df.groupby('firm_id')[topic_cols].mean()
        avg_topics = avg_topics.reindex(A_market.index).fillna(0)
        sim_matrix = cosine_similarity(avg_topics.values)
        A_nlp = pd.DataFrame(sim_matrix, index=A_market.index, columns=A_market.columns)
        A_hybrid = (alpha * A_market) + ((1.0 - alpha) * A_nlp)
    
    edges = []
    firms = A_hybrid.columns
    for i in range(len(firms)):
        for j in range(len(firms)):
            if i != j:
                weight = A_hybrid.iloc[i, j]
                if weight > threshold:
                    edges.append({
                        "source_id": firms[i],
                        "target_id": firms[j],
                        "weight": weight,
                        "relation_type": "hybrid"
                    })
    return pd.DataFrame(edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create static prior graph using mathematical models")
    parser.add_argument("--panel-csv", type=str, default='../data_preparation/data/3_dataset/data_set.csv', help="Input panel data")
    parser.add_argument("--out-csv", type=str, default='../data_preparation/data/4_graph/edges_static_prior.csv', help="Output edges path")
    parser.add_argument("--method", type=str, choices=["pearson", "cosine", "hybrid"], default="hybrid")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum edge weight threshold. Real AI stocks typically correlate 0.3-0.7. Default=0.4")
    parser.add_argument("--alpha", type=float, default=0.5, help="Pearson weight in hybrid method (0~1). Default=0.5")
    parser.add_argument("--diagnose", action="store_true", help="Print Pearson correlation matrix and edge counts per threshold before saving")
    
    args = parser.parse_args()

    if not os.path.exists(args.panel_csv):
        print(f"Error: Panel data not found at:\n  {args.panel_csv}")
        exit(1)

    panel_df = pd.read_csv(args.panel_csv)
    panel_df = panel_df.drop_duplicates(subset=['date', 'firm_id'], keep='first')  # 중복 제거
    print(f"Loaded panel: {panel_df.shape[0]} rows, {panel_df.shape[1]} cols")
    print(f"Firms: {sorted(panel_df['firm_id'].unique().tolist())}")
    topic_cols = [c for c in panel_df.columns if c.startswith('expo_topic_')]
    print(f"NLP topic columns: {len(topic_cols)} found")

    # 상관행렬 + 임계값별 엣지 수 출력
    if args.diagnose or True:
        returns_df = panel_df.pivot(index='date', columns='firm_id', values='stock_return')
        corr = returns_df.corr(method='pearson')
        print("\n=== Pearson Correlation Matrix ===")
        print(corr.round(3).to_string())
        print("\n=== Edge Counts by Threshold ===")
        firms = corr.columns
        for thr in [0.7, 0.6, 0.5, 0.4, 0.3]:
            cnt = sum(1 for i in range(len(firms)) for j in range(len(firms))
                      if i != j and not np.isnan(corr.iloc[i, j]) and corr.iloc[i, j] > thr)
            marker = "  ← selected" if abs(thr - args.threshold) < 1e-9 else ""
            print(f"  threshold={thr:.1f} → {cnt:3d} edges{marker}")
        print()

    if args.method == "pearson":
        edges_df = calculate_pearson_edges(panel_df, args.threshold)
    elif args.method == "cosine":
        edges_df = calculate_cosine_edges(panel_df, args.threshold)
    elif args.method == "hybrid":
        edges_df = calculate_hybrid_edges(panel_df, args.alpha, args.threshold)

    if not edges_df.empty:
        edges_df = edges_df.sort_values(by="weight", ascending=False)
    else:
        print(f"Warning: No edges met threshold={args.threshold}. Try lowering --threshold.")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    edges_df.to_csv(args.out_csv, index=False)
    print(f"\n--- Method: {args.method.upper()} | threshold={args.threshold} | alpha={args.alpha} ---")
    print(f"Saved {len(edges_df)} edges → {args.out_csv}")
    if not edges_df.empty:
        print(edges_df.head(10).to_string())
