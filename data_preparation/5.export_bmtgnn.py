import argparse
import os
import pandas as pd
import numpy as np

# 역할: 가공된 패널 데이터(3.build_dataset.py)와 그래프 데이터(4.create_static_graph.py)를 
#       B-MTGNN PyTorch 모델이 학습할 수 있는 3D 텐서 및 인접 행렬 형태로 변환한다.
# 결과: data/5_bmtgnn_input/bmtgnn_data.npy, adj_mat.csv, node_ids.csv

def export_for_bmtgnn(panel_csv: str, edges_csv: str, out_dir: str):
    """
    패널 데이터를 [노드 수, 시계열 길이, 특성 수] 형태의 3D Numpy 배열로 변환함.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Exporting for B-MTGNN | Loading '{panel_csv}' ===")
    df = pd.read_csv(panel_csv)
    
    # 결측치 처리 (0.0으로 채움)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"[!] Found {nan_count} NaNs. Filling with 0.0 for tensor stability.")
        df.fillna(0.0, inplace=True)
    
    # 1. 딥러닝 입력 특성 정의 (GitHub/HF 제거, Google Trends 추가)
    feature_cols = [
        # Targets & Core Metrics
        'AFCI', 'stock_return', 
        # External Signals (News & Science)
        'news_count', 'news_sentiment', 'news_ai_exposure', 'paper_count',
        # External Signals (Public Attention - Google Trends)
        'gt_firm_attention', 'gt_ai_momentum',
        # Strategic Topic Exposures (NLP)
        'expo_topic_semiconductors', 'expo_topic_cloud', 'expo_topic_software',
        'expo_topic_hardware', 'expo_topic_advertising', 'expo_topic_social',
        'expo_topic_ecommerce', 'expo_topic_data', 'expo_topic_auto'
    ]
    
    # 실제 데이터셋에 존재하는 컬럼만 필터링
    existing_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(existing_cols)
    if missing_cols:
        print(f"[WARN] Missing expected columns: {missing_cols}")
    feature_cols = existing_cols
    
    firms = sorted(df['firm_id'].unique())
    dates = sorted(df['date'].unique())
    
    num_nodes = len(firms)
    seq_length = len(dates)
    num_features = len(feature_cols)
    
    print(f"  Detected: {num_nodes} nodes, {seq_length} timesteps, {num_features} features.")
    
    # 3D Numpy 배열 초기화 [N, T, D]
    data_3d = np.zeros((num_nodes, seq_length, num_features))
    
    firm_to_idx = {firm: i for i, firm in enumerate(firms)}
    date_to_idx = {date: i for i, date in enumerate(dates)}
    
    print("  Transforming panel to 3D Tensor")
    for _, row in df.iterrows():
        if row['firm_id'] in firm_to_idx and row['date'] in date_to_idx:
            n_idx = firm_to_idx[row['firm_id']]
            t_idx = date_to_idx[row['date']]
            data_3d[n_idx, t_idx, :] = row[feature_cols].values
        
    data_out_path = os.path.join(out_dir, "bmtgnn_data.npy")
    np.save(data_out_path, data_3d)
    print(f"  - Saved 3D Feature Tensor {data_3d.shape} to '{data_out_path}'")
    
    # 노드 ID 매핑 저장
    pd.DataFrame({'node_id': range(num_nodes), 'firm_id': firms}).to_csv(
        os.path.join(out_dir, "node_ids.csv"), index=False
    )
    
    # 2. 인접 행렬(Adjacency Matrix) 생성
    if os.path.exists(edges_csv):
        print(f"\nLoading '{edges_csv}' for Adjacency Matrix")
        edges = pd.read_csv(edges_csv)
        adj_mat = np.zeros((num_nodes, num_nodes))
        
        for _, row in edges.iterrows():
            if row['source_id'] in firm_to_idx and row['target_id'] in firm_to_idx:
                src_idx = firm_to_idx[row['source_id']]
                tgt_idx = firm_to_idx[row['target_id']]
                adj_mat[src_idx, tgt_idx] = row['weight']
                
        adj_out_path = os.path.join(out_dir, "adj_mat.csv")
        pd.DataFrame(adj_mat, index=firms, columns=firms).to_csv(adj_out_path)
        print(f"  - Saved {num_nodes}x{num_nodes} Adjacency Matrix to '{adj_out_path}'")
    else:
        print(f"\n[WARN] {edges_csv} not found. Adjacency Matrix export skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export B-MTGNN PyTorch format tensors")
    parser.add_argument("--panel", type=str, default="../data_preparation/data/3_dataset/data_set.csv")
    parser.add_argument("--edges", type=str, default="../data_preparation/data/4_graph/edges_static_prior.csv")
    parser.add_argument("--out-dir", type=str, default="../data_preparation/data/5_bmtgnn_input")
    
    args = parser.parse_args()
    export_for_bmtgnn(args.panel, args.edges, args.out_dir)
