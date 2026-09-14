import argparse
import os
import pandas as pd
import numpy as np

# Convert the panel into a feature tensor and firm IDs. The model learns its
# own graph; no external adjacency is generated, read or required.

def export_for_bmtgnn(panel_csv: str, out_dir: str):
    """Export the monthly panel as a [firm, month, feature] array and firm-ID mapping."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Exporting for B-MTGNN | Loading '{panel_csv}' ===")
    df = pd.read_csv(panel_csv)
    
    # Fill missing values with zero before conversion.
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"[!] Found {nan_count} NaNs. Filling with 0.0 for tensor stability.")
        df.fillna(0.0, inplace=True)
    
    # Fixed feature order expected by the model; AFCI is channel zero.
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
    
    # Require every feature in the expected input schema.
    existing_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(existing_cols)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    feature_cols = existing_cols
    
    firms = sorted(df['firm_id'].unique())
    dates = sorted(df['date'].unique())
    
    num_nodes = len(firms)
    seq_length = len(dates)
    num_features = len(feature_cols)
    
    print(f"  Detected: {num_nodes} nodes, {seq_length} timesteps, {num_features} features.")
    
    # Allocate the [firm, month, feature] tensor.
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
    
    # Save the firm order used by the tensor.
    pd.DataFrame({'node_id': range(num_nodes), 'firm_id': firms}).to_csv(
        os.path.join(out_dir, "node_ids.csv"), index=False
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export B-MTGNN PyTorch format tensors")
    parser.add_argument("--panel", type=str, default="../data_preparation/data/3_dataset/data_set.csv")
    parser.add_argument("--out-dir", type=str, default="../data_preparation/data/5_bmtgnn_input")
    
    args = parser.parse_args()
    export_for_bmtgnn(args.panel, args.out_dir)
