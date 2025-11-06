"""
Apply DBSCAN clustering with Gower distance on training data
Create clusters, centroids, and save in format accessible by LLM1
"""

import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN
from pathlib import Path
import json
from datetime import datetime
import os

# Dataset configuration
DATASETS = {
    'buy': {
        'train_file': 'train_sets/buy_train_original.csv',
        'eps_values': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
        'min_samples_values': [2, 3, 5]
    },
    'phone': {
        'train_file': 'train_sets/phone_train_original.csv',
        'eps_values': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
        'min_samples_values': [2, 3, 5]
    },
    'restaurant': {
        'train_file': 'train_sets/restaurant_train_original.csv',
        'eps_values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
        'min_samples_values': [2, 3, 5]
    },
    'zomato': {
        'train_file': 'train_sets/zomato_train_original.csv',
        'eps_values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
        'min_samples_values': [2, 3, 5]
    }
}

def preprocess_data(df, dataset_name):
    """Preprocess data for clustering"""
    print(f"\nPreprocessing {dataset_name} dataset...")
    df_processed = df.copy()
    
    # Handle missing values
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Unknown')
    
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def calculate_centroid(cluster_df, all_columns):
    """Calculate centroid: mean for numeric, mode for categorical"""
    centroid = {}
    for col in all_columns:
        if col in cluster_df.columns:
            if cluster_df[col].dtype in [np.number, 'int64', 'float64']:
                # Numerical: use mean
                centroid[col] = float(cluster_df[col].mean()) if not cluster_df[col].empty else None
            else:
                # Categorical: use mode
                mode_val = cluster_df[col].mode()
                centroid[col] = mode_val.iloc[0] if not mode_val.empty else None
        else:
            centroid[col] = None
    return centroid

def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def cluster_dataset(dataset_name, config):
    """Cluster a dataset and save results"""
    print(f"\n{'='*70}")
    print(f"CLUSTERING {dataset_name.upper()} DATASET")
    print(f"{'='*70}")
    
    # Load training data
    train_file = config['train_file']
    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found")
        return None
    
    df = pd.read_csv(train_file)
    print(f"Training data: {len(df)} rows, {len(df.columns)} columns")
    
    # Preprocess
    df_processed = preprocess_data(df, dataset_name)
    
    # Calculate Gower distance matrix
    print("\nCalculating Gower distance matrix...")
    try:
        dist_matrix = gower.gower_matrix(df_processed)
        print(f"Distance matrix shape: {dist_matrix.shape}")
    except Exception as e:
        print(f"ERROR calculating Gower distance: {e}")
        return None
    
    # Try different parameter combinations
    best_result = None
    best_score = -1
    eps_values = config['eps_values']
    min_samples_values = config['min_samples_values']
    
    print(f"\nTesting {len(eps_values)} x {len(min_samples_values)} = {len(eps_values) * len(min_samples_values)} parameter combinations...")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Apply DBSCAN
                db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                labels = db.fit_predict(dist_matrix)
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                
                if n_clusters < 2:
                    continue  # Need at least 2 clusters
                
                # Calculate quality score (prefer more clusters, fewer outliers, reasonable sizes)
                avg_cluster_size = (len(df) - n_outliers) / n_clusters if n_clusters > 0 else 0
                
                # Score: higher is better
                # Penalize too many outliers, too many tiny clusters, reward good cluster sizes
                if avg_cluster_size < 5:
                    score = 0.1  # Too many tiny clusters
                else:
                    outlier_ratio = n_outliers / len(df)
                    score = n_clusters / (1 + outlier_ratio * 10) * (avg_cluster_size / 100)
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'labels': labels,
                        'n_clusters': n_clusters,
                        'n_outliers': n_outliers,
                        'score': score,
                        'avg_cluster_size': avg_cluster_size
                    }
                    
                    print(f"  eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, {n_outliers} outliers, score={score:.4f}")
            
            except Exception as e:
                print(f"  ERROR with eps={eps}, min_samples={min_samples}: {e}")
                continue
    
    if best_result is None:
        print("ERROR: No valid clustering found")
        return None
    
    print(f"\nBest configuration:")
    print(f"  eps: {best_result['eps']}")
    print(f"  min_samples: {best_result['min_samples']}")
    print(f"  Clusters: {best_result['n_clusters']}")
    print(f"  Outliers: {best_result['n_outliers']} ({best_result['n_outliers']/len(df)*100:.1f}%)")
    print(f"  Score: {best_result['score']:.4f}")
    
    # Add cluster labels to dataframe (use PREPROCESSED data to avoid NaNs in saved clusters)
    df_with_clusters = df_processed.copy()
    df_with_clusters['cluster_label'] = best_result['labels']
    df_with_clusters['is_outlier'] = (best_result['labels'] == -1)
    
    # Calculate centroids for each cluster
    centroids = {}
    cluster_analysis = {}
    
    unique_clusters = set(best_result['labels'])
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    for cluster_id in sorted(unique_clusters):
        cluster_mask = best_result['labels'] == cluster_id
        # Use PREPROCESSED rows for centroid calculation to avoid NaNs
        cluster_df = df_processed[cluster_mask]
        
        # Calculate centroid
        centroid = calculate_centroid(cluster_df, df.columns)
        centroids[str(cluster_id)] = convert_to_python_types(centroid)
        
        # Cluster characteristics
        cluster_analysis[str(cluster_id)] = {
            'size': len(cluster_df),
            'centroid': centroids[str(cluster_id)]
        }
        
        print(f"\n  Cluster {cluster_id}: {len(cluster_df)} points")
        print(f"    Centroid sample: {list(centroid.items())[:3]}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cluster info JSON (for LLM1)
    cluster_info = {
        'dataset': dataset_name,
        'method': 'DBSCAN',
        'n_clusters': best_result['n_clusters'],
        'n_outliers': best_result['n_outliers'],
        'score': float(best_result['score']),
        'cluster_analysis': cluster_analysis,
        'best_config': {
            'eps': float(best_result['eps']),
            'min_samples': int(best_result['min_samples'])
        }
    }
    
    cluster_info_file = f"{output_dir}/cluster_info_{dataset_name}_{timestamp}.json"
    with open(cluster_info_file, 'w') as f:
        json.dump(cluster_info, f, indent=2, default=str)
    print(f"\n[OK] Saved cluster info: {cluster_info_file}")
    
    # Also save with standard name for LLM1
    cluster_info_standard = f"{output_dir}/cluster_info_{dataset_name}.json"
    with open(cluster_info_standard, 'w') as f:
        json.dump(cluster_info, f, indent=2, default=str)
    print(f"[OK] Saved cluster info (standard): {cluster_info_standard}")
    
    # Save training data with cluster labels (preprocessed, NaN-free where possible)
    clustered_file = f"{output_dir}/{dataset_name}_with_clusters_{timestamp}.csv"
    df_with_clusters.to_csv(clustered_file, index=False)
    print(f"[OK] Saved clustered data: {clustered_file}")
    
    # Also save with standard name for LLM1
    clustered_standard = f"{output_dir}/{dataset_name}_with_clusters.csv"
    df_with_clusters.to_csv(clustered_standard, index=False)
    print(f"[OK] Saved clustered data (standard): {clustered_standard}")

    # Optionally save raw-with-labels for auditing
    raw_with_labels = df.copy()
    raw_with_labels['cluster_label'] = best_result['labels']
    raw_with_labels['is_outlier'] = (best_result['labels'] == -1)
    raw_file = f"{output_dir}/{dataset_name}_with_clusters_raw_{timestamp}.csv"
    raw_with_labels.to_csv(raw_file, index=False)
    print(f"[OK] Saved raw clustered data (audit): {raw_file}")
    
    return cluster_info

def main():
    """Main function to cluster all training datasets"""
    print("=" * 70)
    print("DBSCAN CLUSTERING WITH GOWER DISTANCE")
    print("Training Data Only (70% of original)")
    print("=" * 70)
    
    results = {}
    for dataset_name, config in DATASETS.items():
        result = cluster_dataset(dataset_name, config)
        results[dataset_name] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("CLUSTERING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Clusters':<10} {'Outliers':<10} {'Status':<10}")
    print("-" * 70)
    
    for dataset_name, result in results.items():
        if result:
            status = "[OK]"
            clusters = result['n_clusters']
            outliers = result['n_outliers']
        else:
            status = "[FAILED]"
            clusters = 0
            outliers = 0
        
        print(f"{dataset_name:<12} {clusters:<10} {outliers:<10} {status:<10}")
    
    print(f"\n{'='*70}")
    print("FILES CREATED FOR LLM1:")
    print("1. cluster_info_{dataset}.json - Centroids for each cluster")
    print("2. {dataset}_with_clusters.csv - Training data with cluster labels")
    print("=" * 70)

if __name__ == "__main__":
    main()

