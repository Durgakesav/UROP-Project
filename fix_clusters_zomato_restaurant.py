"""
Fixed clustering script to ensure at least 2 clusters for zomato and restaurant datasets
Uses multiple clustering strategies with parameter optimization
"""

import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN, KMedoids
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import json
from datetime import datetime

def preprocess_for_clustering(df, dataset_name):
    """Preprocess data for clustering"""
    print(f"\nPreprocessing {dataset_name} dataset...")
    print(f"Original shape: {df.shape}")
    
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna('Unknown')
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    print(f"Processed shape: {df_processed.shape}")
    return df_processed

def try_dbscan_with_expanded_parameters(df_processed, dataset_name):
    """Try DBSCAN with expanded parameter ranges"""
    print(f"\n{'='*70}")
    print(f"DBSCAN Clustering: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Calculate Gower distance matrix
    print("\nCalculating Gower distance matrix...")
    try:
        dist_matrix = gower.gower_matrix(df_processed)
        print(f"Distance matrix shape: {dist_matrix.shape}")
    except Exception as e:
        print(f"Error calculating Gower distance: {e}")
        return None
    
    # Expanded parameter ranges to encourage more clusters
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    min_samples_values = [2, 3, 4, 5]
    
    best_result = None
    best_score = -1
    
    print(f"\nTesting {len(eps_values)} x {len(min_samples_values)} = {len(eps_values) * len(min_samples_values)} parameter combinations...")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                labels = db.fit_predict(dist_matrix)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                
                # Calculate score that rewards more clusters and fewer outliers
                # But penalizes too many tiny clusters
                if n_clusters >= 2:  # Only consider if we have at least 2 clusters
                    # Calculate average cluster size
                    non_outliers = len(labels) - n_outliers
                    avg_cluster_size = non_outliers / n_clusters if n_clusters > 0 else 0
                    
                    # Penalize if average cluster size is too small (less than 10 points)
                    size_penalty = 0.5 if avg_cluster_size < 10 else 1.0
                    
                    # Score = (number of clusters) * (1 - outlier_ratio) * size_penalty
                    score = n_clusters * (1 - (n_outliers / len(labels))) * size_penalty
                    print(f"  eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, {n_outliers} outliers ({n_outliers/len(labels)*100:.1f}%), score={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_outliers': n_outliers,
                            'labels': labels,
                            'score': score,
                            'method': 'DBSCAN'
                        }
                else:
                    print(f"  eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters [insufficient]")
                    
            except Exception as e:
                print(f"  Error with eps={eps}, min_samples={min_samples}: {e}")
                continue
    
    if best_result and best_result['n_clusters'] >= 2:
        print(f"\n[BEST] Best DBSCAN: {best_result['n_clusters']} clusters, score={best_result['score']:.3f}")
        return best_result
    else:
        print(f"\n[FAIL] DBSCAN failed to produce at least 2 clusters")
        return None

def try_kmeans_clustering(df_processed, dataset_name):
    """Try KMeans clustering with different k values"""
    print(f"\n{'='*70}")
    print(f"K-Means Clustering: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Prepare data for K-means
    df_encoded = df_processed.copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    
    print("\nTesting K-means with k=2 to k=10...")
    
    best_result = None
    best_score = -1
    
    for k in range(2, min(11, len(df_processed) // 10 + 1)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_scaled)
            
            # Calculate silhouette score (intra-cluster distance)
            within_cluster_sse = 0
            for i in range(k):
                cluster_mask = labels == i
                if cluster_mask.sum() > 0:
                    cluster_data = df_scaled[cluster_mask]
                    centroid = kmeans.cluster_centers_[i]
                    within_cluster_sse += np.sum((cluster_data - centroid) ** 2)
            
            # Score inversely proportional to within-cluster SSE
            score = 1 / (1 + within_cluster_sse / len(df_scaled))
            
            n_clusters = k
            n_outliers = 0  # K-means doesn't produce outliers
            
            print(f"  k={k}: {n_clusters} clusters, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_result = {
                    'k': k,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'labels': labels,
                    'score': score,
                    'method': 'KMeans',
                    'centroids': kmeans.cluster_centers_
                }
                
        except Exception as e:
            print(f"  Error with k={k}: {e}")
            continue
    
    if best_result:
        print(f"\n[BEST] Best K-Means: k={best_result['k']}, score={best_result['score']:.3f}")
    else:
        print(f"\n[FAIL] K-Means failed")
    
    return best_result

def cluster_dataset(dataset_name, file_path):
    """Cluster a dataset using multiple methods"""
    print(f"\n{'='*80}")
    print(f"CLUSTERING: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess
    df_processed = preprocess_for_clustering(df, dataset_name)
    
    # Try DBSCAN
    dbscan_result = try_dbscan_with_expanded_parameters(df_processed, dataset_name)
    
    # Try K-Means
    kmeans_result = try_kmeans_clustering(df_processed, dataset_name)
    
    # Choose best method
    # Prefer K-means if DBSCAN produced too many tiny clusters
    if dbscan_result and kmeans_result:
        # Check if DBSCAN produced reasonable clusters
        n_clusters_dbscan = dbscan_result['n_clusters']
        n_outliers_dbscan = dbscan_result['n_outliers']
        non_outliers = len(df) - n_outliers_dbscan
        avg_cluster_size = non_outliers / n_clusters_dbscan if n_clusters_dbscan > 0 else 0
        
        # If average cluster size is too small (<5 points), prefer K-means
        if avg_cluster_size < 5:
            print(f"\n[INFO] DBSCAN produced too many tiny clusters (avg size={avg_cluster_size:.1f}). Choosing K-Means.")
            best_result = kmeans_result
        elif dbscan_result['score'] > kmeans_result['score']:
            best_result = dbscan_result
        else:
            best_result = kmeans_result
    elif dbscan_result:
        best_result = dbscan_result
    elif kmeans_result:
        best_result = kmeans_result
    else:
        print("\n[FAILED] ALL CLUSTERING METHODS FAILED")
        return None
    
    # Add labels to original dataframe
    df_with_labels = df.copy()
    df_with_labels['cluster_label'] = best_result['labels']
    df_with_labels['is_outlier'] = best_result['labels'] == -1
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df_with_labels, dataset_name, best_result['method'])
    
    return {
        'dataset_name': dataset_name,
        'method': best_result['method'],
        'result': best_result,
        'df_with_labels': df_with_labels,
        'cluster_analysis': cluster_analysis
    }

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

def analyze_clusters(df_with_labels, dataset_name, method):
    """Analyze clustering results"""
    print(f"\nAnalyzing {method} clusters for {dataset_name}...")
    
    cluster_info = {}
    n_clusters = len(set(df_with_labels['cluster_label'])) - (1 if -1 in df_with_labels['cluster_label'] else 0)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of outliers: {df_with_labels['is_outlier'].sum()}")
    
    for cluster_id in sorted(set(df_with_labels['cluster_label'])):
        if cluster_id == -1:
            continue
        
        cluster_data = df_with_labels[df_with_labels['cluster_label'] == cluster_id]
        cluster_size = len(cluster_data)
        
        print(f"\nCluster {cluster_id}: {cluster_size} points ({cluster_size/len(df_with_labels)*100:.1f}%)")
        
        # Analyze characteristics
        cluster_characteristics = {}
        for col in cluster_data.columns:
            if col not in ['cluster_label', 'is_outlier']:
                if cluster_data[col].dtype == 'object':
                    top_values = cluster_data[col].value_counts().head(3)
                    cluster_characteristics[col] = {
                        'type': 'categorical',
                        'top_values': top_values.to_dict()
                    }
                else:
                    cluster_characteristics[col] = {
                        'type': 'numerical',
                        'mean': cluster_data[col].mean(),
                        'std': cluster_data[col].std(),
                        'min': cluster_data[col].min(),
                        'max': cluster_data[col].max()
                    }
        
        cluster_info[cluster_id] = {
            'size': cluster_size,
            'characteristics': cluster_characteristics
        }
    
    return cluster_info

def save_results(results, output_dir="clustering_results"):
    """Save clustering results"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for result in results:
        if result is None:
            continue
            
        dataset_name = result['dataset_name']
        
        # Save CSV with clusters
        df_with_labels = result['df_with_labels']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_dir}/{dataset_name}_with_clusters_{timestamp}.csv"
        df_with_labels.to_csv(output_filename, index=False)
        print(f"  Saved CSV: {output_filename}")
        
        # Save JSON metadata
        json_data = {
            'dataset': dataset_name,
            'method': result['method'],
            'n_clusters': int(result['result']['n_clusters']),
            'n_outliers': int(result['result']['n_outliers']),
            'score': float(result['result']['score']),
            'cluster_analysis': convert_to_native_types(result['cluster_analysis'])
        }
        
        with open(f"{output_dir}/cluster_info_{dataset_name}_{timestamp}.json", "w") as f:
            json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")

def main():
    """Main function"""
    print("="*80)
    print("FIXED CLUSTERING FOR ZOMATO AND RESTAURANT DATASETS")
    print("="*80)
    
    # Define datasets to process
    datasets = {
        "phone": "phone.csv",
        "buy": "buy.csv",
        "zomato": "zomato.csv",
        "restaurant": "restaurant.csv"
    }
    
    all_results = []
    
    for name, file in datasets.items():
        result = cluster_dataset(name, file)
        all_results.append(result)
        
        if result:
            print(f"\n[SUCCESS] {name.upper()} clustered successfully with {result['method']}")
            print(f"  Clusters found: {result['result']['n_clusters']}")
            print(f"  Outliers: {result['result']['n_outliers']}")
        else:
            print(f"\n[FAIL] {name.upper()} clustering failed")
    
    # Save results
    save_results(all_results)
    
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

