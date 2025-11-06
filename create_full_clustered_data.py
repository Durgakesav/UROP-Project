"""
Create Full Clustered Data for All Datasets
This script creates CSV files with cluster labels for phone, buy, and restaurant datasets
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import gower

def create_clustered_data():
    """
    Create full clustered data files for all datasets
    """
    print("CREATING FULL CLUSTERED DATA FOR ALL DATASETS")
    print("=" * 60)
    
    datasets = ['phone', 'buy', 'restaurant']
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        # Load training data
        train_file = f"train_sets/{dataset}_train_original.csv"
        if not os.path.exists(train_file):
            print(f"  ERROR: Training file not found: {train_file}")
            continue
        
        df = pd.read_csv(train_file)
        print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Load cluster info to get parameters
        cluster_file = f"clustering_results/cluster_info_{dataset}.json"
        if not os.path.exists(cluster_file):
            print(f"  ERROR: Cluster info not found: {cluster_file}")
            continue
        
        with open(cluster_file, 'r') as f:
            cluster_info = json.load(f)
        
        n_clusters = cluster_info.get('n_clusters', 0)
        print(f"  Expected clusters: {n_clusters}")
        
        if n_clusters == 0:
            print(f"  WARNING: No clusters found for {dataset} - creating noise-only data")
            df['cluster_label'] = -1  # All noise
            df['is_outlier'] = True
        else:
            # Apply DBSCAN clustering to get cluster labels
            try:
                # Prepare data for clustering
                df_cluster = df.copy()
                
                # Handle mixed data types
                numeric_cols = df_cluster.select_dtypes(include=[np.number]).columns
                categorical_cols = df_cluster.select_dtypes(include=['object']).columns
                
                print(f"  Numeric columns: {len(numeric_cols)}")
                print(f"  Categorical columns: {len(categorical_cols)}")
                
                # Calculate Gower distance matrix
                print("  Calculating Gower distance matrix...")
                distance_matrix = gower.gower_matrix(df_cluster)
                
                # Apply DBSCAN
                print("  Applying DBSCAN clustering...")
                dbscan = DBSCAN(eps=1.0, min_samples=3, metric="precomputed")
                cluster_labels = dbscan.fit_predict(distance_matrix)
                
                df['cluster_label'] = cluster_labels
                df['is_outlier'] = cluster_labels == -1
                
                actual_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                noise_points = sum(cluster_labels == -1)
                
                print(f"  SUCCESS: {actual_clusters} clusters, {noise_points} noise points")
                
            except Exception as e:
                print(f"  ERROR: Clustering failed - {e}")
                df['cluster_label'] = -1
                df['is_outlier'] = True
        
        # Save clustered data
        output_file = f"clustering_results/{dataset}_with_clusters.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Show cluster distribution
        if 'cluster_label' in df.columns:
            cluster_counts = df['cluster_label'].value_counts().sort_index()
            print(f"  Cluster distribution:")
            for cluster, count in cluster_counts.items():
                cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
                print(f"    {cluster_name}: {count} records")
    
    print(f"\n{'='*60}")
    print("FULL CLUSTERED DATA CREATION COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_clustered_data()













