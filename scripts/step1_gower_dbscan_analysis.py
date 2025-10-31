"""
Step 1: DBSCAN Clustering with Gower Distance for Mixed Data Types
This script applies DBSCAN clustering using Gower distance to handle both categorical and numerical data
"""

import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def preprocess_data_for_clustering(df, dataset_name):
    """
    Preprocess data for clustering, handling missing values and data types
    """
    print(f"\nPreprocessing {dataset_name} dataset...")
    print(f"Original shape: {df.shape}")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Handle missing values
    missing_before = df_processed.isnull().sum().sum()
    
    # For categorical columns, fill with 'Unknown'
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Unknown')
    
    # For numerical columns, fill with median
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    missing_after = df_processed.isnull().sum().sum()
    print(f"Missing values: {missing_before} -> {missing_after}")
    
    # Convert all columns to appropriate types for Gower distance
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Keep as object for categorical
            pass
        else:
            # Ensure numerical columns are numeric
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    print(f"Processed shape: {df_processed.shape}")
    print(f"Data types: {df_processed.dtypes.value_counts()}")
    
    return df_processed

def apply_gower_dbscan_clustering(df, dataset_name, eps_values=[0.3, 0.5, 0.7], min_samples_values=[2, 3, 5]):
    """
    Apply DBSCAN clustering with Gower distance for mixed data types
    """
    print(f"\n{'='*70}")
    print(f"DBSCAN Clustering with Gower Distance: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Preprocess data
    df_processed = preprocess_data_for_clustering(df, dataset_name)
    
    # Calculate Gower distance matrix
    print("\nCalculating Gower distance matrix...")
    print("This may take a while for large datasets...")
    
    try:
        dist_matrix = gower.gower_matrix(df_processed)
        print(f"Distance matrix shape: {dist_matrix.shape}")
    except Exception as e:
        print(f"Error calculating Gower distance: {e}")
        return None
    
    # Try different parameter combinations
    best_result = None
    best_score = -1
    all_results = []
    
    print(f"\nTesting {len(eps_values)} x {len(min_samples_values)} = {len(eps_values) * len(min_samples_values)} parameter combinations...")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Apply DBSCAN with precomputed distance matrix
                db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                labels = db.fit_predict(dist_matrix)
                
                # Calculate clustering metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                
                # Calculate silhouette-like score (simplified)
                if n_clusters > 1:
                    # Calculate average intra-cluster distance
                    intra_cluster_distances = []
                    for cluster_id in set(labels):
                        if cluster_id != -1:  # Skip noise points
                            cluster_mask = labels == cluster_id
                            cluster_distances = dist_matrix[cluster_mask][:, cluster_mask]
                            if cluster_distances.size > 0:
                                # Average distance within cluster
                                avg_intra = np.mean(cluster_distances[np.triu_indices_from(cluster_distances, k=1)])
                                intra_cluster_distances.append(avg_intra)
                    
                    avg_intra_cluster = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
                    score = 1 / (1 + avg_intra_cluster)  # Higher is better
                else:
                    score = 0
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'outlier_percentage': (n_outliers / len(labels)) * 100,
                    'score': score,
                    'labels': labels
                }
                
                all_results.append(result)
                
                print(f"  eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, {n_outliers} outliers ({result['outlier_percentage']:.1f}%), score={score:.3f}")
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_result = result
                    
            except Exception as e:
                print(f"  Error with eps={eps}, min_samples={min_samples}: {e}")
                continue
    
    if best_result is None:
        print("No valid clustering results found!")
        return None
    
    print(f"\nBest parameters: eps={best_result['eps']}, min_samples={best_result['min_samples']}")
    print(f"Best result: {best_result['n_clusters']} clusters, {best_result['n_outliers']} outliers ({best_result['outlier_percentage']:.1f}%)")
    
    # Add labels to original dataframe
    df_with_labels = df.copy()
    df_with_labels['cluster_label'] = best_result['labels']
    df_with_labels['is_outlier'] = best_result['labels'] == -1
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df_with_labels, dataset_name)
    
    return {
        'dataset_name': dataset_name,
        'best_result': best_result,
        'all_results': all_results,
        'df_with_labels': df_with_labels,
        'cluster_analysis': cluster_analysis
    }

def analyze_clusters(df_with_labels, dataset_name):
    """
    Analyze the clustering results and provide insights
    """
    print(f"\nAnalyzing clusters for {dataset_name}...")
    
    cluster_info = {}
    n_clusters = len(set(df_with_labels['cluster_label'])) - (1 if -1 in df_with_labels['cluster_label'] else 0)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of outliers: {df_with_labels['is_outlier'].sum()}")
    
    # Analyze each cluster
    for cluster_id in sorted(set(df_with_labels['cluster_label'])):
        if cluster_id == -1:  # Skip outliers
            continue
            
        cluster_data = df_with_labels[df_with_labels['cluster_label'] == cluster_id]
        cluster_size = len(cluster_data)
        
        print(f"\nCluster {cluster_id}: {cluster_size} points")
        
        # Analyze cluster characteristics
        cluster_characteristics = {}
        for col in cluster_data.columns:
            if col not in ['cluster_label', 'is_outlier']:
                if cluster_data[col].dtype == 'object':
                    # Categorical analysis
                    top_values = cluster_data[col].value_counts().head(3)
                    cluster_characteristics[col] = {
                        'type': 'categorical',
                        'top_values': top_values.to_dict(),
                        'unique_count': cluster_data[col].nunique()
                    }
                else:
                    # Numerical analysis
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

def process_single_dataset(dataset_name, file_path):
    """
    Process a single dataset and save results
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Apply clustering
        result = apply_gower_dbscan_clustering(df, dataset_name)
        
        if result is not None:
            # Save results for this dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"step1_clustering_results_{timestamp}"
            Path(output_dir).mkdir(exist_ok=True)
            
            # Save labeled dataset
            result['df_with_labels'].to_csv(f"{output_dir}/{dataset_name}_with_clusters.csv", index=False)
            
            # Save cluster analysis
            with open(f"{output_dir}/{dataset_name}_cluster_analysis.json", "w") as f:
                json.dump(result['cluster_analysis'], f, indent=2, default=str)
            
            # Save detailed results
            detailed_result = {
                'dataset_name': result['dataset_name'],
                'best_result': {
                    'eps': result['best_result']['eps'],
                    'min_samples': result['best_result']['min_samples'],
                    'n_clusters': result['best_result']['n_clusters'],
                    'n_outliers': result['best_result']['n_outliers'],
                    'outlier_percentage': result['best_result']['outlier_percentage'],
                    'score': result['best_result']['score']
                },
                'all_results': [
                    {
                        'eps': r['eps'],
                        'min_samples': r['min_samples'],
                        'n_clusters': r['n_clusters'],
                        'n_outliers': r['n_outliers'],
                        'outlier_percentage': r['outlier_percentage'],
                        'score': r['score']
                    } for r in result['all_results']
                ]
            }
            
            with open(f"{output_dir}/{dataset_name}_detailed_results.json", "w") as f:
                json.dump(detailed_result, f, indent=2)
            
            print(f"\nResults saved to: {output_dir}/")
            return result
        else:
            print(f"No valid clustering results for {dataset_name}")
            return None
            
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return None

def main():
    """
    Main function to run DBSCAN clustering on all datasets
    """
    print("="*80)
    print("STEP 1: DBSCAN CLUSTERING WITH GOWER DISTANCE")
    print("="*80)
    print("\nThis script applies DBSCAN clustering using Gower distance")
    print("to handle both categorical and numerical data types.")
    
    # Define datasets
    datasets = {
        "buy": "buy.csv",
        "phone": "phone.csv", 
        "restaurant": "restaurant.csv",
        "zomato": "zomato.csv"
    }
    
    print("\nDatasets to process:")
    for name, file in datasets.items():
        print(f"  - {name}: {file}")
    
    print("\nStarting clustering analysis...")
    
    all_results = []
    
    for dataset_name, file_path in datasets.items():
        result = process_single_dataset(dataset_name, file_path)
        all_results.append(result)
    
    # Create final summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"step1_dbscan_complete_results_{timestamp}.json", "w") as f:
        summary_results = []
        for result in all_results:
            if result is not None:
                summary_results.append({
                    'dataset_name': result['dataset_name'],
                    'best_result': result['best_result'],
                    'cluster_analysis': result['cluster_analysis']
                })
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("STEP 1 CLUSTERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Summary results saved to: step1_dbscan_complete_results_{timestamp}.json")
    
    # Print summary
    successful_datasets = [r for r in all_results if r is not None]
    print(f"\nSuccessfully processed: {len(successful_datasets)}/{len(datasets)} datasets")
    
    for result in successful_datasets:
        print(f"  - {result['dataset_name']}: {result['best_result']['n_clusters']} clusters, {result['best_result']['n_outliers']} outliers")

if __name__ == "__main__":
    main()

