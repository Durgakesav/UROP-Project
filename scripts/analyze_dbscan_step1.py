"""
Step 1: Complete DBSCAN Analysis for All Datasets
Analyze clustering results, centroids, and cluster characteristics for each dataset
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from pathlib import Path

class DBSCANStep1Analyzer:
    def __init__(self):
        self.results = {}
        self.scaler = StandardScaler()
        
    def analyze_dataset(self, dataset_name, train_file):
        """
        Complete DBSCAN analysis for a single dataset
        """
        print(f"\n{'='*80}")
        print(f"STEP 1: DBSCAN ANALYSIS - {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load training data
        df = pd.read_csv(train_file)
        print(f"Dataset: {dataset_name}")
        print(f"Original size: {len(df)} rows, {len(df.columns)} columns")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric columns: {len(numeric_cols)} - {numeric_cols}")
        print(f"Categorical columns: {len(categorical_cols)} - {categorical_cols[:5]}...")
        
        # Prepare data for clustering
        if len(numeric_cols) > 0:
            # Use numeric data for clustering
            df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
            X = self.scaler.fit_transform(df_numeric)
            clustering_method = "Numeric"
        else:
            # Use categorical data for clustering
            print("No numeric columns - using categorical clustering")
            le = LabelEncoder()
            df_encoded = df.copy()
            for col in categorical_cols:
                df_encoded[col] = le.fit_transform(df[col].astype(str))
            X = df_encoded.values
            clustering_method = "Categorical"
        
        # Apply DBSCAN with different parameters
        dbscan_results = {}
        
        # Test different eps values
        eps_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        min_samples_values = [3, 5, 7, 10]
        
        best_config = None
        best_score = -1
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    clustering = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = clustering.fit_predict(X)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    noise_ratio = n_noise / len(cluster_labels)
                    
                    # Score based on cluster quality
                    score = n_clusters * (1 - noise_ratio)  # More clusters, less noise = better
                    
                    config_key = f"eps_{eps}_min_{min_samples}"
                    dbscan_results[config_key] = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'score': score,
                        'cluster_labels': cluster_labels.tolist()
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_config = config_key
                        
                except Exception as e:
                    print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        # Use best configuration
        if best_config:
            best_result = dbscan_results[best_config]
            cluster_labels = np.array(best_result['cluster_labels'])
            
            print(f"\nBest DBSCAN Configuration:")
            print(f"  eps: {best_result['eps']}")
            print(f"  min_samples: {best_result['min_samples']}")
            print(f"  Clusters: {best_result['n_clusters']}")
            print(f"  Noise points: {best_result['n_noise']} ({best_result['noise_ratio']:.2%})")
            print(f"  Quality score: {best_result['score']:.2f}")
            
            # Analyze clusters
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)
            
            clusters_analysis = {}
            centroids = {}
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_df = df[cluster_mask]
                
                # Calculate centroid
                centroid = {}
                for col in df.columns:
                    if col in numeric_cols:
                        centroid[col] = float(cluster_df[col].mean()) if not cluster_df[col].empty else None
                    else:
                        mode_val = cluster_df[col].mode()
                        centroid[col] = mode_val.iloc[0] if not mode_val.empty else cluster_df[col].iloc[0]
                
                centroids[cluster_id] = centroid
                
                # Cluster characteristics
                cluster_analysis = {
                    'size': len(cluster_df),
                    'percentage': len(cluster_df) / len(df) * 100,
                    'centroid': centroid,
                    'sample_data': cluster_df.head(3).to_dict('records')
                }
                
                clusters_analysis[cluster_id] = cluster_analysis
                
                print(f"\n  Cluster {cluster_id}:")
                print(f"    Size: {len(cluster_df)} points ({len(cluster_df)/len(df)*100:.1f}%)")
                print(f"    Centroid key values:")
                for key, value in list(centroid.items())[:5]:  # Show first 5 centroid values
                    print(f"      {key}: {value}")
            
            # Store results
            self.results[dataset_name] = {
                'dataset_info': {
                    'name': dataset_name,
                    'original_size': len(df),
                    'n_columns': len(df.columns),
                    'numeric_cols': numeric_cols,
                    'categorical_cols': categorical_cols,
                    'clustering_method': clustering_method
                },
                'dbscan_config': {
                    'eps': best_result['eps'],
                    'min_samples': best_result['min_samples'],
                    'n_clusters': best_result['n_clusters'],
                    'n_noise': best_result['n_noise'],
                    'noise_ratio': best_result['noise_ratio'],
                    'quality_score': best_result['score']
                },
                'clusters': clusters_analysis,
                'centroids': centroids,
                'all_configs': dbscan_results
            }
            
            return df, cluster_labels, unique_clusters
            
        else:
            print("No valid DBSCAN configuration found")
            return None, None, None
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive analysis report
        """
        report = []
        report.append("="*100)
        report.append("STEP 1: COMPREHENSIVE DBSCAN CLUSTERING ANALYSIS")
        report.append("="*100)
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Datasets Analyzed: {len(self.results)}")
        report.append("")
        
        # Overall summary
        total_clusters = sum(r['dbscan_config']['n_clusters'] for r in self.results.values())
        total_noise = sum(r['dbscan_config']['n_noise'] for r in self.results.values())
        total_points = sum(r['dataset_info']['original_size'] for r in self.results.values())
        
        report.append("OVERALL SUMMARY")
        report.append("-" * 50)
        report.append(f"Total datasets: {len(self.results)}")
        report.append(f"Total data points: {total_points:,}")
        report.append(f"Total clusters found: {total_clusters}")
        report.append(f"Total noise points: {total_noise:,} ({total_noise/total_points*100:.2f}%)")
        report.append("")
        
        # Dataset-by-dataset analysis
        for dataset_name, result in self.results.items():
            report.append(f"DATASET: {dataset_name.upper()}")
            report.append("=" * 60)
            
            # Dataset info
            info = result['dataset_info']
            report.append(f"Original size: {info['original_size']:,} rows, {info['n_columns']} columns")
            report.append(f"Clustering method: {info['clustering_method']}")
            report.append(f"Numeric columns: {len(info['numeric_cols'])}")
            report.append(f"Categorical columns: {len(info['categorical_cols'])}")
            report.append("")
            
            # DBSCAN configuration
            config = result['dbscan_config']
            report.append("DBSCAN Configuration:")
            report.append(f"  eps: {config['eps']}")
            report.append(f"  min_samples: {config['min_samples']}")
            report.append(f"  Clusters found: {config['n_clusters']}")
            report.append(f"  Noise points: {config['n_noise']:,} ({config['noise_ratio']:.2%})")
            report.append(f"  Quality score: {config['quality_score']:.2f}")
            report.append("")
            
            # Cluster analysis
            report.append("CLUSTER ANALYSIS:")
            report.append("-" * 30)
            
            for cluster_id, cluster_info in result['clusters'].items():
                report.append(f"Cluster {cluster_id}:")
                report.append(f"  Size: {cluster_info['size']:,} points ({cluster_info['percentage']:.1f}%)")
                report.append(f"  Centroid (key values):")
                
                # Show important centroid values
                centroid = cluster_info['centroid']
                for key, value in list(centroid.items())[:8]:  # Show first 8 values
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        report.append(f"    {key}: {value:.2f}")
                    else:
                        report.append(f"    {key}: {value}")
                
                report.append("")
            
            report.append("")
        
        # Comparative analysis
        report.append("COMPARATIVE ANALYSIS")
        report.append("=" * 50)
        
        # Best performing dataset
        best_dataset = max(self.results.items(), 
                          key=lambda x: x[1]['dbscan_config']['quality_score'])
        report.append(f"Best clustering performance: {best_dataset[0].upper()}")
        report.append(f"  Quality score: {best_dataset[1]['dbscan_config']['quality_score']:.2f}")
        report.append(f"  Clusters: {best_dataset[1]['dbscan_config']['n_clusters']}")
        report.append(f"  Noise ratio: {best_dataset[1]['dbscan_config']['noise_ratio']:.2%}")
        report.append("")
        
        # Dataset characteristics
        report.append("Dataset Characteristics:")
        for dataset_name, result in self.results.items():
            config = result['dbscan_config']
            info = result['dataset_info']
            report.append(f"  {dataset_name}:")
            report.append(f"    Size: {info['original_size']:,} rows")
            report.append(f"    Clusters: {config['n_clusters']}")
            report.append(f"    Noise: {config['noise_ratio']:.2%}")
            report.append(f"    Method: {info['clustering_method']}")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("=" * 50)
        report.append("1. Use cluster-specific data for LLM1 (cluster-based imputation)")
        report.append("2. Use full dataset for LLM2 (RAG-based imputation)")
        report.append("3. Consider cluster quality when assigning missing rows to clusters")
        report.append("4. Monitor noise points - they may need special handling")
        report.append("")
        report.append("="*100)
        
        return "\n".join(report)
    
    def save_results(self):
        """
        Save all results to files
        """
        # Save detailed JSON results (convert numpy types to Python types)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(self.results)
        with open("step1_dbscan_complete_results.json", "w") as f:
            json.dump(converted_results, f, indent=2)
        
        # Generate and save comprehensive report
        report = self.generate_comprehensive_report()
        with open("step1_dbscan_analysis_report.txt", "w") as f:
            f.write(report)
        
        print(f"\nResults saved to:")
        print(f"  - step1_dbscan_complete_results.json")
        print(f"  - step1_dbscan_analysis_report.txt")

# Main execution
if __name__ == "__main__":
    analyzer = DBSCANStep1Analyzer()
    
    # Define datasets
    datasets = {
        "buy": "train_sets/buy_train_original.csv",
        "phone": "train_sets/phone_train_original.csv", 
        "restaurant": "train_sets/restaurant_train_original.csv",
        "zomato": "train_sets/zomato_train_original.csv"
    }
    
    print("="*100)
    print("STEP 1: COMPLETE DBSCAN CLUSTERING ANALYSIS")
    print("="*100)
    print("Analyzing all datasets for optimal clustering configuration...")
    
    # Analyze each dataset
    for dataset_name, train_file in datasets.items():
        try:
            df, cluster_labels, clusters = analyzer.analyze_dataset(dataset_name, train_file)
            if df is not None:
                print(f"[SUCCESS] {dataset_name.upper()} analysis completed")
            else:
                print(f"[FAILED] {dataset_name.upper()} analysis failed")
        except Exception as e:
            print(f"[ERROR] Error analyzing {dataset_name}: {e}")
    
    # Generate comprehensive report
    print(f"\n{'='*100}")
    print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print(f"{'='*100}")
    
    analyzer.save_results()
    
    print(f"\n{'='*100}")
    print("STEP 1 ANALYSIS COMPLETE!")
    print(f"{'='*100}")
    print(f"Analyzed {len(analyzer.results)} datasets")
    print("Check the generated files for detailed results.")
