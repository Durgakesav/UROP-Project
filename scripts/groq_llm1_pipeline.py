"""
3-LLM Pipeline with Groq API for LLM1 (Cluster-based Imputation)
Step 1: DBSCAN clustering (completed)
Step 2: LLM1 - Groq API for cluster-based imputation
- LLM1 ONLY receives cluster-specific data + centroid
- No full dataset access
"""

import pandas as pd
import numpy as np
import json
from openai import OpenAI
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

class GroqLLM1Pipeline:
    def __init__(self, groq_api_key):
        """
        Initialize the pipeline with Groq API key
        """
        self.api_key = groq_api_key
        self.clusters = {}
        self.centroids = {}
        self.cluster_data = {}
        
        # Configure Groq API (OpenAI-compatible)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "openai/gpt-oss-20b"
        
        print("SUCCESS: Groq API configured successfully")
    
    def load_cluster_info(self, dataset_name):
        """
        Load pre-computed cluster information from Step 1
        """
        # Try standard filename first, then timestamped version
        cluster_file = f"clustering_results/cluster_info_{dataset_name}.json"
        
        if not os.path.exists(cluster_file):
            # Try to find any cluster_info file for this dataset
            import glob
            files = glob.glob(f"clustering_results/cluster_info_{dataset_name}_*.json")
            if files:
                cluster_file = files[0]  # Use most recent
            else:
                print(f"ERROR: Cluster file not found: {cluster_file}")
                return False
        
        with open(cluster_file, 'r') as f:
            cluster_info = json.load(f)
        
        # Extract centroids from cluster_analysis
        cluster_analysis = cluster_info.get('cluster_analysis', {})
        self.centroids = {}
        
        for cluster_id, cluster_info_data in cluster_analysis.items():
            if cluster_id == '-1':  # Skip outliers
                continue
            
            # Get centroid directly from cluster_analysis
            centroid = cluster_info_data.get('centroid', {})
            if centroid:
                self.centroids[cluster_id] = convert_to_python_types(centroid)
            else:
                # Fallback: extract from characteristics if centroid not available
                centroid = {}
                characteristics = cluster_info_data.get('characteristics', {})
                for col_name, col_info in characteristics.items():
                    if col_info.get('type') == 'categorical':
                        top_values = col_info.get('top_values', {})
                        if top_values:
                            centroid[col_name] = max(top_values, key=top_values.get)
                    elif col_info.get('type') == 'numerical':
                        centroid[col_name] = col_info.get('mean')
                
                self.centroids[cluster_id] = convert_to_python_types(centroid)
        
        print(f"SUCCESS: Loaded cluster info for {dataset_name}")
        print(f"   - Clusters: {len(self.centroids)}")
        print(f"   - Dataset: {cluster_info.get('dataset', 'unknown')}")
        
        return True
    
    def load_clustered_training_data(self, dataset_name):
        """
        Load clustered training data to get cluster members
        """
        # Try standard filename first, then timestamped version
        cluster_file = f"clustering_results/{dataset_name}_with_clusters.csv"
        
        if not os.path.exists(cluster_file):
            # Try to find any clustered file for this dataset
            import glob
            files = glob.glob(f"clustering_results/{dataset_name}_with_clusters_*.csv")
            if files:
                cluster_file = files[0]  # Use most recent
            else:
                print(f"ERROR: Clustered data file not found: {cluster_file}")
                return None
        
        df = pd.read_csv(cluster_file)
        
        # Group by cluster_label
        for cluster_id in df['cluster_label'].unique():
            if cluster_id == -1:  # Skip outliers
                continue
            
            cluster_df = df[df['cluster_label'] == cluster_id].copy()
            cluster_df = cluster_df.drop(columns=['cluster_label', 'is_outlier'], errors='ignore')
            self.cluster_data[str(cluster_id)] = cluster_df
        
        print(f"SUCCESS: Loaded clustered training data")
        print(f"   - Total clusters with data: {len(self.cluster_data)}")
        
        return df
    
    def assign_to_cluster(self, missing_row, target_column, training_df):
        """
        Dynamically assign missing row to most similar cluster
        Computes distance between missing row (excluding target column) and each centroid
        Returns: cluster_id, distance
        """
        if not self.centroids:
            return "0", 0.0
        
        # Calculate distance to each cluster centroid
        # EXCLUDE target_column from distance calculation
        best_cluster = "0"
        min_distance = float('inf')
        
        for cluster_id, centroid in self.centroids.items():
            # Compute distance excluding the target column (the one we're trying to impute)
            distance = self._calculate_similarity(missing_row, centroid, target_column, training_df.columns)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        print(f"Assigned to cluster {best_cluster} (distance: {min_distance:.4f})")
        print(f"  (Excluded target column '{target_column}' from distance calculation)")
        return best_cluster, min_distance
    
    def _calculate_similarity(self, row1, centroid, exclude_column, columns):
        """
        Calculate similarity between a row and cluster centroid
        EXCLUDES the target column from distance calculation
        Uses Hamming distance for categorical, normalized difference for numerical
        """
        if not isinstance(centroid, dict):
            return float('inf')
        
        distance = 0.0
        # Exclude target column and get common columns
        common_cols = (set(row1.index) & set(centroid.keys())) - {exclude_column}
        
        if not common_cols:
            return float('inf')
        
        valid_comparisons = 0
        for col in common_cols:
            val1 = row1[col]
            val2 = centroid[col]
            
            # Skip NaN values
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            valid_comparisons += 1
            
            # Categorical comparison (exact match = 0, mismatch = 1)
            if str(val1).lower() != str(val2).lower():
                distance += 1.0
        
        # Normalize by number of valid comparisons
        return distance / valid_comparisons if valid_comparisons > 0 else float('inf')
    
    def _get_cluster_data(self, cluster_id, training_df, dataset_name):
        """
        Get cluster member data - ONLY cluster-specific data, not full dataset
        """
        cluster_id_str = str(cluster_id)
        
        if cluster_id_str in self.cluster_data:
            cluster_df = self.cluster_data[cluster_id_str].copy()
            # Remove cluster_label and is_outlier columns if present
            cluster_df = cluster_df.drop(columns=['cluster_label', 'is_outlier'], errors='ignore')
            # Return sample of cluster members (max 20 for prompt)
            return cluster_df.head(20)
        else:
            print(f"WARNING: Cluster {cluster_id} data not found, using empty DataFrame")
            return pd.DataFrame()
    
    def step2_llm1_groq_imputation(self, missing_row, target_column, cluster_id, training_df, dataset_name):
        """
        Step 2: LLM1 - Groq API for cluster-based imputation
        ONLY receives cluster-specific data + centroid
        NO full dataset access
        """
        print(f"\n{'='*50}")
        print(f"STEP 2: LLM1 - Groq API Cluster-based Imputation")
        print(f"{'='*50}")
        print(f"Target column: {target_column}")
        print(f"Cluster ID: {cluster_id}")
        
        if cluster_id not in self.centroids:
            return None, "Cluster not found", None
        
        centroid = self.centroids[cluster_id]
        print(f"Using cluster {cluster_id} centroid")
        
        # Get cluster member data - ONLY cluster-specific data
        cluster_data = self._get_cluster_data(cluster_id, training_df, dataset_name)
        print(f"Cluster {cluster_id} has {len(cluster_data)} member records (sample)")
        
        # Create prompt with ONLY cluster data + centroid
        # IMPORTANT: LLM1 receives NO full dataset, only cluster-specific data
        prompt = f"""You are an expert data imputation specialist with access to cluster-specific data.

IMPORTANT: You ONLY have access to data from cluster {cluster_id}. You do NOT have access to the full dataset.

CONTEXT:
- You have access to data from cluster {cluster_id} ONLY
- This cluster represents similar records with common characteristics
- Cluster centroid (representative values for this cluster): {json.dumps(centroid, indent=2)}
- You do NOT have access to data from other clusters or the full dataset

CLUSTER MEMBER DATA (Sample of {len(cluster_data)} similar records from cluster {cluster_id} ONLY - this is the ONLY data available):
{cluster_data.to_string() if len(cluster_data) > 0 else "No cluster member data available"}

NOTE: This cluster data is the ONLY context available. Do not assume you have access to the full dataset.

TASK:
Predict the missing value for column '{target_column}' in this row:

{missing_row.to_string()}

ANALYSIS REQUIREMENTS:
1. Analyze the cluster centroid AND the sample cluster member data
2. Look for patterns in the cluster member records
3. Consider the missing row's existing values and how they relate to the cluster
4. Predict the most likely value for '{target_column}' based on cluster characteristics and member data
5. Provide reasoning for your prediction

RESPONSE FORMAT:
Provide your prediction in this exact format:
PREDICTION: [your predicted value]
REASONING: [brief explanation of why this value makes sense for this cluster]

Based on the cluster centroid and member data, what should be the value for '{target_column}'?
"""
        
        try:
            print("Sending request to Groq API...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data imputation specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content.strip()
                
                # Extract prediction and reasoning
                prediction = None
                reasoning = None
                
                if "PREDICTION:" in response_text:
                    prediction_line = [line for line in response_text.split('\n') if 'PREDICTION:' in line]
                    if prediction_line:
                        prediction = prediction_line[0].split('PREDICTION:')[1].strip()
                
                if "REASONING:" in response_text:
                    reasoning_line = [line for line in response_text.split('\n') if 'REASONING:' in line]
                    if reasoning_line:
                        reasoning = reasoning_line[0].split('REASONING:')[1].strip()
                
                # If no structured response, try to extract prediction from text
                if not prediction:
                    lines = response_text.split('\n')
                    for line in lines:
                        if target_column.lower() in line.lower() or 'prediction' in line.lower():
                            words = line.split()
                            for word in words:
                                if word and not word.lower() in ['prediction', 'value', 'should', 'be', 'is', 'the', 'for']:
                                    prediction = word.strip('.,!?')
                                    break
                            break
                
                if not prediction:
                    prediction = response_text.split('\n')[0].strip()
                
                print(f"SUCCESS: Groq API Response:")
                print(f"   Prediction: {prediction}")
                print(f"   Reasoning: {reasoning}")
                
                return prediction, prompt, reasoning
            else:
                print("ERROR: Empty response from Groq API")
                return None, prompt, "Empty response"
                
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            print(f"ERROR: Error calling Groq API: {error_msg}")
            return None, prompt, f"API Error: {error_msg}"
    
    def run_groq_pipeline(self, train_file, test_file, dataset_name, missing_row_idx, target_column):
        """
        Run the complete Groq LLM1 pipeline for a single missing value
        
        IMPORTANT:
        - Training data is ONLY used to load pre-computed clusters and centroids
        - For each missing cell, we compute distance from test row to each centroid (excluding target column)
        - Find nearest cluster and send ONLY that cluster's data to LLM1
        - LLM1 receives NO full dataset information, only cluster-specific data
        """
        print(f"\n{'='*70}")
        print(f"GROQ LLM1 PIPELINE FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        print("NOTE: Training data is ONLY for loading clusters. LLM1 gets ONLY cluster data.")
        
        # Load training data (ONLY for accessing cluster info - clusters already computed)
        training_df = pd.read_csv(train_file)
        print(f"Training data: {len(training_df)} rows, {len(training_df.columns)} columns")
        print("  (Used only to load pre-computed cluster centroids)")
        
        # Load pre-computed cluster information (from training data clustering)
        if not self.load_cluster_info(dataset_name):
            print("ERROR: Failed to load cluster information")
            return None
        
        # Load clustered training data (to get cluster member samples)
        clustered_df = self.load_clustered_training_data(dataset_name)
        if clustered_df is None:
            print("ERROR: Failed to load clustered training data")
            return None
        
        # Load test data (with missing values)
        test_df = pd.read_csv(test_file)
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"\nMissing row {missing_row_idx}:")
        print(missing_row.to_string())
        
        # STEP 1: Compute distance from missing row to each centroid
        # EXCLUDE target_column from distance calculation
        # Find nearest cluster
        print(f"\nComputing distances to centroids (excluding '{target_column}' column)...")
        cluster_id, distance = self.assign_to_cluster(missing_row, target_column, training_df)
        
        print(f"\n[OK] Nearest cluster: {cluster_id} (distance: {distance:.4f})")
        print(f"[OK] LLM1 will receive ONLY data from cluster {cluster_id} (not full dataset)")
        
        # Step 2: LLM1 - Cluster-based imputation
        llm1_pred, llm1_prompt, llm1_reasoning = self.step2_llm1_groq_imputation(
            missing_row, target_column, cluster_id, training_df, dataset_name
        )
        
        if llm1_pred is None:
            print("ERROR: LLM1 prediction failed")
            return None
        
        # Results
        results = {
            'missing_row_idx': int(missing_row_idx),
            'target_column': str(target_column),
            'cluster_id': str(cluster_id),
            'llm1_prediction': str(llm1_pred),
            'llm1_reasoning': str(llm1_reasoning) if llm1_reasoning else None,
            'cluster_distance': float(distance),
            'prompt': str(llm1_prompt)
        }
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Target column: {target_column}")
        print(f"LLM1 (Cluster) prediction: {llm1_pred}")
        print(f"Cluster ID: {cluster_id}")
        print(f"Reasoning: {llm1_reasoning}")
        
        return results

