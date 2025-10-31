"""
3-LLM Pipeline with Gemini API for LLM1 (Cluster-based Imputation)
Step 1: DBSCAN clustering (completed)
Step 2: LLM1 - Gemini API for cluster-based imputation
Step 3: LLM2 - RAG-based imputation (simulated)
Step 4: LLM3 - Comparison and selection (simulated)
"""

import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class GeminiLLM1Pipeline:
    def __init__(self, gemini_api_key):
        """
        Initialize the pipeline with Gemini API key
        """
        self.api_key = gemini_api_key
        self.clusters = {}
        self.centroids = {}
        self.cluster_data = {}
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        print("SUCCESS: Gemini API configured successfully")
    
    def load_cluster_info(self, dataset_name):
        """
        Load pre-computed cluster information from Step 1
        """
        cluster_file = f"clustering_results/cluster_info_{dataset_name}.json"
        
        if not os.path.exists(cluster_file):
            print(f"ERROR: Cluster file not found: {cluster_file}")
            return False
        
        with open(cluster_file, 'r') as f:
            cluster_info = json.load(f)
        
        self.clusters = cluster_info.get('clusters', {})
        self.centroids = cluster_info.get('clusters', {})  # Same as clusters in our format
        
        print(f"SUCCESS: Loaded cluster info for {dataset_name}")
        print(f"   - Clusters: {len(self.clusters)}")
        print(f"   - Dataset: {cluster_info.get('dataset', 'unknown')}")
        
        return True
    
    def assign_to_cluster(self, missing_row, training_df):
        """
        Dynamically assign missing row to most similar cluster
        Returns: cluster_id, distance
        """
        if not self.centroids:
            return "0", 0.0
        
        # Calculate distance to each cluster centroid
        best_cluster = "0"
        min_distance = float('inf')
        
        for cluster_id, centroid in self.centroids.items():
            distance = self._calculate_similarity(missing_row, centroid, training_df.columns)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        print(f"Assigned to cluster {best_cluster} (distance: {min_distance:.4f})")
        return best_cluster, min_distance
    
    def _calculate_similarity(self, row1, centroid, columns):
        """
        Calculate similarity between a row and cluster centroid
        Uses Hamming distance for categorical, Euclidean for numerical
        """
        if not isinstance(centroid, dict):
            return float('inf')
        
        distance = 0.0
        common_cols = set(row1.index) & set(centroid.keys())
        
        if not common_cols:
            return float('inf')
        
        for col in common_cols:
            val1 = row1[col]
            val2 = centroid[col]
            
            # Skip NaN values
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Numerical comparison
            try:
                num1 = float(val1)
                num2 = float(val2)
                distance += abs(num1 - num2) ** 2  # Euclidean distance squared
            except (ValueError, TypeError):
                # Categorical comparison (Hamming distance)
                if str(val1).lower() != str(val2).lower():
                    distance += 1
        
        return distance
    
    def _get_cluster_data(self, cluster_id, training_df, dataset_name="phone"):
        """
        Get sample data from the specified cluster
        """
        try:
            # Load clustered training data
            cluster_file = f"clustering_results/{dataset_name}_with_clusters.csv"
            
            if os.path.exists(cluster_file):
                df_clustered = pd.read_csv(cluster_file)
                cluster_members = df_clustered[df_clustered['cluster_label'] == int(cluster_id)]
                
                if len(cluster_members) > 0:
                    # Return sample of cluster data (max 10 records)
                    sample_size = min(10, len(cluster_members))
                    return cluster_members.sample(n=sample_size, random_state=42)
                else:
                    print(f"  WARNING: No members found for cluster {cluster_id}")
                    return pd.DataFrame()
            else:
                print(f"  WARNING: Cluster file not found: {cluster_file}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ERROR: Failed to get cluster data - {e}")
            return pd.DataFrame()
    
    def step1_dbscan_clustering(self, train_file, dataset_name):
        """
        Step 1: Apply DBSCAN to training data, find clusters and centroids
        (This is already completed, but we can load the results)
        """
        print(f"\n{'='*70}")
        print(f"STEP 1: Loading DBSCAN Clustering Results for {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load training data
        df = pd.read_csv(train_file)
        print(f"Training data: {len(df)} rows, {len(df.columns)} columns")
        
        # Load pre-computed cluster information
        if not self.load_cluster_info(dataset_name):
            print("ERROR: Failed to load cluster information")
            return None, None
        
        return df, list(self.clusters.keys())
    
    def step2_llm1_gemini_imputation(self, missing_row, target_column, cluster_id, training_df, dataset_name):
        """
        Step 2: LLM1 - Gemini API for cluster-based imputation
        Now includes full cluster data, not just centroids
        """
        print(f"\n{'='*50}")
        print(f"STEP 2: LLM1 - Gemini API Cluster-based Imputation")
        print(f"{'='*50}")
        print(f"Target column: {target_column}")
        print(f"Cluster ID: {cluster_id}")
        
        if cluster_id not in self.centroids:
            return None, "Cluster not found"
        
        centroid = self.centroids[cluster_id]
        print(f"Using cluster {cluster_id} centroid")
        
        # Get cluster member data
        cluster_data = self._get_cluster_data(cluster_id, training_df, dataset_name)
        print(f"Cluster {cluster_id} has {len(cluster_data)} member records")
        
        # Create comprehensive prompt for Gemini with full cluster data
        prompt = f"""
You are an expert data imputation specialist with access to cluster-specific data.

CONTEXT:
- You have access to data from cluster {cluster_id}
- This cluster represents similar records with common characteristics
- Cluster centroid (representative values): {json.dumps(centroid, indent=2)}

CLUSTER MEMBER DATA (Sample of {len(cluster_data)} similar records):
{cluster_data.to_string()}

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
            print("Sending request to Gemini API...")
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Parse the response (remove emojis for Windows compatibility)
                response_text = response.text.strip()
                # Remove emojis and special characters
                response_text = ''.join(char for char in response_text if ord(char) < 128)
                
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
                            # Try to extract value from the line
                            words = line.split()
                            for word in words:
                                if word and not word.lower() in ['prediction', 'value', 'should', 'be', 'is', 'the', 'for']:
                                    prediction = word.strip('.,!?')
                                    break
                            break
                
                if not prediction:
                    prediction = response_text.split('\n')[0].strip()
                
                print(f"SUCCESS: Gemini API Response:")
                print(f"   Prediction: {prediction}")
                print(f"   Reasoning: {reasoning}")
                
                return prediction, prompt, reasoning
                
            else:
                print("ERROR: Empty response from Gemini API")
                return None, prompt, "Empty response"
                
        except Exception as e:
            print(f"ERROR: Error calling Gemini API: {e}")
            return None, prompt, f"API Error: {e}"
    
    def step3_llm2_rag_imputation(self, missing_row, target_column, df_labeled):
        """
        Step 3: LLM2 - RAG-based imputation (simulated for now)
        """
        print(f"\n{'='*50}")
        print(f"STEP 3: LLM2 - RAG-based Imputation (Simulated)")
        print(f"{'='*50}")
        
        # For now, simulate LLM2 with a simple heuristic
        # In a real implementation, this would use another LLM API
        
        # Get sample data for context
        sample_data = df_labeled.head(50)
        
        prompt = f"""
You are an expert data imputation specialist with access to a comprehensive dataset.

Full dataset context ({len(df_labeled)} total records):
{sample_data.to_string()}

Task: Predict the missing value for column '{target_column}' in this row:
{missing_row.to_string()}

Based on the full dataset context, what should be the value for '{target_column}'?
Provide only the predicted value.
"""
        
        # Simple heuristic prediction (replace with actual LLM call)
        if target_column in df_labeled.columns:
            most_common = df_labeled[target_column].mode()
            prediction = most_common.iloc[0] if not most_common.empty else "Unknown"
        else:
            prediction = "Unknown"
        
        print(f"LLM2 Prediction: {prediction}")
        return prediction, prompt
    
    def step4_llm3_comparison(self, llm1_prediction, llm2_prediction, missing_row, target_column):
        """
        Step 4: LLM3 - Comparison and selection (simulated for now)
        """
        print(f"\n{'='*50}")
        print(f"STEP 4: LLM3 - Comparison and Selection (Simulated)")
        print(f"{'='*50}")
        
        prompt = f"""
You are an expert data quality specialist.

You have two predictions for the missing value in column '{target_column}':

Row context:
{missing_row.to_string()}

Prediction 1 (Cluster-based): {llm1_prediction}
Prediction 2 (RAG-based): {llm2_prediction}

Task: Compare these predictions and select the best one.
Consider:
- Consistency with row context
- Data type appropriateness
- Logical coherence

Which prediction is better? Provide only the selected value.
"""
        
        # Simple comparison logic (replace with actual LLM call)
        if llm1_prediction == llm2_prediction:
            final_prediction = llm1_prediction
            confidence = "High (Agreement)"
        else:
            # Prefer cluster-based prediction for now
            final_prediction = llm1_prediction
            confidence = "Medium (Cluster-based preferred)"
        
        print(f"Final Prediction: {final_prediction}")
        print(f"Confidence: {confidence}")
        
        return final_prediction, confidence, prompt
    
    def run_gemini_pipeline(self, train_file, test_file, dataset_name, missing_row_idx, target_column):
        """
        Run the complete pipeline with Gemini API for LLM1
        """
        print(f"\n{'='*70}")
        print(f"3-LLM PIPELINE WITH GEMINI API FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Load DBSCAN Clustering Results
        df_labeled, clusters = self.step1_dbscan_clustering(train_file, dataset_name)
        
        if df_labeled is None:
            print("ERROR: Failed to load clustering results")
            return None
        
        # Load test data
        test_df = pd.read_csv(test_file)
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"\nMissing row {missing_row_idx}:")
        print(missing_row.to_string())
        
        # DYNAMICALLY assign missing row to most similar cluster
        cluster_id, distance = self.assign_to_cluster(missing_row, df_labeled)
        print(f"Selected cluster: {cluster_id} (based on similarity distance: {distance:.4f})")
        
        # Step 2: LLM1 - Gemini API Cluster-based imputation
        llm1_pred, llm1_prompt, llm1_reasoning = self.step2_llm1_gemini_imputation(
            missing_row, target_column, cluster_id, df_labeled, dataset_name
        )
        
        # Step 3: LLM2 - RAG-based imputation (simulated)
        llm2_pred, llm2_prompt = self.step3_llm2_rag_imputation(
            missing_row, target_column, df_labeled
        )
        
        # Step 4: LLM3 - Comparison and selection (simulated)
        final_pred, confidence, llm3_prompt = self.step4_llm3_comparison(
            llm1_pred, llm2_pred, missing_row, target_column
        )
        
        # Results
        results = {
            'missing_row_idx': int(missing_row_idx),
            'target_column': str(target_column),
            'cluster_id': str(cluster_id),
            'llm1_prediction': str(llm1_pred),
            'llm1_reasoning': str(llm1_reasoning),
            'llm2_prediction': str(llm2_pred),
            'final_prediction': str(final_pred),
            'confidence': str(confidence),
            'llm1_prompt': str(llm1_prompt),
            'llm2_prompt': str(llm2_prompt),
            'llm3_prompt': str(llm3_prompt),
            'gemini_api_used': True
        }
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Target column: {target_column}")
        print(f"LLM1 (Gemini Cluster) prediction: {llm1_pred}")
        print(f"LLM1 Reasoning: {llm1_reasoning}")
        print(f"LLM2 (RAG) prediction: {llm2_pred}")
        print(f"Final prediction: {final_pred}")
        print(f"Confidence: {confidence}")
        
        return results

# Example usage
if __name__ == "__main__":
    # You need to provide your Gemini API key
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("ERROR: Please set your Gemini API key in the script")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
    else:
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        # Test on phone dataset
        results = pipeline.run_gemini_pipeline(
            train_file="train_sets/phone_train_original.csv",
            test_file="test_sets/phone_test_MNAR.csv",
            dataset_name="phone",
            missing_row_idx=0,  # First row with missing values
            target_column="brand"  # Column to impute
        )
        
        if results:
            # Save results
            with open("clustering_results/gemini_llm1_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nSUCCESS: Results saved to: clustering_results/gemini_llm1_results.json")
        else:
            print("ERROR: Pipeline failed")
