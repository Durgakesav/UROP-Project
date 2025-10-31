"""
3-LLM Pipeline for Data Imputation using DBSCAN Clustering
Step 1: Apply DBSCAN to training data, find clusters and centroids
Step 2: LLM1 - Cluster-based imputation using specific cluster data
Step 3: LLM2 - RAG-based imputation using full dataset
Step 4: LLM3 - Compare and select best prediction
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

class ThreeLLMPipeline:
    def __init__(self):
        self.clusters = {}
        self.centroids = {}
        self.cluster_data = {}
        self.scaler = StandardScaler()
        
    def step1_dbscan_clustering(self, train_file, dataset_name):
        """
        Step 1: Apply DBSCAN to training data, find clusters and centroids
        """
        print(f"\n{'='*70}")
        print(f"STEP 1: DBSCAN Clustering for {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load training data
        df = pd.read_csv(train_file)
        print(f"Training data: {len(df)} rows, {len(df.columns)} columns")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        if len(numeric_cols) == 0:
            print("No numeric columns - using categorical clustering")
            # Use categorical data for clustering
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_encoded = df.copy()
            for col in categorical_cols:
                df_encoded[col] = le.fit_transform(df[col].astype(str))
            X = df_encoded.values
        else:
            # Use numeric data for clustering
            df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
            X = self.scaler.fit_transform(df_numeric)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=2.0, min_samples=3)
        cluster_labels = clustering.fit_predict(X)
        
        # Store results
        df_labeled = df.copy()
        df_labeled['cluster_id'] = cluster_labels
        
        # Find clusters and centroids
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise cluster
        
        print(f"\nClustering Results:")
        print(f"  Total clusters found: {len(unique_clusters)}")
        print(f"  Noise points: {list(cluster_labels).count(-1)}")
        
        # Calculate centroids for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_df = df[cluster_mask]
            
            # Store cluster data
            self.cluster_data[cluster_id] = cluster_df
            
            # Calculate centroid (mean for numeric, mode for categorical)
            centroid = {}
            for col in df.columns:
                if col in numeric_cols:
                    centroid[col] = cluster_df[col].mean()
                else:
                    centroid[col] = cluster_df[col].mode().iloc[0] if not cluster_df[col].mode().empty else cluster_df[col].iloc[0]
            
            self.centroids[cluster_id] = centroid
            
            print(f"  Cluster {cluster_id}: {len(cluster_df)} points")
            print(f"    Centroid: {centroid}")
        
        # Save cluster information
        cluster_info = {
            'dataset': dataset_name,
            'n_clusters': len(unique_clusters),
            'n_noise': list(cluster_labels).count(-1),
            'clusters': {str(k): v for k, v in self.centroids.items()},
            'cluster_sizes': {str(k): len(self.cluster_data[k]) for k in unique_clusters}
        }
        
        with open(f"cluster_info_{dataset_name}.json", "w") as f:
            json.dump(cluster_info, f, indent=2)
        
        print(f"\nCluster information saved to: cluster_info_{dataset_name}.json")
        
        return df_labeled, unique_clusters
    
    def step2_llm1_cluster_imputation(self, missing_row, target_column, cluster_id):
        """
        Step 2: LLM1 - Cluster-based imputation
        Uses only data from the specific cluster
        """
        print(f"\n{'='*50}")
        print(f"STEP 2: LLM1 - Cluster-based Imputation")
        print(f"{'='*50}")
        print(f"Target column: {target_column}")
        print(f"Cluster ID: {cluster_id}")
        
        if cluster_id not in self.cluster_data:
            return None, "Cluster not found"
        
        cluster_df = self.cluster_data[cluster_id]
        print(f"Cluster data: {len(cluster_df)} rows")
        
        # Prepare context for LLM1
        context_data = cluster_df.head(10)  # Use first 10 rows as context
        
        # Create prompt for LLM1
        prompt = f"""
        You are an expert data imputation specialist. 
        
        Context: You have access to data from cluster {cluster_id} with {len(cluster_df)} similar records.
        
        Cluster centroid: {self.centroids[cluster_id]}
        
        Sample cluster data:
        {context_data.to_string()}
        
        Task: Predict the missing value for column '{target_column}' in this row:
        {missing_row.to_string()}
        
        Based on the cluster data, what should be the value for '{target_column}'?
        Provide only the predicted value.
        """
        
        print(f"LLM1 Prompt prepared (using cluster {cluster_id} data)")
        print(f"Context: {len(context_data)} similar records")
        
        # Simulate LLM1 response (replace with actual LLM call)
        if target_column in cluster_df.columns:
            predicted_value = cluster_df[target_column].mode().iloc[0] if not cluster_df[target_column].mode().empty else cluster_df[target_column].iloc[0]
        else:
            predicted_value = "Unknown"
        
        return predicted_value, prompt
    
    def step3_llm2_rag_imputation(self, missing_row, target_column, full_dataset):
        """
        Step 3: LLM2 - RAG-based imputation
        Uses full dataset in RAG pipeline
        """
        print(f"\n{'='*50}")
        print(f"STEP 3: LLM2 - RAG-based Imputation")
        print(f"{'='*50}")
        print(f"Target column: {target_column}")
        print(f"Full dataset: {len(full_dataset)} rows")
        
        # Prepare RAG context (sample from full dataset)
        rag_context = full_dataset.sample(min(50, len(full_dataset))).reset_index(drop=True)
        
        # Create prompt for LLM2
        prompt = f"""
        You are an expert data imputation specialist with access to a comprehensive dataset.
        
        Full dataset context ({len(full_dataset)} total records):
        {rag_context.to_string()}
        
        Task: Predict the missing value for column '{target_column}' in this row:
        {missing_row.to_string()}
        
        Based on the full dataset context, what should be the value for '{target_column}'?
        Provide only the predicted value.
        """
        
        print(f"LLM2 Prompt prepared (using full dataset)")
        print(f"RAG context: {len(rag_context)} records from {len(full_dataset)} total")
        
        # Simulate LLM2 response (replace with actual LLM call)
        if target_column in full_dataset.columns:
            predicted_value = full_dataset[target_column].mode().iloc[0] if not full_dataset[target_column].mode().empty else full_dataset[target_column].iloc[0]
        else:
            predicted_value = "Unknown"
        
        return predicted_value, prompt
    
    def step4_llm3_comparison(self, llm1_prediction, llm2_prediction, missing_row, target_column):
        """
        Step 4: LLM3 - Compare and select best prediction
        """
        print(f"\n{'='*50}")
        print(f"STEP 4: LLM3 - Comparison and Selection")
        print(f"{'='*50}")
        print(f"LLM1 prediction: {llm1_prediction}")
        print(f"LLM2 prediction: {llm2_prediction}")
        
        # Create prompt for LLM3
        prompt = f"""
        You are an expert data quality specialist.
        
        You have two predictions for the missing value in column '{target_column}':
        
        Row context: {missing_row.to_string()}
        
        Prediction 1 (Cluster-based): {llm1_prediction}
        Prediction 2 (RAG-based): {llm2_prediction}
        
        Task: Compare these predictions and select the best one.
        Consider:
        - Consistency with row context
        - Data type appropriateness
        - Logical coherence
        
        Which prediction is better? Provide only the selected value.
        """
        
        print(f"LLM3 Prompt prepared for comparison")
        
        # Simulate LLM3 response (replace with actual LLM call)
        # Simple heuristic: prefer non-null, non-"Unknown" values
        if llm1_prediction != "Unknown" and llm1_prediction is not None:
            final_prediction = llm1_prediction
            confidence = "High (Cluster-based)"
        elif llm2_prediction != "Unknown" and llm2_prediction is not None:
            final_prediction = llm2_prediction
            confidence = "Medium (RAG-based)"
        else:
            final_prediction = llm1_prediction  # Default to first
            confidence = "Low (Default)"
        
        return final_prediction, confidence, prompt
    
    def run_full_pipeline(self, train_file, test_file, dataset_name, missing_row_idx, target_column):
        """
        Run the complete 3-LLM pipeline
        """
        print(f"\n{'='*70}")
        print(f"3-LLM PIPELINE FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Step 1: DBSCAN Clustering
        df_labeled, clusters = self.step1_dbscan_clustering(train_file, dataset_name)
        
        # Load test data
        test_df = pd.read_csv(test_file)
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"\nMissing row {missing_row_idx}:")
        print(missing_row.to_string())
        
        # Determine which cluster the missing row belongs to
        # (This would be done by finding nearest centroid in practice)
        cluster_id = list(clusters)[0] if clusters else 0  # Simplified for demo
        
        # Step 2: LLM1 - Cluster-based imputation
        llm1_pred, llm1_prompt = self.step2_llm1_cluster_imputation(
            missing_row, target_column, cluster_id
        )
        
        # Step 3: LLM2 - RAG-based imputation
        llm2_pred, llm2_prompt = self.step3_llm2_rag_imputation(
            missing_row, target_column, df_labeled
        )
        
        # Step 4: LLM3 - Comparison and selection
        final_pred, confidence, llm3_prompt = self.step4_llm3_comparison(
            llm1_pred, llm2_pred, missing_row, target_column
        )
        
        # Results
        results = {
            'missing_row_idx': int(missing_row_idx),
            'target_column': str(target_column),
            'cluster_id': int(cluster_id),
            'llm1_prediction': str(llm1_pred),
            'llm2_prediction': str(llm2_pred),
            'final_prediction': str(final_pred),
            'confidence': str(confidence),
            'llm1_prompt': str(llm1_prompt),
            'llm2_prompt': str(llm2_prompt),
            'llm3_prompt': str(llm3_prompt)
        }
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Target column: {target_column}")
        print(f"LLM1 (Cluster) prediction: {llm1_pred}")
        print(f"LLM2 (RAG) prediction: {llm2_pred}")
        print(f"Final prediction: {final_pred}")
        print(f"Confidence: {confidence}")
        
        return results

# Example usage
if __name__ == "__main__":
    pipeline = ThreeLLMPipeline()
    
    # Test on phone dataset
    results = pipeline.run_full_pipeline(
        train_file="train_sets/phone_train_original.csv",
        test_file="test_sets/phone_test_MNAR.csv",
        dataset_name="phone",
        missing_row_idx=0,  # First row with missing values
        target_column="brand"  # Column to impute
    )
    
    # Save results
    with open("3llm_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: 3llm_pipeline_results.json")
