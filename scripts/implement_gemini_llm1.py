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
        self.model = genai.GenerativeModel('gemini-pro')
        
        print("✅ Gemini API configured successfully")
    
    def load_cluster_info(self, dataset_name):
        """
        Load pre-computed cluster information from Step 1
        """
        cluster_file = f"step1_dbscan_complete_results.json"
        
        if not os.path.exists(cluster_file):
            print(f"❌ Cluster file not found: {cluster_file}")
            return False
        
        with open(cluster_file, 'r') as f:
            all_results = json.load(f)
        
        if dataset_name not in all_results:
            print(f"❌ Dataset {dataset_name} not found in cluster results")
            return False
        
        dataset_results = all_results[dataset_name]
        
        # Extract cluster information
        self.clusters = dataset_results['clusters']
        self.centroids = dataset_results['centroids']
        
        print(f"✅ Loaded cluster info for {dataset_name}")
        print(f"   Clusters: {len(self.clusters)}")
        print(f"   Quality score: {dataset_results['dbscan_config']['quality_score']}")
        
        return True
    
    def assign_missing_row_to_cluster(self, missing_row, target_column):
        """
        Assign missing row to the most appropriate cluster
        """
        if not self.centroids:
            return 0  # Default to cluster 0
        
        # Simple assignment: find cluster with most similar non-missing values
        best_cluster = 0
        best_similarity = 0
        
        for cluster_id, centroid in self.centroids.items():
            similarity = 0
            matches = 0
            
            for col, value in missing_row.items():
                if pd.notna(value) and col in centroid:
                    if str(value).lower() == str(centroid[col]).lower():
                        similarity += 1
                    matches += 1
            
            if matches > 0:
                similarity_score = similarity / matches
                if similarity_score > best_similarity:
                    best_similarity = similarity_score
                    best_cluster = int(cluster_id)
        
        print(f"   Assigned to cluster {best_cluster} (similarity: {best_similarity:.2f})")
        return best_cluster
    
    def step2_gemini_llm1_cluster_imputation(self, missing_row, target_column, cluster_id):
        """
        Step 2: LLM1 - Gemini API for cluster-based imputation
        """
        print(f"\n{'='*60}")
        print(f"STEP 2: GEMINI LLM1 - Cluster-based Imputation")
        print(f"{'='*60}")
        print(f"Target column: {target_column}")
        print(f"Cluster ID: {cluster_id}")
        
        if str(cluster_id) not in self.clusters:
            print(f"❌ Cluster {cluster_id} not found")
            return None, "Cluster not found"
        
        cluster_info = self.clusters[str(cluster_id)]
        centroid = self.centroids[str(cluster_id)]
        
        print(f"Cluster size: {cluster_info['size']} points")
        print(f"Cluster percentage: {cluster_info['percentage']:.1f}%")
        
        # Prepare context for Gemini
        context_data = cluster_info['sample_data'][:5]  # Use first 5 samples
        
        # Create comprehensive prompt for Gemini
        prompt = f"""
You are an expert data imputation specialist with access to a specific cluster of similar data records.

CLUSTER CONTEXT:
- Cluster ID: {cluster_id}
- Cluster size: {cluster_info['size']} similar records
- Cluster centroid (representative values): {json.dumps(centroid, indent=2)}

SAMPLE CLUSTER DATA:
{json.dumps(context_data, indent=2)}

MISSING ROW TO IMPUTE:
{missing_row.to_dict()}

TASK:
Predict the missing value for column '{target_column}' in the above row.

INSTRUCTIONS:
1. Analyze the cluster data and centroid to understand the pattern
2. Consider the context of the missing row
3. Make a prediction that is consistent with the cluster characteristics
4. Provide only the predicted value (no explanation needed)

PREDICTED VALUE:
"""
        
        try:
            print("🤖 Sending request to Gemini API...")
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            prediction = response.text.strip()
            
            print(f"✅ Gemini prediction: {prediction}")
            
            return prediction, prompt
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return None, f"API Error: {e}"
    
    def step3_llm2_rag_imputation(self, missing_row, target_column, full_dataset):
        """
        Step 3: LLM2 - RAG-based imputation (simulated for now)
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: LLM2 - RAG-based Imputation (Simulated)")
        print(f"{'='*60}")
        print(f"Target column: {target_column}")
        print(f"Full dataset: {len(full_dataset)} rows")
        
        # Simulate RAG-based prediction
        if target_column in full_dataset.columns:
            prediction = full_dataset[target_column].mode().iloc[0] if not full_dataset[target_column].mode().empty else "Unknown"
        else:
            prediction = "Unknown"
        
        print(f"LLM2 prediction: {prediction}")
        return prediction
    
    def step4_llm3_comparison(self, llm1_prediction, llm2_prediction, missing_row, target_column):
        """
        Step 4: LLM3 - Comparison and selection (simulated for now)
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: LLM3 - Comparison and Selection (Simulated)")
        print(f"{'='*60}")
        print(f"LLM1 (Gemini) prediction: {llm1_prediction}")
        print(f"LLM2 (RAG) prediction: {llm2_prediction}")
        
        # Simple comparison logic
        if llm1_prediction and llm1_prediction != "Unknown":
            final_prediction = llm1_prediction
            confidence = "High (Gemini cluster-based)"
        elif llm2_prediction and llm2_prediction != "Unknown":
            final_prediction = llm2_prediction
            confidence = "Medium (RAG-based)"
        else:
            final_prediction = llm1_prediction or llm2_prediction
            confidence = "Low (Default)"
        
        print(f"Final prediction: {final_prediction}")
        print(f"Confidence: {confidence}")
        
        return final_prediction, confidence
    
    def run_gemini_pipeline(self, dataset_name, test_file, missing_row_idx, target_column):
        """
        Run the complete pipeline with Gemini LLM1
        """
        print(f"\n{'='*80}")
        print(f"GEMINI 3-LLM PIPELINE FOR {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Step 1: Load cluster information
        if not self.load_cluster_info(dataset_name):
            return None
        
        # Load test data
        test_df = pd.read_csv(test_file)
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"\nMissing row {missing_row_idx}:")
        print(missing_row.to_string())
        
        # Assign missing row to cluster
        cluster_id = self.assign_missing_row_to_cluster(missing_row, target_column)
        
        # Step 2: Gemini LLM1 - Cluster-based imputation
        llm1_pred, llm1_prompt = self.step2_gemini_llm1_cluster_imputation(
            missing_row, target_column, cluster_id
        )
        
        if llm1_pred is None:
            print("❌ Gemini LLM1 failed")
            return None
        
        # Step 3: LLM2 - RAG-based imputation
        llm2_pred = self.step3_llm2_rag_imputation(
            missing_row, target_column, test_df
        )
        
        # Step 4: LLM3 - Comparison and selection
        final_pred, confidence = self.step4_llm3_comparison(
            llm1_pred, llm2_pred, missing_row, target_column
        )
        
        # Results
        results = {
            'dataset': dataset_name,
            'missing_row_idx': int(missing_row_idx),
            'target_column': str(target_column),
            'cluster_id': int(cluster_id),
            'gemini_prediction': str(llm1_pred),
            'rag_prediction': str(llm2_pred),
            'final_prediction': str(final_pred),
            'confidence': str(confidence),
            'gemini_prompt': str(llm1_prompt)
        }
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Target column: {target_column}")
        print(f"Gemini (LLM1) prediction: {llm1_pred}")
        print(f"RAG (LLM2) prediction: {llm2_pred}")
        print(f"Final prediction: {final_pred}")
        print(f"Confidence: {confidence}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Get Gemini API key from environment or user input
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("Please set your Gemini API key:")
        print("Option 1: Set environment variable: GEMINI_API_KEY=your_key_here")
        print("Option 2: Enter it when prompted")
        api_key = input("Enter Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        exit(1)
    
    # Initialize pipeline
    pipeline = GeminiLLM1Pipeline(api_key)
    
    # Test on phone dataset (best clustering performance)
    print("Testing on phone dataset (best clustering performance)...")
    
    results = pipeline.run_gemini_pipeline(
        dataset_name="phone",
        test_file="test_sets/phone_test_MNAR.csv",
        missing_row_idx=0,  # First row with missing values
        target_column="brand"  # Column to impute
    )
    
    if results:
        # Save results
        with open("gemini_llm1_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: gemini_llm1_results.json")
    else:
        print("❌ Pipeline failed")



