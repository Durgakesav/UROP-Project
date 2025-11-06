"""
Re-cluster training data after proper 70/30 split
Uses the comprehensive clustering script
"""

import subprocess
import sys
import os

# Datasets to cluster
DATASETS = ['buy', 'phone', 'restaurant', 'zomato']

def recluster_dataset(dataset_name):
    """Run clustering on a dataset"""
    print(f"\n{'='*70}")
    print(f"CLUSTERING {dataset_name.upper()} TRAINING DATA")
    print(f"{'='*70}")
    
    train_file = f"train_sets/{dataset_name}_train_original.csv"
    
    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found")
        return False
    
    # Run the comprehensive clustering script
    # This should be the script that creates cluster_info and cluster_with_clusters files
    cmd = [
        sys.executable,
        "scripts/comprehensive_gower_dbscan_clustering.py",
        train_file,
        dataset_name
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"[OK] Clustering completed for {dataset_name}")
            print(result.stdout)
            return True
        else:
            print(f"ERROR: Clustering failed for {dataset_name}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main function to re-cluster all training data"""
    print("=" * 70)
    print("RE-CLUSTERING TRAINING DATA")
    print("=" * 70)
    print("This will create clusters and centroids from training data only")
    print("(70% of original data)")
    
    results = {}
    for dataset in DATASETS:
        success = recluster_dataset(dataset)
        results[dataset] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("CLUSTERING SUMMARY")
    print(f"{'='*70}")
    for dataset, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{dataset:<15} {status}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("1. Verify cluster files created in clustering_results/")
    print("2. Create test sets with missing values")
    print("3. Test LLM1 with Groq API")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()





