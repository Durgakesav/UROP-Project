# 3-LLM Pipeline for Data Imputation using DBSCAN Clustering

## Overview

This pipeline implements a sophisticated 3-LLM approach for data imputation:

1. **LLM1**: Cluster-based imputation using specific cluster data
2. **LLM2**: RAG-based imputation using full dataset
3. **LLM3**: Comparison and selection of best prediction

## Pipeline Architecture

```
Training Data → DBSCAN → Clusters + Centroids
                    ↓
Missing Value → LLM1 (Cluster) → Prediction 1
                    ↓
Missing Value → LLM2 (RAG) → Prediction 2
                    ↓
Predictions → LLM3 (Compare) → Final Prediction
```

## Step-by-Step Implementation

### Step 1: DBSCAN Clustering on Training Data

**Purpose**: Find clusters and centroids in training data

**Process**:
1. Load training dataset
2. Apply DBSCAN clustering
3. Calculate centroids for each cluster
4. Store cluster data for LLM1

**Output**:
- Cluster assignments for each training record
- Centroids (mean/mode for each cluster)
- Cluster-specific datasets

### Step 2: LLM1 - Cluster-based Imputation

**Purpose**: Predict missing values using only relevant cluster data

**Process**:
1. Identify which cluster the missing row belongs to
2. Retrieve only data from that specific cluster
3. Send cluster data + missing row to LLM1
4. LLM1 predicts based on cluster context

**Input to LLM1**:
- Missing row with context
- Cluster-specific data (similar records)
- Cluster centroid information

**Output**: Prediction 1

### Step 3: LLM2 - RAG-based Imputation

**Purpose**: Predict missing values using full dataset in RAG pipeline

**Process**:
1. Retrieve relevant context from full dataset
2. Send full dataset context + missing row to LLM2
3. LLM2 predicts based on comprehensive context

**Input to LLM2**:
- Missing row with context
- Full dataset sample (RAG context)
- Complete data distribution

**Output**: Prediction 2

### Step 4: LLM3 - Comparison and Selection

**Purpose**: Compare both predictions and select the best one

**Process**:
1. Receive both predictions from LLM1 and LLM2
2. Send comparison task to LLM3
3. LLM3 evaluates and selects best prediction

**Input to LLM3**:
- Prediction 1 (Cluster-based)
- Prediction 2 (RAG-based)
- Original missing row context

**Output**: Final prediction with confidence

## Implementation Details

### DBSCAN Parameters
- `eps=2.0`: Maximum distance between points in same cluster
- `min_samples=3`: Minimum points to form a cluster
- Handles both numeric and categorical data

### LLM Prompts Structure

#### LLM1 Prompt:
```
Context: Cluster {cluster_id} with {n} similar records
Cluster centroid: {centroid_info}
Sample cluster data: {cluster_sample}
Task: Predict missing value for column '{target_column}'
```

#### LLM2 Prompt:
```
Context: Full dataset ({n} total records)
RAG context: {sample_data}
Task: Predict missing value for column '{target_column}'
```

#### LLM3 Prompt:
```
Prediction 1 (Cluster): {llm1_result}
Prediction 2 (RAG): {llm2_result}
Task: Compare and select best prediction
```

## Usage Example

```python
from implement_3llm_pipeline import ThreeLLMPipeline

# Initialize pipeline
pipeline = ThreeLLMPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    train_file="train_sets/phone_train_original.csv",
    test_file="test_sets/phone_test_MNAR.csv", 
    dataset_name="phone",
    missing_row_idx=0,
    target_column="brand"
)

# Results contain:
# - LLM1 prediction (cluster-based)
# - LLM2 prediction (RAG-based) 
# - Final prediction (LLM3 selection)
# - Confidence level
# - All prompts for debugging
```

## Benefits of This Approach

### 1. **Specialized Context**
- LLM1 gets focused, relevant cluster data
- LLM2 gets comprehensive dataset context
- Each LLM has optimal information for its task

### 2. **Quality Assurance**
- LLM3 acts as quality control
- Compares multiple approaches
- Selects best prediction

### 3. **Robustness**
- Handles different types of missing patterns
- Works with both numeric and categorical data
- Adapts to dataset characteristics

### 4. **Explainability**
- Clear reasoning from each LLM
- Confidence levels for predictions
- Audit trail of decision process

## File Structure

```
3LLM_Pipeline/
├── implement_3llm_pipeline.py    # Main pipeline implementation
├── 3LLM_PIPELINE_GUIDE.md        # This guide
├── cluster_info_*.json          # DBSCAN cluster information
└── 3llm_pipeline_results.json   # Pipeline results
```

## Next Steps

1. **Replace simulated LLM calls** with actual LLM API calls
2. **Implement cluster assignment** for missing rows
3. **Add confidence scoring** mechanisms
4. **Test on all datasets** (buy, phone, restaurant, zomato)
5. **Compare with baseline** imputation methods

## Integration with Existing Workflow

This pipeline integrates with your existing data preparation:

- **Training data**: Use `train_sets/*_train_original.csv` or `train_sets_clean/*_train_clean.csv`
- **Test data**: Use `test_sets/*_test_MNAR.csv` for missing value imputation
- **Ground truth**: Use `test_sets/*_test_original.csv` for evaluation

The 3-LLM pipeline provides a sophisticated approach to data imputation that leverages both local cluster context and global dataset knowledge for optimal results.



