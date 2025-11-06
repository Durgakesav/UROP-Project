# CLUSTER AND CENTROID SELECTION EXPLANATION

## How Clusters Are Selected

### Current Implementation (Line 277 in gemini_llm1_pipeline.py):

```python
cluster_id = list(clusters)[0] if clusters else "0"
```

**PROBLEM**: This ALWAYS uses cluster 0 for every missing row, regardless of similarity!

### What Should Happen:

1. **Calculate Distance**: Compute distance between missing row and each cluster centroid
2. **Find Nearest Cluster**: Assign missing row to the cluster with minimum distance
3. **Use That Cluster**: Use the centroid from the assigned cluster

## How Centroids Are Calculated

### Method:
1. **For NUMERICAL columns**: Calculate **MEAN** of all values in that cluster
2. **For CATEGORICAL columns**: Calculate **MODE** (most frequent value) in that cluster

### Example from phone dataset:

**Cluster 0 Centroid** (phone dataset):
- brand: "Samsung" (most frequent brand in cluster 0)
- model: "X500" 
- approx_price_EUR: 171.75 (mean price in cluster 0)
- OS: "Android 4.4.2" (most frequent OS in cluster 0)

## What Is Sent to LLM1

### Input to LLM1:
1. **Missing row**: The row with NaN values that needs imputation
2. **Cluster centroid**: Representative values for the cluster this row belongs to
3. **Cluster ID**: The cluster number (0, 1, or 2 for phone dataset)

### LLM1 Prompt Structure:
```
You are an expert data imputation specialist with access to cluster-specific data.

CONTEXT:
- You have access to data from cluster 0
- This cluster represents similar records with common characteristics
- Cluster centroid: {centroid values shown here}

TASK:
Predict the missing value for column 'brand' in this row:
{missing row data shown here}

Based on cluster 0 characteristics, predict the value for 'brand'
```

## Current Limitations

### Issue 1: No Proper Cluster Assignment
- Missing rows are NOT assigned to clusters based on similarity
- Always uses cluster 0 regardless of row characteristics
- This reduces prediction accuracy

### Issue 2: Centroid Only (No Cluster Data)
- LLM1 gets centroid (mean/mode values) but NOT actual cluster member data
- According to 3LLM guide, LLM1 should get "cluster-specific data (similar records)"
- Currently only sending centroid summary, not individual cluster records

## Proper Implementation Needed

### Step 1: Assign Missing Row to Cluster
```python
def assign_to_cluster(missing_row, all_clusters):
    # Calculate distance to each cluster centroid
    distances = {}
    for cluster_id, centroid in all_clusters.items():
        distance = calculate_similarity(missing_row, centroid)
        distances[cluster_id] = distance
    
    # Return cluster with minimum distance
    best_cluster = min(distances, key=distances.get)
    return best_cluster
```

### Step 2: Get Cluster Member Data
```python
def get_cluster_data(cluster_id, training_data):
    # Get all rows that belong to this cluster
    cluster_rows = training_data[training_data['cluster'] == cluster_id]
    
    # Return sample of cluster data (not just centroid)
    return cluster_rows.sample(min(10, len(cluster_rows)))
```

### Step 3: Send to LLM1
- Missing row
- Cluster ID and centroid
- Sample cluster member data (similar records)
- Task: Predict based on cluster context

## Summary

### Current Approach:
1. Always use cluster 0
2. Only send centroid (mean/mode values)
3. Does NOT use actual cluster member data

### Proper Approach (According to 3LLM Guide):
1. Assign missing row to most similar cluster
2. Send centroid + sample cluster member data
3. Use cluster-specific data for prediction

### Why Current Results Are Limited:
- Accuracy: 0.8% (very low)
- Brand column: 30% accuracy (acceptable)
- No proper cluster assignment reduces effectiveness

### Recommendation:
- Implement proper cluster assignment based on similarity
- Send cluster member data to LLM1 (not just centroid)
- This should improve accuracy significantly













