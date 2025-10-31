# Data Sampling Guide for LDI+LLM Data Imputation Project

## Purpose
This document explains the methodology used to:
1. Create smaller datasets (< 10,000 rows) from large datasets while maintaining uniform proportions
2. Split data into train-test sets (70/30) for evaluation
3. Introduce realistic missing value patterns (MNAR - Missing Not at Random) for imputation experiments

This is critical for ensuring data imputation experiments produce reliable and generalizable results.

---

## Problem Statement

When working with large datasets for data imputation experiments, we need to:
1. Reduce dataset size to manageable levels (< 10,000 rows)
2. **Maintain the original distribution/proportions** of categorical variables
3. Preserve dataset integrity for LDI+LLM experimentation

Simply taking random samples can distort the original data distribution, leading to biased results.

---

## Solution: Stratified Sampling

### What is Stratified Sampling?

**Stratified sampling** is a statistical method that divides a population into homogeneous subgroups (strata) and then randomly samples from each subgroup proportionally. This ensures each category maintains its representation in the smaller dataset.

### Why Does This Matter?

For data imputation tasks, maintaining uniform proportions ensures:
- **Representative training data**: Models see a balanced view of all categories
- **Fair evaluation**: Test results reflect real-world distributions
- **Generalizability**: Models perform well across all data types, not just over-represented ones
- **Consistent benchmarks**: Different iterations produce comparable results

---

## Methodology Used

### Process Overview

1. **Identify Stratification Column**: Select a categorical variable that represents the key dimension to preserve
2. **Calculate Proportions**: Determine the distribution of each category in the original dataset
3. **Sample Proportionally**: Use scikit-learn's `train_test_split` with `stratify` parameter
4. **Verify Results**: Compare proportions before and after sampling

### Technical Implementation

```python
from sklearn.model_selection import train_test_split

# Stratified sampling maintains proportions
df_small, _ = train_test_split(
    df, 
    train_size=max_rows,      # Target size (e.g., 10,000 rows)
    stratify=df[stratify_col], # Column to preserve proportions
    random_state=42            # Reproducibility
)
```

### Key Differences: Random vs Stratified Sampling

| Aspect | Random Sampling | Stratified Sampling |
|--------|----------------|---------------------|
| **Selection Method** | Completely random | Proportional to each category |
| **Proportions** | May drift from original | Maintains original exactly |
| **Use Case** | When proportions don't matter | When balance is critical |
| **Example Deviation** | Category A: 50% → 52% (drift) | Category A: 50% → 50% (preserved) |

---

## Datasets and Stratification Strategy

### 1. Buy Dataset
- **Original Size**: 651 rows
- **Stratification Column**: `manufacturer`
- **Reason**: Preserve brand diversity for product imputation
- **Result**: All manufacturers represented proportionally

### 2. Phone Dataset
- **Original Size**: 8,628 rows
- **Stratification Column**: `brand`
- **Reason**: Maintain phone brand distribution for realistic imputation
- **Result**: Brands like Samsung, Apple, Xiaomi maintain their market share

### 3. Restaurant Dataset
- **Original Size**: 864 rows
- **Stratification Column**: `type`
- **Reason**: Preserve cuisine type diversity (American, Italian, French, etc.)
- **Result**: Each restaurant type maintains its proportion

### 4. Zomato Dataset
- **Original Size**: 8,500 rows
- **Stratification Column**: `cuisine`
- **Reason**: Keep cuisine diversity for restaurant data imputation
- **Result**: All cuisines (Indian, Chinese, Italian, etc.) represented fairly

---

## Verification: Ensuring Uniformity

### Proportion Verification

After sampling, we verify proportions:

```python
# Check original proportions
original_props = df[stratify_col].value_counts(normalize=True)

# Check sampled proportions
sampled_props = df_small[stratify_col].value_counts(normalize=True)

# Compare deviations
for category in original_props.index:
    deviation = sampled_props[category] - original_props[category]
    print(f"{category}: {deviation:.4f} ({deviation*100:+.2f}%)")
```

### Expected Results

With stratified sampling:
- **Deviation**: < 0.01% (essentially zero)
- **Perfect preservation**: Original proportions maintained exactly

Without stratified sampling:
- **Deviation**: ±0.1% to ±2% (proportional drift)
- **Risk**: Under-represented categories may disappear entirely

---

## Why This Matters for LDI+LLM Experiments

### Data Imputation Context

In data imputation tasks:
1. **Missing values** need to be filled based on existing patterns
2. **Imbalanced data** can lead to:
   - Overfitting to majority categories
   - Poor performance on minority categories
   - Biased imputation predictions

### Benefits of Uniform Sampling

✓ **Consistent Training**: Model sees all categories equally  
✓ **Fair Evaluation**: Test on representative data distribution  
✓ **Better Generalization**: Works well across diverse data types  
✓ **Reproducible Results**: Same proportions = comparable experiments  
✓ **Real-world Relevance**: Maintains actual data characteristics  

---

## Best Practices

### When to Use Stratified Sampling

✅ **DO use stratified sampling when:**
- Dataset has categorical variables
- Category balance is important
- You need representative samples
- Building models for production use

❌ **DON'T use stratified sampling when:**
- Dataset is already balanced
- No categorical variables exist
- Proportions don't matter for your task
- Dataset is too small (< 1,000 rows)

### Selection Criteria for Stratification Column

Choose the column that:
1. **Represents key diversity** in your data
2. **Has moderate cardinality** (not too many unique values)
3. **Is most relevant** to your imputation task
4. **Has adequate samples** in each category

---

## Technical Details

### Library Requirements

```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Key Parameters

- `train_size`: Target number of rows (e.g., 10,000)
- `stratify`: Column to maintain proportions
- `random_state`: Seed for reproducibility (42)

### Edge Cases Handled

1. **Dataset already small**: If < 10,000 rows, copy as-is
2. **Sparse categories**: If category has too few samples, fall back to random sampling
3. **Missing values**: Handle NA values in stratification column

---

## Step 2: Train-Test Split and Missing Value Introduction

### Train-Test Split (70/30)

After creating smaller datasets, we split them into training and testing sets:

- **Training Set**: 70% of data - Used for model training
- **Test Set**: 30% of data - Used for evaluation

#### Methodology

```python
from sklearn.model_selection import train_test_split

# Stratified split maintains proportions
train_df, test_df = train_test_split(
    df,
    test_size=0.3,           # 30% for testing
    stratify=df[stratify_col], # Maintain category proportions
    random_state=42           # Reproducibility
)
```

#### Why Stratified Split?

- Maintains category proportions in both train and test sets
- Ensures fair evaluation across all data types
- Produces consistent results across experiments

### Missing Value Introduction: MNAR (Missing Not at Random)

Real-world data often has missing values. We introduce **MNAR** missing patterns for realistic imputation experiments.

#### MNAR (Missing Not at Random)

**Definition**: Missingness depends on the **value itself** or unobserved data.

**Characteristics**:
- Missingness is related to the actual missing value
- Missing data mechanism is **not** ignorable
- Requires special handling and assumptions

**Implementation Strategy**:
- **For numerical columns**: Extreme values (top/bottom 10%) more likely to be missing
- **For categorical columns**: Rare categories more likely to be missing
- Missingness probability: ~15% base rate, adjusted based on value extremity

**Example**:
```python
# Extreme values more likely to be missing
lower_threshold = values.quantile(0.1)  # Bottom 10%
upper_threshold = values.quantile(0.9)  # Top 10%

if value <= lower_threshold or value >= upper_threshold:
    missing_prob = 0.15 * 1.5  # 1.5x more likely
else:
    missing_prob = 0.15 * 0.5  # 0.5x less likely
```

### Why MNAR?

MNAR (Missing Not at Random) is chosen because:
- **Realistic**: Common in real-world scenarios (e.g., surveys, medical data)
- **Challenging**: Requires sophisticated imputation methods
- **Evaluates Robustness**: Tests how well LDI+LLM handles non-ignorable missingness
- **LLM Performance**: Requires inferring missingness mechanism from context

### Dataset Organization

Each dataset now has **3 versions**:

#### Training Sets (train_sets/)
- `*_train_original.csv` - Complete data, no missing values (used for training)

#### Test Sets (test_sets/)
- `*_test_original.csv` - Complete data, no missing values (ground truth for evaluation)
- `*_test_MNAR.csv` - Missing Not at Random (~20% missing) (for imputation evaluation)

### Missing Value Statistics

MNAR missingness is applied only to test sets:

#### Buy Dataset
- **MNAR Test**: ~41% missing
- Most affected columns: `price`, `description`, `name`

#### Phone Dataset
- **MNAR Test**: ~39% missing
- Most affected columns: `NFC`, `4G_bands`, `GPU`, `Chipset`, `model`

#### Restaurant Dataset
- **MNAR Test**: ~30% missing
- Most affected columns: `name`, `addr`, `phone`

#### Zomato Dataset
- **MNAR Test**: ~17% missing
- Most affected columns: `restaurant_name`, `address`

### Key Benefits

✓ **Clean Training**: No missing values in training data for optimal model learning  
✓ **Realistic Testing**: MNAR missingness on test set simulates real-world scenarios  
✓ **Fair Evaluation**: Complete ground truth available for comparison  
✓ **Method Robustness**: Tests how well LDI+LLM handles non-ignorable missingness  
✓ **Reproducible Experiments**: Fixed random seeds ensure consistency  

---

## Conclusion

By using **stratified sampling** and **MNAR missing value patterns**, we ensure that our datasets:
- Maintain uniform proportions across categories (Step 1)
- Preserve original data characteristics in train-test splits (Step 2)
- Provide complete training data for optimal model learning (Step 3)
- Include realistic MNAR missing patterns only in test sets (Step 3)
- Produce reliable and generalizable imputation results
- Support fair and consistent LDI+LLM experiments

This comprehensive approach is essential for maintaining data integrity while working with manageable dataset sizes and realistic missingness scenarios for experimentation.

---

## References

- [Scikit-learn Documentation: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Stratified Sampling in Statistics](https://en.wikipedia.org/wiki/Stratified_sampling)
- LDI Repository: https://github.com/soroushomidvar/LDI

---

**Document Version**: 2.0  
**Last Updated**: 2025  
**Project**: UROP - LDI+LLM Data Imputation Methods

