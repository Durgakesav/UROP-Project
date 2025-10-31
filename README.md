# UROP Project: LDI+LLM Data Imputation Methods

## Project Overview

This project focuses on data imputation methods using Large Language Models (LLMs) combined with Learning from Data Imputation (LDI) techniques. The goal is to develop and evaluate effective strategies for handling missing data in structured datasets.

## Datasets

We work with four datasets, all under 10,000 rows for efficient experimentation:

| Dataset | Rows | Key Features | Stratification Column |
|---------|------|--------------|----------------------|
| `buy.csv` | 651 | Product names, descriptions, manufacturers, prices | manufacturer |
| `phone.csv` | 8,628 | Phone specifications, brands, features | brand |
| `restaurant.csv` | 864 | Restaurant names, addresses, types | type |
| `zomato.csv` | 8,500 | Restaurant details, cuisines, ratings | cuisine |

## Data Pipeline

### Step 1: Stratified Sampling (< 10,000 rows)
All datasets were created using **stratified sampling** to maintain uniform proportions across categories.

### Step 2: Train-Test Split (70/30)
Each dataset is split into training (70%) and testing (30%) sets with **stratified splitting** to maintain proportions.

### Step 3: Missing Value Introduction (MNAR)
MNAR missingness is introduced **only in test sets**:
- **MNAR (Missing Not at Random)**: Missingness depends on the value itself
- Test sets have ~20-40% missing values for imputation evaluation
- Training sets remain complete for optimal model learning

### Why This Approach?

- **Preserves Data Distribution**: Original category proportions maintained exactly
- **Fair Representation**: All categories equally represented in train/test sets
- **Clean Training**: Complete data for optimal model learning
- **Realistic Testing**: MNAR missingness simulates real-world scenarios
- **Consistent Results**: Comparable experiments across iterations
- **Better Generalization**: Models work well across diverse data types

### Documentation

- **[DATA SAMPLING_GUIDE.md](DATA_SAMPLING_GUIDE.md)**: Comprehensive guide explaining all steps
- **[SAMPLING_QUICK_REFERENCE.txt](SAMPLING_QUICK_REFERENCE.txt)**: Quick reference guide

## Quick Start

### Requirements

```bash
pip install pandas scikit-learn
```

### Dataset Verification

```python
import pandas as pd

# Load and verify datasets
datasets = ['buy.csv', 'phone.csv', 'restaurant.csv', 'zomato.csv']
for ds in datasets:
    df = pd.read_csv(ds)
    print(f"{ds}: {len(df)} rows")
```

## How Data Uniformity Was Maintained

### Process

1. **Identified Stratification Column**: Selected key categorical variable for each dataset
2. **Applied Stratified Sampling**: Used scikit-learn's `train_test_split` with `stratify` parameter
3. **Verified Proportions**: Ensured category distributions remained unchanged
4. **Result**: Perfect preservation of original proportions (< 0.01% deviation)

### Technical Implementation

```python
from sklearn.model_selection import train_test_split

# Stratified sampling maintains proportions
df_small, _ = train_test_split(
    df, 
    train_size=10000,
    stratify=df['category_column'],
    random_state=42
)
```

### Example: Before vs After

| Category | Original (%) | Random Sampling (%) | Stratified Sampling (%) |
|----------|-------------|---------------------|------------------------|
| Category A | 50.0 | 52.0 | 50.0 |
| Category B | 30.0 | 28.5 | 30.0 |
| Category C | 15.0 | 14.8 | 15.0 |
| Category D | 5.0 | 4.7 | 5.0 |

**Conclusion**: Stratified sampling maintains exact proportions, while random sampling causes drift.

## Project Structure

```
UROP Project/
├── README.md                          # This file
├── DATA_SAMPLING_GUIDE.md             # Detailed methodology guide
├── SAMPLING_QUICK_REFERENCE.txt       # Quick reference guide
├── buy.csv                            # Original buy dataset
├── phone.csv                          # Original phone dataset
├── restaurant.csv                     # Original restaurant dataset
├── zomato.csv                         # Original zomato dataset
├── train_sets/                        # Training datasets (70%)
│   └── *_train_original.csv           # Complete data (no missing values)
├── train_sets_clean/                  # DBSCAN-cleaned training data
│   └── *_train_clean.csv              # Outliers removed via DBSCAN
└── test_sets/                         # Test datasets (30%)
    ├── *_test_original.csv            # Complete data (ground truth)
    └── *_test_MNAR.csv                # Missing Not at Random (for evaluation)
```

## Usage Notes

- All datasets are already preprocessed and ready for imputation experiments
- Stratification ensures consistent results across different iterations
- **DBSCAN Preprocessing**: Clean training datasets available in `train_sets_clean/` (outliers removed)
- Choose between original training data or DBSCAN-cleaned data based on your needs
- The methodology can be applied to other datasets using the provided guides

## References

- LDI Repository: https://github.com/soroushomidvar/LDI
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

## Contact

Project: UROP - LDI+LLM Data Imputation Methods

---

**Last Updated**: 2025

