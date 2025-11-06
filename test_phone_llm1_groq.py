"""
Test Groq LLM1 on phone_test_10percent_missing.csv
Compute MSE, SMAPE, KS Statistic evaluation metrics
(Limited to first 20 missing cells to respect API limits)
"""

import pandas as pd
import numpy as np
import json
from scripts.groq_llm1_pipeline import GroqLLM1Pipeline
from scipy import stats
from datetime import datetime
import os

# Groq API Key
GROQ_API_KEY = ""

MAX_MISSING_TO_PROCESS = 20


def clean_numeric(value):
    if pd.isna(value) or value == "":
        return np.nan
    try:
        s = str(value).replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan


def calculate_mse(original_values, imputed_values):
    errors = []
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig) and pd.notna(imp):
            try:
                o = clean_numeric(orig)
                i = clean_numeric(imp)
                if not (np.isnan(o) or np.isnan(i)):
                    errors.append((o - i) ** 2)
            except Exception:
                pass
    return np.mean(errors) if errors else np.nan


def calculate_smape(original_values, imputed_values):
    errors = []
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig) and pd.notna(imp):
            try:
                o = clean_numeric(orig)
                i = clean_numeric(imp)
                if not (np.isnan(o) or np.isnan(i)) and (abs(o) + abs(i)) > 0:
                    errors.append(abs(o - i) / ((abs(o) + abs(i)) / 2))
            except Exception:
                pass
    return np.mean(errors) * 100 if errors else np.nan


def calculate_ks_statistic(original_values, imputed_values):
    orig_nums, imp_nums = [], []
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig):
            o = clean_numeric(orig)
            if not np.isnan(o):
                orig_nums.append(o)
        if pd.notna(imp):
            i = clean_numeric(imp)
            if not np.isnan(i):
                imp_nums.append(i)
    if orig_nums and imp_nums:
        ks_stat, p_value = stats.ks_2samp(orig_nums, imp_nums)
        return ks_stat, p_value
    return np.nan, np.nan


def evaluate_imputation_results(results, original_df, test_df):
    metrics = {
        "total_imputations": len(results),
        "successful_imputations": 0,
        "failed_imputations": 0,
        "accuracy": 0,
        "correct_matches": 0,
        "mse": np.nan,
        "smape": np.nan,
        "ks_statistic": np.nan,
        "ks_p_value": np.nan,
        "by_column": {},
    }
    successful_results = [r for r in results if r.get("success", False)]
    metrics["successful_imputations"] = len(successful_results)
    metrics["failed_imputations"] = len(results) - len(successful_results)

    by_column = {}
    for r in successful_results:
        col = r["target_column"]
        by_column.setdefault(col, {"original_values": [], "imputed_values": [], "matches": []})
        by_column[col]["original_values"].append(r.get("original_value"))
        by_column[col]["imputed_values"].append(r.get("imputed_value"))
        if pd.notna(r.get("original_value")) and pd.notna(r.get("imputed_value")):
            match = str(r.get("original_value")).lower().strip() == str(r.get("imputed_value")).lower().strip()
            by_column[col]["matches"].append(match)
            if match:
                metrics["correct_matches"] += 1

    if successful_results:
        metrics["accuracy"] = metrics["correct_matches"] / len(successful_results) * 100

    all_o, all_i = [], []
    for r in successful_results:
        o = r.get("original_value")
        i = r.get("imputed_value")
        if pd.notna(o) and pd.notna(i):
            all_o.append(o)
            all_i.append(i)
    if all_o and all_i:
        metrics["mse"] = calculate_mse(all_o, all_i)
        metrics["smape"] = calculate_smape(all_o, all_i)
        ks_s, ks_p = calculate_ks_statistic(all_o, all_i)
        metrics["ks_statistic"], metrics["ks_p_value"] = ks_s, ks_p

    for col, data in by_column.items():
        if data["original_values"] and data["imputed_values"]:
            col_metrics = {
                "total": len(data["original_values"]),
                "correct": sum(data["matches"]) if data["matches"] else 0,
                "accuracy": (sum(data["matches"]) / len(data["matches"]) * 100) if data["matches"] else 0,
                "mse": calculate_mse(data["original_values"], data["imputed_values"]),
                "smape": calculate_smape(data["original_values"], data["imputed_values"]),
                "ks_statistic": calculate_ks_statistic(data["original_values"], data["imputed_values"])[0],
                "ks_p_value": calculate_ks_statistic(data["original_values"], data["imputed_values"])[1],
            }
            metrics["by_column"][col] = col_metrics

    return metrics


def test_phone_llm1_groq():
    print("=" * 70)
    print("GROQ LLM1 TEST ON PHONE DATASET (10% MISSING)")
    print("=" * 70)

    pipeline = GroqLLM1Pipeline(GROQ_API_KEY)

    train_file = "train_sets/phone_train_original.csv"
    test_file = "test_sets_missing/phone_test_10percent_missing.csv"
    original_test_file = "test_sets/phone_test_original.csv"

    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        return
    if not os.path.exists(original_test_file):
        print(f"ERROR: Original test file not found: {original_test_file}")
        return

    test_df = pd.read_csv(test_file)
    original_df = pd.read_csv(original_test_file)

    if len(test_df) != len(original_df):
        print(f"WARNING: Test data ({len(test_df)} rows) and original ({len(original_df)} rows) have different lengths")
        print("Using row-by-row matching by index")

    print(f"\nTest data: {len(test_df)} rows, {len(test_df.columns)} columns")
    print(f"Original data: {len(original_df)} rows, {len(original_df.columns)} columns")

    missing_cells = []
    for idx, row in test_df.iterrows():
        for col in test_df.columns:
            if pd.isna(row[col]) or str(row[col]).strip() == "":
                original_value = original_df.iloc[idx][col] if idx < len(original_df) else None
                missing_cells.append({"row_idx": idx, "column": col, "original_value": original_value})

    print(f"\nTotal missing cells: {len(missing_cells)}")
    if len(missing_cells) > MAX_MISSING_TO_PROCESS:
        print(f"NOTE: Limiting to first {MAX_MISSING_TO_PROCESS} missing cells to respect API limits")
        missing_cells = missing_cells[:MAX_MISSING_TO_PROCESS]

    results = []
    successful_count = 0
    failed_count = 0

    for i, mc in enumerate(missing_cells):
        row_idx = mc["row_idx"]
        col_name = mc["column"]
        original_value = mc["original_value"]

        print(f"\n{'='*70}")
        print(f"Processing {i+1}/{len(missing_cells)}: Row {row_idx}, Column '{col_name}'")
        print(f"{'='*70}")

        try:
            result = pipeline.run_groq_pipeline(
                train_file=train_file,
                test_file=test_file,
                dataset_name="phone",
                missing_row_idx=row_idx,
                target_column=col_name,
            )

            if result and result.get("llm1_prediction"):
                imp = result["llm1_prediction"]
                if pd.isna(imp) or str(imp).strip().lower() in ["nan", "none", "null", ""]:
                    print("  WARNING: LLM1 predicted NaN/empty - skipping")
                    results.append({
                        "row_idx": row_idx,
                        "target_column": col_name,
                        "original_value": original_value,
                        "imputed_value": None,
                        "cluster_id": result.get("cluster_id"),
                        "cluster_distance": result.get("cluster_distance", 0),
                        "success": False,
                        "error": "Prediction was NaN or empty",
                    })
                    failed_count += 1
                    continue

                match = False
                if pd.notna(original_value) and pd.notna(imp):
                    match = str(original_value).lower().strip() == str(imp).lower().strip()

                print(f"  Original: {original_value}")
                print(f"  Imputed: {imp}")
                print(
                    f"  Cluster: {result.get('cluster_id')} (distance: {result.get('cluster_distance', 0):.4f})"
                )
                print(f"  Match: {match}")

                results.append({
                    "row_idx": row_idx,
                    "target_column": col_name,
                    "original_value": original_value,
                    "imputed_value": imp,
                    "cluster_id": result.get("cluster_id"),
                    "cluster_distance": result.get("cluster_distance", 0),
                    "success": True,
                    "match": match,
                })
                successful_count += 1
            else:
                print(f"  FAILED: {col_name}")
                results.append({
                    "row_idx": row_idx,
                    "target_column": col_name,
                    "original_value": original_value,
                    "imputed_value": None,
                    "success": False,
                    "error": "No prediction returned",
                })
                failed_count += 1
        except Exception as e:
            error_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
            print(f"  ERROR: {col_name} - {error_msg[:100]}")
            results.append({
                "row_idx": row_idx,
                "target_column": col_name,
                "original_value": original_value,
                "imputed_value": None,
                "success": False,
                "error": error_msg[:200],
            })
            failed_count += 1

    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")

    metrics = evaluate_imputation_results(results, original_df, test_df)

    print(f"\nOverall Results:")
    print(f"  Total missing cells: {metrics['total_imputations']}")
    print(f"  Successful imputations: {metrics['successful_imputations']}")
    print(f"  Failed imputations: {metrics['failed_imputations']}")
    print(
        f"  Success rate: {(metrics['successful_imputations'] / metrics['total_imputations'] * 100) if metrics['total_imputations'] else 0:.1f}%"
    )
    print(
        f"  Correct matches: {metrics['correct_matches']}/{metrics['successful_imputations']}"
    )
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")

    print(f"\nNumerical Metrics:")
    if pd.notna(metrics["mse"]):
        print(f"  MSE (Mean Squared Error): {metrics['mse']:.4f}")
    if pd.notna(metrics["smape"]):
        print(f"  SMAPE (Symmetric MAPE): {metrics['smape']:.2f}%")
    if pd.notna(metrics["ks_statistic"]):
        print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"  KS p-value: {metrics['ks_p_value']:.6f}")

    print(f"\nResults by Column:")
    for col, col_metrics in metrics["by_column"].items():
        print(f"\n  {col}:")
        print(f"    Total: {col_metrics['total']}")
        print(f"    Correct: {col_metrics['correct']}")
        print(f"    Accuracy: {col_metrics['accuracy']:.2f}%")
        if pd.notna(col_metrics["mse"]):
            print(f"    MSE: {col_metrics['mse']:.4f}")
        if pd.notna(col_metrics["smape"]):
            print(f"    SMAPE: {col_metrics['smape']:.2f}%")
        if pd.notna(col_metrics["ks_statistic"]):
            print(f"    KS Statistic: {col_metrics['ks_statistic']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "clustering_results/llm1_imputation"
    os.makedirs(out_dir, exist_ok=True)
    results_file = f"{out_dir}/phone_10percent_groq_results_{timestamp}.json"
    metrics_file = f"{out_dir}/phone_10percent_groq_metrics_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results: {results_file}")
    print(f"Metrics: {metrics_file}")

    return results, metrics


if __name__ == "__main__":
    test_phone_llm1_groq()


