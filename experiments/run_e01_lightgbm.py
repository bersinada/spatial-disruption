"""Experiment Runner for E01 — Continuous Spatial Baseline (LightGBM).

Formal Protocol:
  - Model: LightGBM (ReachabilityClassifier + RelativeDetourRegressor)
  - Training Data: Seattle Train (20 disruption scenarios, 2,000 samples)
  - Validation Data: Seattle Val (5 disruption scenarios, 500 samples)
  - Tier 1 Test: Seattle Test (5 unseen disruption scenarios, 500 samples)
  - Tier 2 Test: Portland Transfer (10 unseen disruption scenarios on unseen network, 1,000 samples)
  - Leakage Safeguard: DEC-015 contract strictly verified; normalizers frozen on Train.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.features.tabular import extract_tabular_features
from src.models.preprocessing import (
    assert_leakage_safe,
    partition_benchmark_splits,
    FrozenPreprocessor,
)
from src.models.lightgbm_baseline import (
    ReachabilityClassifier,
    RelativeDetourRegressor,
)


def run_e01_experiment(
    benchmark_samples_path: str = "data/processed/benchmark_samples.parquet",
    scenarios_metadata_path: str = "data/processed/benchmark_scenarios_metadata.json",
    results_dir: str = "results/e01_lightgbm",
) -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING EXPERIMENT E01: CONTINUOUS SPATIAL BASELINE (LIGHTGBM)")
    print("=" * 80)
    
    start_time = time.time()
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load benchmark dataset
    print(f"\n[Step 1] Loading benchmark samples from {benchmark_samples_path}...")
    samples_p = Path(benchmark_samples_path)
    if samples_p.suffix == ".parquet":
        df = pd.read_parquet(samples_p)
    else:
        df = pd.read_json(samples_p)
    print(f"Loaded {len(df)} samples across {df['city'].nunique()} cities and {df['scenario_id'].nunique()} scenarios.")
    
    # 2. Extract tabular features
    print("\n[Step 2] Extracting tabular continuous spatial and intact routing features...")
    X, y_reach, y_detour, meta = extract_tabular_features(
        df,
        scenarios_metadata_path=scenarios_metadata_path,
    )
    print(f"Extracted {X.shape[1]} features across {len(X)} samples.")
    
    # 3. Verify DEC-015 Leakage Contract
    print("\n[Step 3] Verifying DEC-015 Feature Leakage Contract...")
    assert_leakage_safe(X)
    print("DEC-015 Contract PASSED: Zero target, post-disruption routing, or scenario identifiers in feature matrix.")
    
    # 4. Partition Splits
    print("\n[Step 4] Partitioning splits (Seattle Train, Seattle Val, Seattle Test, Portland Transfer)...")
    splits = partition_benchmark_splits(X, y_reach, y_detour, meta)
    
    for s_name, s_data in splits.items():
        n_reach = int((s_data["y_reach"] == 1).sum())
        n_unreach = int((s_data["y_reach"] == 0).sum())
        print(f"  - Split '{s_name}': {len(s_data['X'])} samples ({n_reach} reachable, {n_unreach} severed)")
        
    X_train = splits["train"]["X"]
    y_reach_train = splits["train"]["y_reach"]
    y_detour_train = splits["train"]["y_detour"]
    
    X_val = splits["val"]["X"]
    y_reach_val = splits["val"]["y_reach"]
    y_detour_val = splits["val"]["y_detour"]
    
    X_test = splits["test"]["X"]
    y_reach_test = splits["test"]["y_reach"]
    y_detour_test = splits["test"]["y_detour"]
    
    X_transfer = splits["transfer"]["X"]
    y_reach_transfer = splits["transfer"]["y_reach"]
    y_detour_transfer = splits["transfer"]["y_detour"]
    
    # Fit frozen preprocessor on Train
    preprocessor = FrozenPreprocessor(feature_names=list(X.columns))
    preprocessor.fit(X_train)
    
    # 5. Train Primary Reachability Classifier
    print("\n[Step 5] Training Primary Reachability Classifier (LightGBM)...")
    clf = ReachabilityClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
    )
    clf.fit(X_train, y_reach_train, X_val=X_val, y_val=y_reach_val, early_stopping_rounds=30)
    
    # Evaluate Classifier across splits
    clf_metrics = {
        "train": clf.evaluate(X_train, y_reach_train),
        "val": clf.evaluate(X_val, y_reach_val),
        "test_in_city": clf.evaluate(X_test, y_reach_test),
        "transfer_cross_city": clf.evaluate(X_transfer, y_reach_transfer),
    }
    
    # 6. Train Secondary Relative Detour Regressor
    print("\n[Step 6] Training Secondary Relative Detour Regressor (LightGBM Huber)...")
    reg = RelativeDetourRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=25,
        random_state=42,
    )
    reg.fit(X_train, y_detour_train, X_val=X_val, y_val=y_detour_val, early_stopping_rounds=30)
    
    # Evaluate Regressor across splits
    reg_metrics = {
        "train": reg.evaluate(X_train, y_detour_train),
        "val": reg.evaluate(X_val, y_detour_val),
        "test_in_city": reg.evaluate(X_test, y_detour_test),
        "transfer_cross_city": reg.evaluate(X_transfer, y_detour_transfer),
    }
    
    # 7. Feature Importances
    imp_df = clf.get_feature_importances()
    imp_path = res_path / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"\nTop 10 Most Predictive Spatial Features for Reachability:")
    for idx, row in imp_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<30} (Gain: {row['gain_importance']:10.1f}, Splits: {row['split_importance']})")
        
    # 8. Print Formatted Benchmark Tables
    print("\n" + "=" * 80)
    print("E01 EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n--- PRIMARY TASK: Post-Disruption Reachability (Binary Classification) ---")
    clf_df = pd.DataFrame([
        {
            "Split": "Train (Seattle In-Sample)",
            "ROC-AUC": clf_metrics["train"]["roc_auc"],
            "PR-AUC": clf_metrics["train"]["pr_auc"],
            "Accuracy": clf_metrics["train"]["accuracy"],
            "Balanced Acc": clf_metrics["train"]["balanced_accuracy"],
            "F1": clf_metrics["train"]["f1"],
            "Brier Score": clf_metrics["train"]["brier_score"],
        },
        {
            "Split": "Val (Seattle Selection)",
            "ROC-AUC": clf_metrics["val"]["roc_auc"],
            "PR-AUC": clf_metrics["val"]["pr_auc"],
            "Accuracy": clf_metrics["val"]["accuracy"],
            "Balanced Acc": clf_metrics["val"]["balanced_accuracy"],
            "F1": clf_metrics["val"]["f1"],
            "Brier Score": clf_metrics["val"]["brier_score"],
        },
        {
            "Split": "Test (Seattle Tier 1 In-City)",
            "ROC-AUC": clf_metrics["test_in_city"]["roc_auc"],
            "PR-AUC": clf_metrics["test_in_city"]["pr_auc"],
            "Accuracy": clf_metrics["test_in_city"]["accuracy"],
            "Balanced Acc": clf_metrics["test_in_city"]["balanced_accuracy"],
            "F1": clf_metrics["test_in_city"]["f1"],
            "Brier Score": clf_metrics["test_in_city"]["brier_score"],
        },
        {
            "Split": "Transfer (Portland Tier 2 Zero-Shot)",
            "ROC-AUC": clf_metrics["transfer_cross_city"]["roc_auc"],
            "PR-AUC": clf_metrics["transfer_cross_city"]["pr_auc"],
            "Accuracy": clf_metrics["transfer_cross_city"]["accuracy"],
            "Balanced Acc": clf_metrics["transfer_cross_city"]["balanced_accuracy"],
            "F1": clf_metrics["transfer_cross_city"]["f1"],
            "Brier Score": clf_metrics["transfer_cross_city"]["brier_score"],
        },
    ])
    print(clf_df.to_string(index=False))
    
    print("\n--- SECONDARY TASK: Conditional Relative Detour (Regression on Reachable Pairs) ---")
    reg_df = pd.DataFrame([
        {
            "Split": "Train (Seattle In-Sample)",
            "MAE": reg_metrics["train"]["mae"],
            "Median AE": reg_metrics["train"]["median_ae"],
            "RMSE": reg_metrics["train"]["rmse"],
            "Pearson r": reg_metrics["train"]["pearson_r"],
            "Spearman rho": reg_metrics["train"]["spearman_rho"],
        },
        {
            "Split": "Val (Seattle Selection)",
            "MAE": reg_metrics["val"]["mae"],
            "Median AE": reg_metrics["val"]["median_ae"],
            "RMSE": reg_metrics["val"]["rmse"],
            "Pearson r": reg_metrics["val"]["pearson_r"],
            "Spearman rho": reg_metrics["val"]["spearman_rho"],
        },
        {
            "Split": "Test (Seattle Tier 1 In-City)",
            "MAE": reg_metrics["test_in_city"]["mae"],
            "Median AE": reg_metrics["test_in_city"]["median_ae"],
            "RMSE": reg_metrics["test_in_city"]["rmse"],
            "Pearson r": reg_metrics["test_in_city"]["pearson_r"],
            "Spearman rho": reg_metrics["test_in_city"]["spearman_rho"],
        },
        {
            "Split": "Transfer (Portland Tier 2 Zero-Shot)",
            "MAE": reg_metrics["transfer_cross_city"]["mae"],
            "Median AE": reg_metrics["transfer_cross_city"]["median_ae"],
            "RMSE": reg_metrics["transfer_cross_city"]["rmse"],
            "Pearson r": reg_metrics["transfer_cross_city"]["pearson_r"],
            "Spearman rho": reg_metrics["transfer_cross_city"]["spearman_rho"],
        },
    ])
    print(reg_df.to_string(index=False))
    
    # 9. Save full metrics payload
    total_duration = round(time.time() - start_time, 2)
    results_payload = {
        "experiment_id": "E01_continuous_spatial_baseline",
        "model": "LightGBM",
        "timestamp_unix": int(time.time()),
        "runtime_seconds": total_duration,
        "features": {
            "count": int(X.shape[1]),
            "names": list(X.columns),
            "top_10_by_gain": imp_df.head(10).to_dict(orient="records"),
        },
        "reachability_classification": clf_metrics,
        "detour_regression": reg_metrics,
    }
    
    metrics_path = res_path / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(results_payload, fp, indent=2)
    print(f"\nSaved structured metrics to {metrics_path}")
    
    # 10. Generate Markdown Report
    summary_md_path = res_path / "e01_summary.md"
    with open(summary_md_path, "w") as fp:
        fp.write("# E01 Continuous Spatial Baseline — Results\n\n")
        fp.write(f"- **Model**: LightGBM GBDT (Classifier + Huber Regressor)\n")
        fp.write(f"- **Runtime**: {total_duration} seconds\n")
        fp.write(f"- **Features**: {X.shape[1]} continuous spatial, intact routing, hazard, and density attributes\n")
        fp.write(f"- **DEC-015 Leakage Contract**: 100% verified; zero target or post-disruption leakage\n\n")
        
        fp.write("## Primary Task: Post-Disruption Reachability (Binary Classification)\n\n")
        fp.write(clf_df.to_markdown(index=False) + "\n\n")
        
        fp.write("## Secondary Task: Relative Detour (Conditional Regression on Reachable OD Pairs)\n\n")
        fp.write(reg_df.to_markdown(index=False) + "\n\n")
        
        fp.write("## Top 10 Features by Predictive Gain\n\n")
        fp.write(imp_df.head(10).to_markdown(index=False) + "\n\n")
        
        fp.write("## Key Scientific Takeaways\n\n")
        fp.write(f"1. **In-City Disruption Generalization (Tier 1)**: LightGBM achieves ROC-AUC {clf_metrics['test_in_city']['roc_auc']:.4f} and Balanced Accuracy {clf_metrics['test_in_city']['balanced_accuracy']:.4f} on unseen disruption events within Seattle.\n")
        fp.write(f"2. **Zero-Shot Cross-City Transfer (Tier 2)**: On Portland, reachability ROC-AUC is {clf_metrics['transfer_cross_city']['roc_auc']:.4f} (PR-AUC: {clf_metrics['transfer_cross_city']['pr_auc']:.4f}), establishing the strong empirical non-graph baseline benchmark.\n")
        fp.write("3. **Graph Baseline Target for E02/E03**: For GraphSAGE (E02) and CompGCN (E03) to demonstrate topological reasoning superiority, they must outperform these tabular spatial generalization frontiers under Tier 1 and Tier 2 transfer.\n")
        
    print(f"Saved executive markdown report to {summary_md_path}")
    print("=" * 80)
    
    return results_payload


if __name__ == "__main__":
    run_e01_experiment()
