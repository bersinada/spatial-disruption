"""Leakage-Safe Preprocessing and Split Isolation (DEC-015 Enforcement).

Ensures:
  1. Strict enforcement of forbidden columns per DEC-015.
  2. Normalizers / scalers are fitted exclusively on Seattle Train data and frozen.
  3. Clean partition across Train, Validation, In-City Test, and Zero-Shot Transfer splits.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FORBIDDEN_INPUT_COLUMNS = {
    "reachable",
    "relative_detour",
    "disrupted_distance_m",
    "disrupt_path_hops",
    "sample_type",
    "split",
    "scenario_id",
    "seed",
    "disrupted_edge_ids",
}


def assert_leakage_safe(X: pd.DataFrame) -> None:
    """Validate that no target, post-disruption ground truth, or metadata leaks into X."""
    cols = set(X.columns)
    leaked = cols.intersection(FORBIDDEN_INPUT_COLUMNS)
    if leaked:
        raise ValueError(
            f"DEC-015 VIOLATION: Forbidden columns found in feature matrix: {sorted(list(leaked))}"
        )
        
    for col in cols:
        # Prevent accidental inclusion of post-disruption fields
        lower_c = col.lower()
        if "disrupt" in lower_c and col != "disruption_radius_m":
            raise ValueError(f"DEC-015 VIOLATION: Potential disruption leakage column '{col}' detected.")
        if "target" in lower_c:
            raise ValueError(f"DEC-015 VIOLATION: Target column '{col}' detected in features.")


class FrozenPreprocessor:
    """Leakage-safe preprocessor fitted strictly on Train split and frozen.
    
    Fits standard scaling parameters exclusively on Seattle Train observations.
    Transforms Validation, In-City Test, and Zero-Shot Transfer data without refitting.
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame) -> "FrozenPreprocessor":
        assert_leakage_safe(X_train)
        self.scaler.fit(X_train[self.feature_names])
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on Train data before transform.")
        assert_leakage_safe(X)
        scaled_vals = self.scaler.transform(X[self.feature_names])
        return pd.DataFrame(scaled_vals, columns=self.feature_names, index=X.index)


def partition_benchmark_splits(
    X: pd.DataFrame,
    y_reach: pd.Series,
    y_detour: pd.Series,
    meta: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Partition the dataset into Train, Val, Test, and Transfer splits.
    
    Verifies zero scenario leakage across partitions.
    """
    assert_leakage_safe(X)
    
    splits = {}
    for split_name in ["train", "val", "test", "transfer"]:
        mask = meta["split"] == split_name
        splits[split_name] = {
            "X": X.loc[mask].copy(),
            "y_reach": y_reach.loc[mask].copy(),
            "y_detour": y_detour.loc[mask].copy(),
            "meta": meta.loc[mask].copy(),
        }
        
    # Verify scenario isolation
    train_scenarios = set(splits["train"]["meta"]["scenario_id"].unique())
    val_scenarios = set(splits["val"]["meta"]["scenario_id"].unique())
    test_scenarios = set(splits["test"]["meta"]["scenario_id"].unique())
    transfer_scenarios = set(splits["transfer"]["meta"]["scenario_id"].unique())
    
    assert len(train_scenarios.intersection(val_scenarios)) == 0, "Scenario leakage: Train <-> Val"
    assert len(train_scenarios.intersection(test_scenarios)) == 0, "Scenario leakage: Train <-> Test"
    assert len(train_scenarios.intersection(transfer_scenarios)) == 0, "Scenario leakage: Train <-> Transfer"
    assert len(val_scenarios.intersection(test_scenarios)) == 0, "Scenario leakage: Val <-> Test"
    assert len(test_scenarios.intersection(transfer_scenarios)) == 0, "Scenario leakage: Test <-> Transfer"
    
    return splits
