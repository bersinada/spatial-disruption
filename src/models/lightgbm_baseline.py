"""LightGBM Baseline Models for Counterfactual Urban Accessibility (E01).

Implements:
  1. Primary Model: Reachability Binary Classifier (post-disruption reachability 0/1).
  2. Secondary Model: Conditional Relative Detour Regressor (for reachable pairs).
  3. Comprehensive evaluation metrics (ROC-AUC, PR-AUC, Brier score, MAE, Pearson r).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


class ReachabilityClassifier:
    """Primary LightGBM classifier for binary post-disruption reachability."""
    
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "verbose": -1,
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_names: List[str] = []
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
    ) -> "ReachabilityClassifier":
        self.feature_names = list(X_train.columns)
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)] if X_val is not None else []
        
        self.model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=callbacks,
        )
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        y = y_true.values
        
        # Handle cases where only one class exists in target split
        try:
            roc_auc = float(roc_auc_score(y, probs))
        except ValueError:
            roc_auc = float("nan")
            
        try:
            pr_auc = float(average_precision_score(y, probs))
        except ValueError:
            pr_auc = float("nan")
            
        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y, probs)), 4),
            "log_loss": round(float(log_loss(y, probs, labels=[0, 1])), 4),
            "support_total": int(len(y)),
            "support_pos": int(np.sum(y == 1)),
            "support_neg": int(np.sum(y == 0)),
        }
        
    def get_feature_importances(self) -> pd.DataFrame:
        splits = self.model.booster_.feature_importance(importance_type="split")
        gains = self.model.booster_.feature_importance(importance_type="gain")
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "split_importance": splits,
            "gain_importance": np.round(gains, 2),
        }).sort_values(by="gain_importance", ascending=False).reset_index(drop=True)
        return df_imp


class RelativeDetourRegressor:
    """Secondary LightGBM regressor for conditional relative detour on reachable pairs."""
    
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 25,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "huber",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "verbose": -1,
        }
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_names: List[str] = []
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
    ) -> "RelativeDetourRegressor":
        # Strictly train on valid non-null detour targets
        mask_train = y_train.notna()
        X_tr = X_train.loc[mask_train]
        y_tr = y_train.loc[mask_train]
        self.feature_names = list(X_tr.columns)
        
        callbacks = []
        eval_X = None
        eval_y = None
        if X_val is not None and y_val is not None:
            mask_val = y_val.notna()
            eval_X = X_val.loc[mask_val]
            eval_y = y_val.loc[mask_val]
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            
        self.model.fit(
            X_tr,
            y_tr,
            eval_X=eval_X,
            eval_y=eval_y,
            callbacks=callbacks,
        )
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names])
        # Detour cannot physically be negative
        return np.maximum(0.0, preds)
        
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        mask = y_true.notna()
        if not mask.any():
            return {"mae": float("nan"), "rmse": float("nan"), "support": 0}
            
        X_eval = X.loc[mask]
        y_eval = y_true.loc[mask].values
        preds = self.predict(X_eval)
        
        mae = float(mean_absolute_error(y_eval, preds))
        medae = float(median_absolute_error(y_eval, preds))
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        
        if len(y_eval) > 2 and np.std(preds) > 1e-6 and np.std(y_eval) > 1e-6:
            r_val, p_val = stats.pearsonr(y_eval, preds)
            rho_val, _ = stats.spearmanr(y_eval, preds)
        else:
            r_val, rho_val = 0.0, 0.0
            
        return {
            "mae": round(mae, 4),
            "median_ae": round(medae, 4),
            "rmse": round(rmse, 4),
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "support_reachable": int(len(y_eval)),
        }
