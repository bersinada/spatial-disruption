"""Inductive Relational Graph Neural Network for Counterfactual Accessibility (E03).

Implements:
  1. Inductive RGCN node encoder with typed relations (connects_to, accessible_from, serves).
  2. Pairwise interaction between origin road junctions and destination facility entities.
  3. Multi-task prediction: Post-disruption reachability classifier + smooth detour regressor.
  4. Standardized evaluation protocol identical to E01 and E02.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
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


class InductiveRelationalGNN(nn.Module):
    """Inductive Relational GNN (RGCN) modeling road junctions and critical facility relations."""
    
    def __init__(
        self,
        node_in_dim: int = 13,
        pair_in_dim: int = 9,
        hidden_dim: int = 64,
        num_relations: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.pair_in_dim = pair_in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # Node projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Multi-relational GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        
        # Pairwise interaction dimension: [h_o, h_d, |h_o - h_d|, h_o * h_d, pair_features]
        pair_repr_dim = (hidden_dim * 4) + pair_in_dim
        
        # Primary Task: Reachability Head (Binary Logits)
        self.reach_head = nn.Sequential(
            nn.Linear(pair_repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Secondary Task: Relative Detour Head (Smooth Non-negative)
        self.detour_head = nn.Sequential(
            nn.Linear(pair_repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        origin_idx: torch.Tensor,
        dest_idx: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a scenario multi-relational graph.
        
        Args:
            x: Node feature tensor [N_total, D_node].
            edge_index: Relational edges [2, E_rel].
            edge_type: Relation type tensor [E_rel] (0: connects_to, 1: accessible_from, 2: serves).
            origin_idx: Query origin junction indices [B].
            dest_idx: Query destination facility indices [B].
            pair_features: Pre-disruption query pairwise routing attributes [B, D_pair].
        """
        # 1. Project node attributes
        h = self.node_proj(x)
        h_res = h
        
        # 2. Relational message passing
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_type)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            
        # Residual connection
        h = h + h_res
        
        # 3. Formulate query OD pair representation (Junction -> Facility)
        h_o = h[origin_idx]
        h_d = h[dest_idx]
        
        diff = torch.abs(h_o - h_d)
        prod = h_o * h_d
        
        z_od = torch.cat([h_o, h_d, diff, prod, pair_features], dim=-1)
        
        # 4. Predict targets
        reach_logits = self.reach_head(z_od).squeeze(-1)
        detour_preds = self.detour_head(z_od).squeeze(-1)
        
        return reach_logits, detour_preds
        
    def compute_loss(
        self,
        reach_logits: torch.Tensor,
        detour_preds: torch.Tensor,
        y_reach: torch.Tensor,
        y_detour: torch.Tensor,
        detour_mask: torch.Tensor,
        detour_loss_weight: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-task loss for reachability and conditional detour."""
        loss_reach = F.binary_cross_entropy_with_logits(reach_logits, y_reach)
        
        loss_detour = torch.tensor(0.0, device=reach_logits.device)
        if detour_mask.any():
            loss_detour = F.l1_loss(detour_preds[detour_mask], y_detour[detour_mask])
            
        total_loss = loss_reach + detour_loss_weight * loss_detour
        return total_loss, loss_reach, loss_detour


def evaluate_relational_scenarios(
    model: InductiveRelationalGNN,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate Relational GNN model across a list of scenarios."""
    model.eval()
    
    all_y_reach = []
    all_reach_probs = []
    all_reach_preds = []
    
    all_y_detour = []
    all_detour_preds = []
    
    with torch.no_grad():
        for sc in scenarios:
            logits, detours = model(
                x=sc["x"],
                edge_index=sc["edge_index"],
                edge_type=sc["edge_type"],
                origin_idx=sc["origin_idx"],
                dest_idx=sc["dest_idx"],
                pair_features=sc["pair_features"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_r = sc["y_reach"].cpu().numpy()
            
            all_y_reach.extend(y_r)
            all_reach_probs.extend(probs)
            all_reach_preds.extend(preds)
            
            # Detour metrics on reachable pairs
            mask = sc["detour_mask"].cpu().numpy()
            if mask.any():
                all_y_detour.extend(sc["y_detour"].cpu().numpy()[mask])
                all_detour_preds.extend(detours.cpu().numpy()[mask])
                
    y_reach_np = np.array(all_y_reach)
    probs_np = np.array(all_reach_probs)
    preds_np = np.array(all_reach_preds)
    
    try:
        roc_auc = float(roc_auc_score(y_reach_np, probs_np))
    except ValueError:
        roc_auc = float("nan")
        
    try:
        pr_auc = float(average_precision_score(y_reach_np, probs_np))
    except ValueError:
        pr_auc = float("nan")
        
    reach_metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(float(accuracy_score(y_reach_np, preds_np)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_reach_np, preds_np)), 4),
        "precision": round(float(precision_score(y_reach_np, preds_np, zero_division=0)), 4),
        "recall": round(float(recall_score(y_reach_np, preds_np, zero_division=0)), 4),
        "f1": round(float(f1_score(y_reach_np, preds_np, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_reach_np, probs_np)), 4),
        "log_loss": round(float(log_loss(y_reach_np, probs_np, labels=[0, 1])), 4),
        "support_total": int(len(y_reach_np)),
        "support_pos": int(np.sum(y_reach_np == 1)),
        "support_neg": int(np.sum(y_reach_np == 0)),
    }
    
    y_detour_np = np.array(all_y_detour)
    detour_preds_np = np.array(all_detour_preds)
    
    if len(y_detour_np) > 0:
        mae = float(mean_absolute_error(y_detour_np, detour_preds_np))
        medae = float(median_absolute_error(y_detour_np, detour_preds_np))
        rmse = float(np.sqrt(mean_squared_error(y_detour_np, detour_preds_np)))
        
        if len(y_detour_np) > 2 and np.std(detour_preds_np) > 1e-6 and np.std(y_detour_np) > 1e-6:
            r_val, _ = stats.pearsonr(y_detour_np, detour_preds_np)
            rho_val, _ = stats.spearmanr(y_detour_np, detour_preds_np)
        else:
            r_val, rho_val = 0.0, 0.0
            
        detour_metrics = {
            "mae": round(mae, 4),
            "median_ae": round(medae, 4),
            "rmse": round(rmse, 4),
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "support_reachable": int(len(y_detour_np)),
        }
    else:
        detour_metrics = {
            "mae": float("nan"),
            "median_ae": float("nan"),
            "rmse": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "support_reachable": 0,
        }
        
    return {
        "reachability": reach_metrics,
        "detour": detour_metrics,
    }
