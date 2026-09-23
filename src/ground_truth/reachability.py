"""Exact Ground-Truth Accessibility and Shortest-Path Calculation.

Uses deterministic Dijkstra on physical road edge lengths to compute:
  1. original_distance_m: Exact shortest-path distance before disruption.
  2. disrupted_distance_m: Exact shortest-path distance after disruption (None if unreachable).
  3. reachable: Binary 1 (reachable) or 0 (unreachable).
  4. relative_detour: (disrupted_distance - original_distance) / original_distance if reachable, else None.
"""

from typing import Dict, Any, Optional
import networkx as nx


def compute_ground_truth_sample(
    G_orig: nx.DiGraph,
    G_disrupt: nx.DiGraph,
    origin_node: Any,
    facility_node: Any,
    weight: str = "length",
) -> Dict[str, Any]:
    """Compute exact ground truth before and after disruption for an (origin, facility) pair."""
    # 1. Pre-disruption Dijkstra
    orig_reachable = False
    orig_distance = None
    orig_path = None
    
    if nx.has_path(G_orig, origin_node, facility_node):
        orig_reachable = True
        try:
            orig_distance = float(nx.shortest_path_length(G_orig, origin_node, facility_node, weight=weight))
            orig_path = nx.shortest_path(G_orig, origin_node, facility_node, weight=weight)
        except nx.NetworkXNoPath:
            orig_reachable = False
            
    # 2. Post-disruption Dijkstra
    disrupt_reachable = False
    disrupt_distance = None
    disrupt_path = None
    
    if orig_reachable and nx.has_path(G_disrupt, origin_node, facility_node):
        disrupt_reachable = True
        try:
            disrupt_distance = float(nx.shortest_path_length(G_disrupt, origin_node, facility_node, weight=weight))
            disrupt_path = nx.shortest_path(G_disrupt, origin_node, facility_node, weight=weight)
        except nx.NetworkXNoPath:
            disrupt_reachable = False
            
    # 3. Targets and Detour
    reachable_target = 1 if disrupt_reachable else 0
    relative_detour = None
    delta_distance = None
    
    if orig_reachable and disrupt_reachable:
        delta_distance = float(disrupt_distance - orig_distance)
        # Numerical tolerance for zero delta
        if delta_distance < 0 and abs(delta_distance) < 1e-4:
            delta_distance = 0.0
            disrupt_distance = orig_distance
            
        relative_detour = float(delta_distance / max(orig_distance, 1e-6))
        if relative_detour < 0 and abs(relative_detour) < 1e-5:
            relative_detour = 0.0
            
    return {
        "origin_node": origin_node,
        "destination_node": facility_node,
        "original_distance_m": round(orig_distance, 2) if orig_distance is not None else None,
        "disrupted_distance_m": round(disrupt_distance, 2) if disrupt_distance is not None else None,
        "reachable": reachable_target,
        "relative_detour": round(relative_detour, 6) if relative_detour is not None else None,
        "orig_path_hops": len(orig_path) if orig_path else 0,
        "disrupt_path_hops": len(disrupt_path) if disrupt_path else 0,
        "orig_path_nodes": orig_path,
    }
