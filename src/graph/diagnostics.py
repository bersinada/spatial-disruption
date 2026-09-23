"""Graph Diagnostics and Topological Validation."""

from typing import Dict, Any
import numpy as np
import networkx as nx


def compute_graph_diagnostics(
    G: nx.DiGraph,
    facility_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute comprehensive topological, geometric, and facility diagnostics.
    
    Args:
        G: Physical road DiGraph.
        facility_stats: Statistics from facility mapping.
        
    Returns:
        Structured diagnostics dictionary.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # 1. Connected components (Weakly and Strongly)
    wcc = list(nx.weakly_connected_components(G))
    num_wcc = len(wcc)
    largest_wcc_size = len(max(wcc, key=len)) if wcc else 0
    wcc_fraction = largest_wcc_size / max(num_nodes, 1)
    
    scc = list(nx.strongly_connected_components(G))
    num_scc = len(scc)
    largest_scc_size = len(max(scc, key=len)) if scc else 0
    scc_fraction = largest_scc_size / max(num_nodes, 1)
    
    # 2. Degree distributions
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [in_d + out_d for in_d, out_d in zip(in_degrees, out_degrees)]
    
    isolated_nodes = [n for n, d in G.degree() if d == 0]
    dead_ends_out = [n for n, d in G.out_degree() if d == 0]
    dead_ends_in = [n for n, d in G.in_degree() if d == 0]
    
    # 3. Road length distribution
    lengths = [data.get("length", 0.0) for _, _, data in G.edges(data=True)]
    total_length_km = float(np.sum(lengths) / 1000.0) if lengths else 0.0
    
    # 4. Highway class breakdown
    highway_counts = {}
    bridge_count = 0
    tunnel_count = 0
    for _, _, data in G.edges(data=True):
        hw = data.get("highway", "unknown")
        highway_counts[hw] = highway_counts.get(hw, 0) + 1
        if data.get("bridge"):
            bridge_count += 1
        if data.get("tunnel"):
            tunnel_count += 1
            
    diagnostics = {
        "network_summary": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "total_network_length_km": round(total_length_km, 2),
            "bridge_edges": bridge_count,
            "tunnel_edges": tunnel_count,
        },
        "connectivity": {
            "weakly_connected_components": num_wcc,
            "largest_wcc_nodes": largest_wcc_size,
            "largest_wcc_ratio": round(wcc_fraction, 4),
            "strongly_connected_components": num_scc,
            "largest_scc_nodes": largest_scc_size,
            "largest_scc_ratio": round(scc_fraction, 4),
            "isolated_nodes_degree_0": len(isolated_nodes),
            "source_only_nodes_in_0": len(dead_ends_in),
            "sink_only_nodes_out_0": len(dead_ends_out),
        },
        "degree_distribution": {
            "mean_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "max_in_degree": int(np.max(in_degrees)) if in_degrees else 0,
            "mean_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "max_out_degree": int(np.max(out_degrees)) if out_degrees else 0,
            "mean_total_degree": float(np.mean(total_degrees)) if total_degrees else 0.0,
            "median_total_degree": float(np.median(total_degrees)) if total_degrees else 0.0,
        },
        "length_distribution_meters": {
            "min_length_m": round(float(np.min(lengths)), 2) if lengths else 0.0,
            "mean_length_m": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "median_length_m": round(float(np.median(lengths)), 2) if lengths else 0.0,
            "max_length_m": round(float(np.max(lengths)), 2) if lengths else 0.0,
            "std_length_m": round(float(np.std(lengths)), 2) if lengths else 0.0,
        },
        "facility_mapping": facility_stats,
        "highway_classes": highway_counts,
    }
    
    return diagnostics
