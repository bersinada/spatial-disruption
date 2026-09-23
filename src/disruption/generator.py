"""Deterministic Counterfactual Disruption Generator.

Implements localized spatial disruption protocol:
  - Epicenter strictly inside municipal boundary.
  - Disrupts all canonical directed edges within a specified radius (100m to 250m).
  - Represents disruption using BOTH:
      1. explicit list of disabled canonical edge IDs [(u1, v1), (u2, v2), ...]
      2. edge-level binary disruption mask { (u, v): 1 (operational) or 0 (disrupted) }
"""

from typing import List, Tuple, Dict, Any, Optional
import random
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point


def get_candidate_epicenters(
    G: nx.DiGraph,
    boundary_gdf: gpd.GeoDataFrame,
    min_degree: int = 3,
) -> List[Any]:
    """Find routing junction nodes strictly inside the municipal boundary.
    
    Prefers arterial junctions (primary, secondary, tertiary) with degree >= min_degree.
    """
    poly = boundary_gdf.geometry.iloc[0]
    candidate_nodes = []
    
    for n, data in G.nodes(data=True):
        pt = Point(data["x"], data["y"])
        if not poly.contains(pt):
            continue
            
        # Check degree
        if G.degree(n) < min_degree:
            continue
            
        # Check incident road classes (prefer arterials/connectors)
        incident_hw = set()
        for _, _, edata in G.in_edges(n, data=True):
            incident_hw.add(edata.get("highway", ""))
        for _, _, edata in G.out_edges(n, data=True):
            incident_hw.add(edata.get("highway", ""))
            
        if any(h in incident_hw for h in ["primary", "secondary", "tertiary", "trunk", "residential"]):
            candidate_nodes.append(n)
            
    return candidate_nodes if candidate_nodes else list(G.nodes())


def generate_localized_disruption(
    G: nx.DiGraph,
    scenario_id: str,
    city: str,
    split: str,
    seed: int,
    boundary_gdf: gpd.GeoDataFrame,
    radius_meters: Optional[float] = None,
    epicenter_node: Optional[Any] = None,
) -> Tuple[nx.DiGraph, Dict[str, Any], Dict[Tuple[Any, Any], int]]:
    """Deterministically generate a localized disruption scenario.
    
    Args:
        G: Intact canonical road DiGraph.
        scenario_id: Unique scenario identifier (e.g. 'seattle_s01').
        city: City name ('seattle' or 'portland').
        split: Split assignment ('train', 'val', 'test', 'transfer').
        seed: Deterministic integer seed.
        boundary_gdf: Official municipal boundary GeoDataFrame.
        radius_meters: Optional explicit radius in meters (sampled from [100, 250] if None).
        epicenter_node: Optional explicit epicenter node.
        
    Returns:
        Tuple of:
          - G_disrupt: Disrupted DiGraph with disabled edges removed.
          - scenario_metadata: Structured metadata dict.
          - edge_disruption_mask: Dict mapping (u, v) -> 1 (operational) or 0 (disrupted).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # 1. Select Epicenter strictly inside municipal boundary
    if epicenter_node is None or epicenter_node not in G:
        candidate_epicenters = get_candidate_epicenters(G, boundary_gdf)
        if not candidate_epicenters:
            raise ValueError(f"No candidate epicenters found inside municipal boundary for {city}.")
        epicenter_node = rng.choice(candidate_epicenters)
        
    center_x = G.nodes[epicenter_node]["x"]
    center_y = G.nodes[epicenter_node]["y"]
    
    # 2. Select Radius in range [100m, 250m] if not specified
    if radius_meters is None:
        radius_meters = float(rng.choice([100.0, 150.0, 200.0, 250.0]))
        
    # Local planar metric projection factors
    lat_rad = np.radians(center_y)
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(lat_rad)
    
    # 3. Identify all edges within disruption radius
    disrupted_edge_ids = []
    
    for u, v, data in G.edges(data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        
        d_u = np.hypot((ux - center_x) * m_per_deg_lon, (uy - center_y) * m_per_deg_lat)
        d_v = np.hypot((vx - center_x) * m_per_deg_lon, (vy - center_y) * m_per_deg_lat)
        d_mid = np.hypot(((ux + vx)/2 - center_x) * m_per_deg_lon, ((uy + vy)/2 - center_y) * m_per_deg_lat)
        
        if min(d_u, d_v, d_mid) <= radius_meters:
            disrupted_edge_ids.append((u, v))
            
    # Guarantee at least incident edges are disrupted if tiny radius
    if not disrupted_edge_ids:
        for v in G.successors(epicenter_node):
            disrupted_edge_ids.append((epicenter_node, v))
            
    # 4. Construct disrupted graph and edge-level binary mask (DEC-013)
    G_disrupt = G.copy()
    G_disrupt.remove_edges_from(disrupted_edge_ids)
    
    canonical_edges = sorted(list(G.edges()))
    disrupted_edge_set = set(disrupted_edge_ids)
    
    edge_disruption_mask = {}
    disrupted_indices = []
    mask_chars = []
    
    for idx, (u, v) in enumerate(canonical_edges):
        if (u, v) in disrupted_edge_set:
            edge_disruption_mask[(u, v)] = 0
            disrupted_indices.append(idx)
            mask_chars.append("0")
        else:
            edge_disruption_mask[(u, v)] = 1
            mask_chars.append("1")
            
    edge_disruption_mask_binary = "".join(mask_chars)
        
    scenario_metadata = {
        "scenario_id": scenario_id,
        "city": city,
        "split": split,
        "seed": seed,
        "disruption_center": {
            "node_id": epicenter_node,
            "x": center_x,
            "y": center_y,
        },
        "disruption_radius_m": radius_meters,
        "disrupted_edge_ids": [[u, v] for u, v in disrupted_edge_ids],
        "number_of_disrupted_edges": len(disrupted_edge_ids),
        "original_edge_count": G.number_of_edges(),
        "disrupted_edge_count": G_disrupt.number_of_edges(),
        "canonical_edge_count": len(canonical_edges),
        "disrupted_canonical_edge_indices": disrupted_indices,
        "edge_disruption_mask_binary": edge_disruption_mask_binary,
    }
    
    return G_disrupt, scenario_metadata, edge_disruption_mask
