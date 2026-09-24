"""Tabular Feature Engineering for E01 LightGBM Baseline.

Extracts continuous spatial, pre-disruption network, facility, and hazard geometry
features strictly complying with DEC-015 (Feature Leakage Contract).

Permitted features:
  - Origin / Destination coordinates
  - Projected Euclidean distance and bearing
  - Pre-disruption network distance (original_distance_m), hops, circuity
  - Facility amenity and snap distance
  - Disruption hazard geometry (epicenter distance, radius, chord clearance)
  - Intact network degree and H3 spatial density
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import h3

from src.data.osm_loader import load_city_data
from src.graph.physical_graph import build_physical_digraph


def project_latlng_to_meters(
    lons: np.ndarray,
    lats: np.ndarray,
    ref_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lon/lat arrays to local metric coordinates (x_m, y_m) relative to a reference latitude."""
    lat_rad = math.radians(ref_lat)
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * math.cos(lat_rad)
    x_m = lons * m_per_deg_lon
    y_m = lats * m_per_deg_lat
    return x_m, y_m


def point_to_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> np.ndarray:
    """Vectorized calculation of shortest distance from points (px, py) to line segments [(x1, y1), (x2, y2)]."""
    dx = x2 - x1
    dy = y2 - y1
    lensq = dx * dx + dy * dy
    
    # Avoid zero division
    lensq_safe = np.where(lensq > 1e-6, lensq, 1.0)
    
    # Projection factor t = ((px - x1)*dx + (py - y1)*dy) / lensq
    t = ((px - x1) * dx + (py - y1) * dy) / lensq_safe
    t_clamped = np.clip(t, 0.0, 1.0)
    
    # Projected point
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    
    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return dist


def precompute_city_graph_features(
    city: str,
    raw_dir: str = "data/raw",
) -> Dict[str, Any]:
    """Precompute intact topological and H3 spatial density features for a city.
    
    Strictly uses pre-disruption intact graph; zero test scenario or post-disruption leakage.
    """
    _, G_multi, _ = load_city_data(city, raw_dir=raw_dir)
    G = build_physical_digraph(G_multi)
    
    node_degrees = {}
    node_in_degrees = {}
    node_out_degrees = {}
    node_street_counts = {}
    h3_res8_counts = {}
    h3_res9_counts = {}
    
    node_to_h8 = {}
    node_to_h9 = {}
    
    for n, d in G.nodes(data=True):
        lat = d["y"]
        lon = d["x"]
        deg = G.degree(n)
        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)
        sc = int(d.get("street_count", deg))
        
        node_degrees[n] = deg
        node_in_degrees[n] = in_deg
        node_out_degrees[n] = out_deg
        node_street_counts[n] = sc
        
        h8 = h3.latlng_to_cell(lat, lon, 8)
        h9 = h3.latlng_to_cell(lat, lon, 9)
        node_to_h8[n] = h8
        node_to_h9[n] = h9
        
        h3_res8_counts[h8] = h3_res8_counts.get(h8, 0) + 1
        h3_res9_counts[h9] = h3_res9_counts.get(h9, 0) + 1
        
    return {
        "degrees": node_degrees,
        "in_degrees": node_in_degrees,
        "out_degrees": node_out_degrees,
        "street_counts": node_street_counts,
        "h8_counts": h3_res8_counts,
        "h9_counts": h3_res9_counts,
        "node_to_h8": node_to_h8,
        "node_to_h9": node_to_h9,
    }


def extract_tabular_features(
    samples_df: pd.DataFrame,
    scenarios_metadata_path: str = "data/processed/benchmark_scenarios_metadata.json",
    raw_dir: str = "data/raw",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Extract full feature set for E01 LightGBM baseline.
    
    Returns:
        X: DataFrame of permitted model input features.
        y_reach: Series of primary binary target ('reachable').
        y_detour: Series of secondary relative detour target (float or NaN).
        metadata: DataFrame containing identifiers and split metadata for auditing.
    """
    df = samples_df.copy()
    
    # Load scenario hazard metadata
    with open(scenarios_metadata_path) as fp:
        meta_list = json.load(fp)
    meta_by_id = {m["scenario_id"]: m for m in meta_list}
    
    # Precompute intact graph features per city
    city_graph_feats = {}
    for city in df["city"].unique():
        city_graph_feats[city] = precompute_city_graph_features(city, raw_dir=raw_dir)
        
    # Attach hazard geometry from metadata (without keeping scenario_id in X)
    hazard_x = []
    hazard_y = []
    hazard_r = []
    
    for _, row in df.iterrows():
        sid = row["scenario_id"]
        sm = meta_by_id[sid]
        hazard_x.append(sm["disruption_center"]["x"])
        hazard_y.append(sm["disruption_center"]["y"])
        hazard_r.append(sm["disruption_radius_m"])
        
    df["epicenter_x"] = hazard_x
    df["epicenter_y"] = hazard_y
    df["disruption_radius_m"] = hazard_r
    
    # Projected metric coordinates
    avg_lat = float(df["origin_y"].mean())
    ox_m, oy_m = project_latlng_to_meters(df["origin_x"].values, df["origin_y"].values, avg_lat)
    dx_m, dy_m = project_latlng_to_meters(df["destination_x"].values, df["destination_y"].values, avg_lat)
    ex_m, ey_m = project_latlng_to_meters(df["epicenter_x"].values, df["epicenter_y"].values, avg_lat)
    
    # 1. Continuous Euclidean geometry
    delta_x_m = dx_m - ox_m
    delta_y_m = dy_m - oy_m
    euclidean_dist_m = np.sqrt(delta_x_m ** 2 + delta_y_m ** 2)
    bearing_rad = np.arctan2(delta_x_m, delta_y_m)
    bearing_deg = np.degrees(bearing_rad) % 360.0
    
    # 2. Hazard geometry & proximity features
    dist_origin_to_epicenter_m = np.sqrt((ox_m - ex_m) ** 2 + (oy_m - ey_m) ** 2)
    dist_dest_to_epicenter_m = np.sqrt((dx_m - ex_m) ** 2 + (dy_m - ey_m) ** 2)
    min_dist_od_to_epicenter_m = np.minimum(dist_origin_to_epicenter_m, dist_dest_to_epicenter_m)
    max_dist_od_to_epicenter_m = np.maximum(dist_origin_to_epicenter_m, dist_dest_to_epicenter_m)
    
    origin_in_hazard = (dist_origin_to_epicenter_m <= df["disruption_radius_m"].values).astype(int)
    dest_in_hazard = (dist_dest_to_epicenter_m <= df["disruption_radius_m"].values).astype(int)
    
    # Orthogonal distance from epicenter to the origin-destination Euclidean chord
    epicenter_to_od_line_dist_m = point_to_segment_distance(
        ex_m, ey_m, ox_m, oy_m, dx_m, dy_m
    )
    hazard_intersects_od_segment = (epicenter_to_od_line_dist_m <= df["disruption_radius_m"].values).astype(int)
    excess_dist_via_epicenter_m = (dist_origin_to_epicenter_m + dist_dest_to_epicenter_m) - euclidean_dist_m
    
    # 3. Intact pre-disruption network features
    orig_dist_m = df["original_distance_m"].values
    orig_hops = df["orig_path_hops"].values
    network_circuity = orig_dist_m / np.maximum(euclidean_dist_m, 1.0)
    orig_avg_edge_length = orig_dist_m / np.maximum(orig_hops, 1)
    
    # 4. Facility features
    facility_is_hospital = (df["facility_amenity"] == "hospital").astype(int)
    facility_snap_dist = df["facility_snap_distance_m"].values
    
    # 5. Local intact network degree & H3 density
    origin_street_counts = []
    origin_degrees = []
    origin_in_degrees = []
    origin_out_degrees = []
    origin_h8_density = []
    origin_h9_density = []
    
    dest_street_counts = []
    dest_degrees = []
    dest_in_degrees = []
    dest_out_degrees = []
    dest_h8_density = []
    dest_h9_density = []
    
    for _, row in df.iterrows():
        c = row["city"]
        c_feats = city_graph_feats[c]
        onode = row["origin_node"]
        dnode = row["destination_node"]
        
        origin_street_counts.append(c_feats["street_counts"].get(onode, 3))
        origin_degrees.append(c_feats["degrees"].get(onode, 3))
        origin_in_degrees.append(c_feats["in_degrees"].get(onode, 2))
        origin_out_degrees.append(c_feats["out_degrees"].get(onode, 2))
        h8_o = c_feats["node_to_h8"].get(onode)
        h9_o = c_feats["node_to_h9"].get(onode)
        origin_h8_density.append(c_feats["h8_counts"].get(h8_o, 0))
        origin_h9_density.append(c_feats["h9_counts"].get(h9_o, 0))
        
        dest_street_counts.append(c_feats["street_counts"].get(dnode, 3))
        dest_degrees.append(c_feats["degrees"].get(dnode, 3))
        dest_in_degrees.append(c_feats["in_degrees"].get(dnode, 2))
        dest_out_degrees.append(c_feats["out_degrees"].get(dnode, 2))
        h8_d = c_feats["node_to_h8"].get(dnode)
        h9_d = c_feats["node_to_h9"].get(dnode)
        dest_h8_density.append(c_feats["h8_counts"].get(h8_d, 0))
        dest_h9_density.append(c_feats["h9_counts"].get(h9_d, 0))
        
    feature_dict = {
        # Spatial Coordinates (DEC-015 permitted)
        "origin_x": df["origin_x"].values,
        "origin_y": df["origin_y"].values,
        "destination_x": df["destination_x"].values,
        "destination_y": df["destination_y"].values,
        "delta_x_m": delta_x_m,
        "delta_y_m": delta_y_m,
        "euclidean_distance_m": euclidean_dist_m,
        "bearing_deg": bearing_deg,
        
        # Pre-disruption Intact Routing (DEC-015 permitted)
        "original_distance_m": orig_dist_m,
        "orig_path_hops": orig_hops,
        "network_circuity": network_circuity,
        "orig_avg_edge_length": orig_avg_edge_length,
        
        # Facility Metadata (DEC-015 permitted)
        "facility_is_hospital": facility_is_hospital,
        "facility_snap_distance_m": facility_snap_dist,
        
        # Disruption Hazard Geometry (DEC-015 permitted)
        "disruption_radius_m": df["disruption_radius_m"].values,
        "dist_origin_to_epicenter_m": dist_origin_to_epicenter_m,
        "dist_dest_to_epicenter_m": dist_dest_to_epicenter_m,
        "min_dist_od_to_epicenter_m": min_dist_od_to_epicenter_m,
        "max_dist_od_to_epicenter_m": max_dist_od_to_epicenter_m,
        "origin_in_hazard_zone": origin_in_hazard,
        "dest_in_hazard_zone": dest_in_hazard,
        "epicenter_to_od_line_dist_m": epicenter_to_od_line_dist_m,
        "hazard_intersects_od_segment": hazard_intersects_od_segment,
        "excess_dist_via_epicenter_m": excess_dist_via_epicenter_m,
        
        # Local Intact Topology & Spatial Density (DEC-015 permitted)
        "origin_street_count": np.array(origin_street_counts),
        "origin_degree": np.array(origin_degrees),
        "origin_in_degree": np.array(origin_in_degrees),
        "origin_out_degree": np.array(origin_out_degrees),
        "destination_street_count": np.array(dest_street_counts),
        "destination_degree": np.array(dest_degrees),
        "destination_in_degree": np.array(dest_in_degrees),
        "destination_out_degree": np.array(dest_out_degrees),
        "h3_origin_node_density_res8": np.array(origin_h8_density),
        "h3_origin_node_density_res9": np.array(origin_h9_density),
        "h3_dest_node_density_res8": np.array(dest_h8_density),
        "h3_dest_node_density_res9": np.array(dest_h9_density),
    }
    
    X = pd.DataFrame(feature_dict, index=df.index)
    y_reach = df["reachable"].copy()
    y_detour = df["relative_detour"].copy()
    
    # Metadata for splitting and tracking
    metadata_cols = [
        "sample_id",
        "city",
        "scenario_id",
        "split",
        "sample_type",
        "seed",
        "distance_tier",
    ]
    meta_df = df[[c for c in metadata_cols if c in df.columns]].copy()
    
    return X, y_reach, y_detour, meta_df
