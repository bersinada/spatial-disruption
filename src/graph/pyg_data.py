"""PyG Graph Dataset and Scenario Graph Builder for E02 (GraphSAGE).

Prepares PyTorch Geometric representations of physical road networks under localized disruptions:
  - Canonical nodes with continuous spatial, degree, H3 density, and hazard proximity attributes.
  - Operational edge indices filtered per scenario disruption mask (DEC-013).
  - Scenario query OD pairs with permitted pre-disruption routing attributes (DEC-015).
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import h3

from src.data.osm_loader import load_city_data
from src.graph.physical_graph import build_physical_digraph


class CityGraphData:
    """Holds static physical graph topology and precomputed node/edge attributes for a city."""
    
    def __init__(
        self,
        city: str,
        raw_dir: str = "data/raw",
    ):
        self.city = city
        self.raw_dir = raw_dir
        
        # Load raw data and build canonical DiGraph
        b_gdf, G_multi, fac_gdf = load_city_data(city, raw_dir=raw_dir)
        self.G = build_physical_digraph(G_multi)
        
        # Node ordering
        self.node_list = sorted(list(self.G.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)
        
        # Canonical edge ordering
        self.canonical_edges = sorted(list(self.G.edges()))
        self.canonical_edge_indices = [
            (self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.canonical_edges
        ]
        self.canonical_edge_tensor = torch.tensor(
            self.canonical_edge_indices, dtype=torch.long
        ).t().contiguous()
        self.num_edges = len(self.canonical_edges)
        
        # City centroid for invariant local metric projection
        all_lons = np.array([self.G.nodes[n]["x"] for n in self.node_list])
        all_lats = np.array([self.G.nodes[n]["y"] for n in self.node_list])
        self.center_lon = float(all_lons.mean())
        self.center_lat = float(all_lats.mean())
        
        lat_rad = math.radians(self.center_lat)
        self.m_per_deg_lat = 111000.0
        self.m_per_deg_lon = 111000.0 * math.cos(lat_rad)
        
        # Compute local metric coordinates in km relative to city center
        self.node_x_km = (all_lons - self.center_lon) * self.m_per_deg_lon / 1000.0
        self.node_y_km = (all_lats - self.center_lat) * self.m_per_deg_lat / 1000.0
        
        # Identify facility nodes
        self.facility_node_set = set()
        self.hospital_node_set = set()
        for _, row in fac_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            pt = geom.centroid if geom.geom_type != "Point" else geom
            # Find closest surface node
            # In benchmark, destination_node is already exact snapped node
            
        # H3 density
        h8_counts = {}
        h9_counts = {}
        node_h8 = []
        node_h9 = []
        for n in self.node_list:
            lat = self.G.nodes[n]["y"]
            lon = self.G.nodes[n]["x"]
            c8 = h3.latlng_to_cell(lat, lon, 8)
            c9 = h3.latlng_to_cell(lat, lon, 9)
            h8_counts[c8] = h8_counts.get(c8, 0) + 1
            h9_counts[c9] = h9_counts.get(c9, 0) + 1
            node_h8.append(c8)
            node_h9.append(c9)
            
        # Construct static node feature matrix [N, D_static]
        static_feats = []
        for i, n in enumerate(self.node_list):
            deg = self.G.degree(n)
            in_deg = self.G.in_degree(n)
            out_deg = self.G.out_degree(n)
            sc = int(self.G.nodes[n].get("street_count", deg))
            
            static_feats.append([
                self.node_x_km[i],
                self.node_y_km[i],
                deg / 10.0,
                in_deg / 10.0,
                out_deg / 10.0,
                sc / 10.0,
                h8_counts[node_h8[i]] / 50.0,
                h9_counts[node_h9[i]] / 20.0,
            ])
            
        self.static_node_features = np.array(static_feats, dtype=np.float32)


def build_scenario_pyg_data(
    city_data: CityGraphData,
    scenario_meta: Dict[str, Any],
    edge_mask: np.ndarray,
    scenario_samples: pd.DataFrame,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Construct PyG graph data for a specific disruption scenario.
    
    Filters physical road corridors to operational corridors (mask == 1).
    Augments node features with localized hazard proximity.
    """
    # 1. Operational edge index (DEC-013)
    # edge_mask: 1 = operational, 0 = disrupted
    op_edge_mask_torch = torch.from_numpy(edge_mask == 1).to(torch.bool)
    op_edge_index = city_data.canonical_edge_tensor[:, op_edge_mask_torch].to(device)
    
    # 2. Dynamic hazard proximity per node
    center_lon = scenario_meta["disruption_center"]["x"]
    center_lat = scenario_meta["disruption_center"]["y"]
    radius_km = scenario_meta["disruption_radius_m"] / 1000.0
    
    epi_x_km = (center_lon - city_data.center_lon) * city_data.m_per_deg_lon / 1000.0
    epi_y_km = (center_lat - city_data.center_lat) * city_data.m_per_deg_lat / 1000.0
    
    dist_to_epi_km = np.sqrt(
        (city_data.node_x_km - epi_x_km) ** 2 + (city_data.node_y_km - epi_y_km) ** 2
    )
    in_hazard_zone = (dist_to_epi_km <= radius_km).astype(np.float32)
    
    # Count disrupted incident edges per node
    disrupted_edge_indices = np.where(edge_mask == 0)[0]
    disrupted_node_counts = np.zeros(city_data.num_nodes, dtype=np.float32)
    for e_idx in disrupted_edge_indices:
        u_idx, v_idx = city_data.canonical_edge_indices[e_idx]
        disrupted_node_counts[u_idx] += 1.0
        disrupted_node_counts[v_idx] += 1.0
    disrupted_node_counts_norm = disrupted_node_counts / 10.0
    
    # Facility indicators from scenario queries
    is_fac = np.zeros(city_data.num_nodes, dtype=np.float32)
    is_hosp = np.zeros(city_data.num_nodes, dtype=np.float32)
    for _, row in scenario_samples.iterrows():
        d_idx = city_data.node_to_idx.get(row["destination_node"])
        if d_idx is not None:
            is_fac[d_idx] = 1.0
            if row.get("facility_amenity") == "hospital":
                is_hosp[d_idx] = 1.0
                
    # Combine full node feature matrix [N, D_node]
    dynamic_feats = np.column_stack([
        city_data.static_node_features,
        is_fac,
        is_hosp,
        dist_to_epi_km / 5.0,  # normalized by 5km typical city radius
        in_hazard_zone,
        disrupted_node_counts_norm,
    ])
    x_nodes = torch.tensor(dynamic_feats, dtype=torch.float32, device=device)
    
    # 3. Query OD pairs and permitted pre-disruption routing attributes
    origin_indices = [city_data.node_to_idx[n] for n in scenario_samples["origin_node"]]
    dest_indices = [city_data.node_to_idx[n] for n in scenario_samples["destination_node"]]
    
    # Pairwise features (DEC-015 permitted)
    orig_x = scenario_samples["origin_x"].values
    orig_y = scenario_samples["origin_y"].values
    dest_x = scenario_samples["destination_x"].values
    dest_y = scenario_samples["destination_y"].values
    
    # Metric Euclidean distance
    dx_m = (dest_x - orig_x) * city_data.m_per_deg_lon
    dy_m = (dest_y - orig_y) * city_data.m_per_deg_lat
    euclid_dist_km = np.sqrt(dx_m ** 2 + dy_m ** 2) / 1000.0
    
    orig_dist_km = scenario_samples["original_distance_m"].values / 1000.0
    orig_hops_norm = scenario_samples["orig_path_hops"].values / 20.0
    circuity = orig_dist_km / np.maximum(euclid_dist_km, 0.001)
    snap_dist_km = scenario_samples["facility_snap_distance_m"].values / 1000.0
    
    # Hazard distances for origin and destination
    dist_o_epi_km = np.sqrt(
        ((orig_x - center_lon) * city_data.m_per_deg_lon) ** 2 +
        ((orig_y - center_lat) * city_data.m_per_deg_lat) ** 2
    ) / 1000.0
    dist_d_epi_km = np.sqrt(
        ((dest_x - center_lon) * city_data.m_per_deg_lon) ** 2 +
        ((dest_y - center_lat) * city_data.m_per_deg_lat) ** 2
    ) / 1000.0
    min_dist_od_epi_km = np.minimum(dist_o_epi_km, dist_d_epi_km)
    
    pair_features_np = np.column_stack([
        euclid_dist_km,
        orig_dist_km,
        orig_hops_norm,
        circuity,
        snap_dist_km,
        dist_o_epi_km,
        dist_d_epi_km,
        min_dist_od_epi_km,
        radius_km * np.ones_like(euclid_dist_km),
    ])
    
    pair_features_torch = torch.tensor(pair_features_np, dtype=torch.float32, device=device)
    
    # 4. Targets
    y_reach = torch.tensor(scenario_samples["reachable"].values, dtype=torch.float32, device=device)
    
    # Detour target: fill NaN with 0.0, use mask for loss
    detour_vals = scenario_samples["relative_detour"].fillna(0.0).values
    y_detour = torch.tensor(detour_vals, dtype=torch.float32, device=device)
    detour_mask = torch.tensor(
        (scenario_samples["reachable"] == 1).values, dtype=torch.bool, device=device
    )
    
    return {
        "scenario_id": scenario_meta["scenario_id"],
        "city": scenario_meta["city"],
        "split": scenario_meta["split"],
        "x": x_nodes,
        "edge_index": op_edge_index,
        "origin_idx": torch.tensor(origin_indices, dtype=torch.long, device=device),
        "dest_idx": torch.tensor(dest_indices, dtype=torch.long, device=device),
        "pair_features": pair_features_torch,
        "y_reach": y_reach,
        "y_detour": y_detour,
        "detour_mask": detour_mask,
        "sample_count": len(scenario_samples),
    }


def load_all_pyg_scenarios(
    benchmark_samples_path: str = "data/processed/benchmark_samples.parquet",
    scenarios_metadata_path: str = "data/processed/benchmark_scenarios_metadata.json",
    edge_masks_path: str = "data/processed/benchmark_edge_masks.npz",
    raw_dir: str = "data/raw",
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[Dict[str, Any]]]:
    """Load and construct all PyG scenario graph objects partitioned by split."""
    samples_p = Path(benchmark_samples_path)
    if samples_p.suffix == ".parquet":
        df = pd.read_parquet(samples_p)
    else:
        df = pd.read_json(samples_p)
        
    with open(scenarios_metadata_path) as fp:
        meta_list = json.load(fp)
    meta_by_id = {m["scenario_id"]: m for m in meta_list}
    
    edge_masks_npz = np.load(edge_masks_path)
    
    # Initialize city graph structures
    city_cache = {}
    for city_name in df["city"].unique():
        city_cache[city_name] = CityGraphData(city_name, raw_dir=raw_dir)
        
    split_scenarios: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "transfer": [],
    }
    
    for sid, group in df.groupby("scenario_id"):
        sm = meta_by_id[sid]
        cname = sm["city"]
        split = sm["split"]
        mask = edge_masks_npz[sid]
        
        sc_data = build_scenario_pyg_data(
            city_data=city_cache[cname],
            scenario_meta=sm,
            edge_mask=mask,
            scenario_samples=group,
            device=device,
        )
        split_scenarios[split].append(sc_data)
        
    return split_scenarios
