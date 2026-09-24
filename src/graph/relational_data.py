"""Relational Graph Data and Scenario Graph Builder for E03 (Relational GNN).

Implements the multi-relational urban schema:
  - Entity types:
      1. Road Intersections (V_junction)
      2. Critical Facilities (V_facility: Hospitals and Clinics)
  - Relation types:
      - Relation 0: connects_to (road corridor between physical junctions)
      - Relation 1: accessible_from (junction -> facility access corridor)
      - Relation 2: serves (facility -> junction outbound access)
  - Disruption:
      - Road corridors filtered per scenario operational mask (DEC-013).
      - Zero post-disruption target or test-city leakage (DEC-015).
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import h3

from src.data.osm_loader import load_city_data
from src.graph.physical_graph import build_physical_digraph


RELATION_CONNECTS_TO = 0
RELATION_ACCESSIBLE_FROM = 1
RELATION_SERVES = 2
NUM_RELATIONS = 3


class CityRelationalGraphData:
    """Holds multi-relational graph topology and precomputed node attributes for a city."""
    
    def __init__(
        self,
        city: str,
        benchmark_samples_df: pd.DataFrame,
        raw_dir: str = "data/raw",
    ):
        self.city = city
        self.raw_dir = raw_dir
        
        # 1. Load raw data and build canonical DiGraph
        b_gdf, G_multi, fac_gdf = load_city_data(city, raw_dir=raw_dir)
        self.G = build_physical_digraph(G_multi)
        
        # 2. Junction node ordering
        self.junction_nodes = sorted(list(self.G.nodes()))
        self.num_junctions = len(self.junction_nodes)
        self.junction_to_idx = {n: i for i, n in enumerate(self.junction_nodes)}
        
        # 3. City centroid for invariant local metric projection
        all_lons = np.array([self.G.nodes[n]["x"] for n in self.junction_nodes])
        all_lats = np.array([self.G.nodes[n]["y"] for n in self.junction_nodes])
        self.center_lon = float(all_lons.mean())
        self.center_lat = float(all_lats.mean())
        
        lat_rad = math.radians(self.center_lat)
        self.m_per_deg_lat = 111000.0
        self.m_per_deg_lon = 111000.0 * math.cos(lat_rad)
        
        # Junction coordinates in local km
        self.junction_x_km = (all_lons - self.center_lon) * self.m_per_deg_lon / 1000.0
        self.junction_y_km = (all_lats - self.center_lat) * self.m_per_deg_lat / 1000.0
        
        # 4. Canonical road corridor edges (Relation 0: connects_to)
        self.canonical_road_edges = sorted(list(self.G.edges()))
        self.num_road_edges = len(self.canonical_road_edges)
        self.road_edge_indices = [
            (self.junction_to_idx[u], self.junction_to_idx[v])
            for u, v in self.canonical_road_edges
        ]
        self.road_edge_tensor = torch.tensor(
            self.road_edge_indices, dtype=torch.long
        ).t().contiguous()
        
        # 5. Facility entity nodes (Heterogeneous nodes appended after junctions)
        # Extract unique facilities evaluated for this city
        city_samples = benchmark_samples_df[benchmark_samples_df["city"] == city]
        fac_group = city_samples.groupby("destination_facility_id").first().reset_index()
        
        self.facility_ids = sorted(list(fac_group["destination_facility_id"].unique()))
        self.num_facilities = len(self.facility_ids)
        self.facility_to_idx = {
            fid: self.num_junctions + i for i, fid in enumerate(self.facility_ids)
        }
        self.total_nodes = self.num_junctions + self.num_facilities
        
        # Map facility attributes and snapped junctions
        self.facility_x_km = []
        self.facility_y_km = []
        self.facility_is_hospital = []
        self.facility_access_edges_forward = []  # junction -> facility (accessible_from)
        self.facility_access_edges_reverse = []  # facility -> junction (serves)
        
        fac_h8 = []
        fac_h9 = []
        
        for fid in self.facility_ids:
            frow = fac_group[fac_group["destination_facility_id"] == fid].iloc[0]
            fx = frow["destination_x"]
            fy = frow["destination_y"]
            fx_km = (fx - self.center_lon) * self.m_per_deg_lon / 1000.0
            fy_km = (fy - self.center_lat) * self.m_per_deg_lat / 1000.0
            self.facility_x_km.append(fx_km)
            self.facility_y_km.append(fy_km)
            
            is_hosp = 1.0 if frow.get("facility_amenity") == "hospital" else 0.0
            self.facility_is_hospital.append(is_hosp)
            
            snapped_jnode = frow["destination_node"]
            j_idx = self.junction_to_idx[snapped_jnode]
            f_idx = self.facility_to_idx[fid]
            
            self.facility_access_edges_forward.append((j_idx, f_idx))
            self.facility_access_edges_reverse.append((f_idx, j_idx))
            
            c8 = h3.latlng_to_cell(fy, fx, 8)
            c9 = h3.latlng_to_cell(fy, fx, 9)
            fac_h8.append(c8)
            fac_h9.append(c9)
            
        self.facility_x_km = np.array(self.facility_x_km, dtype=np.float32)
        self.facility_y_km = np.array(self.facility_y_km, dtype=np.float32)
        self.facility_is_hospital = np.array(self.facility_is_hospital, dtype=np.float32)
        
        # Tensors for facility access relations
        self.access_edges_fwd_tensor = torch.tensor(
            self.facility_access_edges_forward, dtype=torch.long
        ).t().contiguous()
        self.access_edges_rev_tensor = torch.tensor(
            self.facility_access_edges_reverse, dtype=torch.long
        ).t().contiguous()
        
        # Combined coordinates for all nodes [total_nodes]
        self.all_node_x_km = np.concatenate([self.junction_x_km, self.facility_x_km])
        self.all_node_y_km = np.concatenate([self.junction_y_km, self.facility_y_km])
        
        # H3 density for junctions
        h8_counts = {}
        h9_counts = {}
        j_h8 = []
        j_h9 = []
        for n in self.junction_nodes:
            lat = self.G.nodes[n]["y"]
            lon = self.G.nodes[n]["x"]
            c8 = h3.latlng_to_cell(lat, lon, 8)
            c9 = h3.latlng_to_cell(lat, lon, 9)
            h8_counts[c8] = h8_counts.get(c8, 0) + 1
            h9_counts[c9] = h9_counts.get(c9, 0) + 1
            j_h8.append(c8)
            j_h9.append(c9)
            
        # Static node feature matrix [total_nodes, D_static]
        # Features:
        # [x_km, y_km, deg_norm, in_deg_norm, out_deg_norm, sc_norm, h8_norm, h9_norm, is_facility, is_hospital]
        static_feats = []
        for i, n in enumerate(self.junction_nodes):
            deg = self.G.degree(n)
            in_deg = self.G.in_degree(n)
            out_deg = self.G.out_degree(n)
            sc = int(self.G.nodes[n].get("street_count", deg))
            static_feats.append([
                self.junction_x_km[i],
                self.junction_y_km[i],
                deg / 10.0,
                in_deg / 10.0,
                out_deg / 10.0,
                sc / 10.0,
                h8_counts.get(j_h8[i], 0) / 50.0,
                h9_counts.get(j_h9[i], 0) / 20.0,
                0.0,  # is_facility = 0
                0.0,  # is_hospital = 0
            ])
            
        for i in range(self.num_facilities):
            static_feats.append([
                self.facility_x_km[i],
                self.facility_y_km[i],
                1.0 / 10.0,  # degree (access edge)
                1.0 / 10.0,
                1.0 / 10.0,
                1.0 / 10.0,
                h8_counts.get(fac_h8[i], 0) / 50.0,
                h9_counts.get(fac_h9[i], 0) / 20.0,
                1.0,  # is_facility = 1
                self.facility_is_hospital[i],
            ])
            
        self.static_node_features = np.array(static_feats, dtype=np.float32)


def build_scenario_relational_data(
    city_data: CityRelationalGraphData,
    scenario_meta: Dict[str, Any],
    edge_mask: np.ndarray,
    scenario_samples: pd.DataFrame,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Construct multi-relational graph data for a specific disruption scenario."""
    # 1. Operational connects_to road edges (Relation 0)
    op_mask_torch = torch.from_numpy(edge_mask == 1).to(torch.bool)
    op_road_edge_index = city_data.road_edge_tensor[:, op_mask_torch].to(device)
    num_op_roads = op_road_edge_index.shape[1]
    
    # 2. Facility access edges: accessible_from (Rel 1) and serves (Rel 2)
    acc_fwd_index = city_data.access_edges_fwd_tensor.to(device)
    acc_rev_index = city_data.access_edges_rev_tensor.to(device)
    num_fwd = acc_fwd_index.shape[1]
    num_rev = acc_rev_index.shape[1]
    
    # Concatenate all operational relational edges
    all_edges = torch.cat([op_road_edge_index, acc_fwd_index, acc_rev_index], dim=1)
    
    # Construct edge_type tensor
    road_types = torch.full((num_op_roads,), RELATION_CONNECTS_TO, dtype=torch.long, device=device)
    fwd_types = torch.full((num_fwd,), RELATION_ACCESSIBLE_FROM, dtype=torch.long, device=device)
    rev_types = torch.full((num_rev,), RELATION_SERVES, dtype=torch.long, device=device)
    all_edge_types = torch.cat([road_types, fwd_types, rev_types], dim=0)
    
    # 3. Dynamic hazard proximity per node (for both junctions and facilities)
    center_lon = scenario_meta["disruption_center"]["x"]
    center_lat = scenario_meta["disruption_center"]["y"]
    radius_km = scenario_meta["disruption_radius_m"] / 1000.0
    
    epi_x_km = (center_lon - city_data.center_lon) * city_data.m_per_deg_lon / 1000.0
    epi_y_km = (center_lat - city_data.center_lat) * city_data.m_per_deg_lat / 1000.0
    
    dist_to_epi_km = np.sqrt(
        (city_data.all_node_x_km - epi_x_km) ** 2 + (city_data.all_node_y_km - epi_y_km) ** 2
    )
    in_hazard_zone = (dist_to_epi_km <= radius_km).astype(np.float32)
    
    # Count disrupted incident road edges for junctions
    disrupted_edge_indices = np.where(edge_mask == 0)[0]
    disrupted_node_counts = np.zeros(city_data.total_nodes, dtype=np.float32)
    for e_idx in disrupted_edge_indices:
        u_idx, v_idx = city_data.road_edge_indices[e_idx]
        disrupted_node_counts[u_idx] += 1.0
        disrupted_node_counts[v_idx] += 1.0
    disrupted_node_counts_norm = disrupted_node_counts / 10.0
    
    # Combine full node feature matrix [total_nodes, D_node]
    dynamic_feats = np.column_stack([
        city_data.static_node_features,
        dist_to_epi_km / 5.0,  # normalized by 5km
        in_hazard_zone,
        disrupted_node_counts_norm,
    ])
    x_nodes = torch.tensor(dynamic_feats, dtype=torch.float32, device=device)
    
    # 4. Query OD pairs: Origin junction -> Destination facility node
    origin_indices = [
        city_data.junction_to_idx[n] for n in scenario_samples["origin_node"]
    ]
    dest_facility_indices = [
        city_data.facility_to_idx[fid] for fid in scenario_samples["destination_facility_id"]
    ]
    
    # Pairwise features (DEC-015 permitted)
    orig_x = scenario_samples["origin_x"].values
    orig_y = scenario_samples["origin_y"].values
    dest_x = scenario_samples["destination_x"].values
    dest_y = scenario_samples["destination_y"].values
    
    dx_m = (dest_x - orig_x) * city_data.m_per_deg_lon
    dy_m = (dest_y - orig_y) * city_data.m_per_deg_lat
    euclid_dist_km = np.sqrt(dx_m ** 2 + dy_m ** 2) / 1000.0
    
    orig_dist_km = scenario_samples["original_distance_m"].values / 1000.0
    orig_hops_norm = scenario_samples["orig_path_hops"].values / 20.0
    circuity = orig_dist_km / np.maximum(euclid_dist_km, 0.001)
    snap_dist_km = scenario_samples["facility_snap_distance_m"].values / 1000.0
    
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
    y_reach = torch.tensor(scenario_samples["reachable"].values, dtype=torch.float32, device=device)
    
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
        "edge_index": all_edges,
        "edge_type": all_edge_types,
        "origin_idx": torch.tensor(origin_indices, dtype=torch.long, device=device),
        "dest_idx": torch.tensor(dest_facility_indices, dtype=torch.long, device=device),
        "pair_features": pair_features_torch,
        "y_reach": y_reach,
        "y_detour": y_detour,
        "detour_mask": detour_mask,
        "sample_count": len(scenario_samples),
    }


def load_all_relational_scenarios(
    benchmark_samples_path: str = "data/processed/benchmark_samples.parquet",
    scenarios_metadata_path: str = "data/processed/benchmark_scenarios_metadata.json",
    edge_masks_path: str = "data/processed/benchmark_edge_masks.npz",
    raw_dir: str = "data/raw",
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[Dict[str, Any]]]:
    """Load and construct multi-relational PyG scenario graph objects partitioned by split."""
    samples_p = Path(benchmark_samples_path)
    if samples_p.suffix == ".parquet":
        df = pd.read_parquet(samples_p)
    else:
        df = pd.read_json(samples_p)
        
    with open(scenarios_metadata_path) as fp:
        meta_list = json.load(fp)
    meta_by_id = {m["scenario_id"]: m for m in meta_list}
    
    edge_masks_npz = np.load(edge_masks_path)
    
    # Initialize city relational graph caches
    city_cache = {}
    for city_name in df["city"].unique():
        city_cache[city_name] = CityRelationalGraphData(
            city=city_name,
            benchmark_samples_df=df,
            raw_dir=raw_dir,
        )
        
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
        
        sc_data = build_scenario_relational_data(
            city_data=city_cache[cname],
            scenario_meta=sm,
            edge_mask=mask,
            scenario_samples=group,
            device=device,
        )
        split_scenarios[split].append(sc_data)
        
    return split_scenarios
