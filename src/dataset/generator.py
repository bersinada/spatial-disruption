import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
"""Deterministic Benchmark Dataset Generator.

Generates counterfactual urban accessibility evaluation samples across:
  - City A: Seattle (Train, Val, In-City Test)
  - City B: Portland (Zero-Shot Transfer Test)
  - Secondary Experiment: Seattle Buffered Spatial Split (South/Central vs North)

Every scenario contains exactly 100 samples:
  - 50 Active Core (intact shortest path traverses at least one disrupted canonical edge)
  - 50 Control Context (intact shortest path does not traverse any disrupted edge, stratified by distance tiers)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.data.osm_loader import load_city_data
from src.graph.physical_graph import build_physical_digraph, map_facilities_to_network
from src.disruption.generator import generate_localized_disruption
from src.ground_truth.reachability import compute_ground_truth_sample


def get_scenario_definitions(mode: str = "pilot") -> List[Dict[str, Any]]:
    """Define deterministic scenario configurations.
    
    Pilot: 5 scenarios (2 Seattle train, 1 Seattle val, 1 Seattle test, 1 Portland transfer).
    Full: 40 scenarios (20 Seattle train, 5 Seattle val, 5 Seattle test, 10 Portland transfer).
    """
    scenarios = []
    
    if mode == "pilot":
        scenarios.extend([
            {"city": "seattle", "split": "train", "scenario_id": "seattle_s01", "seed": 101, "radius_m": 150.0},
            {"city": "seattle", "split": "train", "scenario_id": "seattle_s02", "seed": 102, "radius_m": 200.0},
            {"city": "seattle", "split": "val", "scenario_id": "seattle_s21", "seed": 121, "radius_m": 150.0},
            {"city": "seattle", "split": "test", "scenario_id": "seattle_s26", "seed": 126, "radius_m": 175.0},
            {"city": "portland", "split": "transfer", "scenario_id": "portland_s01", "seed": 201, "radius_m": 150.0},
        ])
    else:
        # Full benchmark: 30 Seattle + 10 Portland = 40 scenarios
        # Seattle Train (s01 to s20)
        radii_train = [150.0, 200.0, 125.0, 175.0, 250.0] * 4
        for i in range(1, 21):
            scenarios.append({
                "city": "seattle",
                "split": "train",
                "scenario_id": f"seattle_s{i:02d}",
                "seed": 100 + i,
                "radius_m": radii_train[i - 1],
            })
            
        # Seattle Val (s21 to s25)
        radii_val = [150.0, 200.0, 175.0, 125.0, 225.0]
        for i in range(21, 26):
            scenarios.append({
                "city": "seattle",
                "split": "val",
                "scenario_id": f"seattle_s{i:02d}",
                "seed": 100 + i,
                "radius_m": radii_val[i - 21],
            })
            
        # Seattle In-City Test (s26 to s30)
        radii_test = [150.0, 175.0, 200.0, 250.0, 125.0]
        for i in range(26, 31):
            scenarios.append({
                "city": "seattle",
                "split": "test",
                "scenario_id": f"seattle_s{i:02d}",
                "seed": 100 + i,
                "radius_m": radii_test[i - 26],
            })
            
        # Portland Transfer Test (s01 to s10)
        radii_transfer = [150.0, 200.0, 125.0, 175.0, 250.0, 150.0, 200.0, 175.0, 125.0, 225.0]
        for i in range(1, 11):
            scenarios.append({
                "city": "portland",
                "split": "transfer",
                "scenario_id": f"portland_s{i:02d}",
                "seed": 200 + i,
                "radius_m": radii_transfer[i - 1],
            })
            
    return scenarios


def generate_scenario_samples(
    G: nx.DiGraph,
    boundary_gdf: gpd.GeoDataFrame,
    mapped_facilities: List[Dict[str, Any]],
    scenario_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate exactly 100 samples (50 active, 50 control) for a single disruption scenario."""
    city = scenario_cfg["city"]
    split = scenario_cfg["split"]
    scenario_id = scenario_cfg["scenario_id"]
    seed = scenario_cfg["seed"]
    radius_m = scenario_cfg["radius_m"]
    
    poly = boundary_gdf.geometry.iloc[0]
    
    # Generate deterministic disruption
    G_disrupt, scenario_meta, edge_mask = generate_localized_disruption(
        G=G,
        scenario_id=scenario_id,
        city=city,
        split=split,
        seed=seed,
        boundary_gdf=boundary_gdf,
        radius_meters=radius_m,
    )
    
    disrupted_edge_set = set((u, v) for u, v in scenario_meta["disrupted_edge_ids"])
    
    # Filter valid destinations (mapped, inside boundary, in G)
    valid_facs = [
        f for f in mapped_facilities 
        if f["is_mapped"] and f["snapped_node"] in G
    ]
    
    # Filter valid origins (inside boundary, in largest SCC)
    scc = max(nx.strongly_connected_components(G), key=len)
    valid_origins = [
        n for n in scc 
        if poly.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"]))
    ]
    
    rng = np.random.RandomState(seed)
    shuffled_origins = list(valid_origins)
    rng.shuffle(shuffled_origins)
    
    active_candidates = []
    control_candidates = []
    seen_od_nodes = set()
    
    # Search for candidate pairs
    for u in shuffled_origins:
        # Stop early when we have enough candidates in both pools
        if len(active_candidates) >= 150 and len(control_candidates) >= 250:
            break
            
        for fac in valid_facs:
            d = fac["snapped_node"]
            if u == d:
                continue
            if not nx.has_path(G, u, d):
                continue
                
            path = nx.shortest_path(G, u, d, weight="length")
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            is_active = any(e in disrupted_edge_set for e in path_edges)
            
            if (u, d) in seen_od_nodes:
                continue
            seen_od_nodes.add((u, d))
            
            orig_dist = float(nx.shortest_path_length(G, u, d, weight="length"))
            item = (u, fac, orig_dist)
            
            if is_active:
                active_candidates.append(item)
            else:
                control_candidates.append(item)
                
    # If active candidates < 50, expand search across remaining origins
    if len(active_candidates) < 50:
        for u in valid_origins:
            if len(active_candidates) >= 50:
                break
            for fac in valid_facs:
                d = fac["snapped_node"]
                if u == d or not nx.has_path(G, u, d):
                    continue
                path = nx.shortest_path(G, u, d, weight="length")
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                if any(e in disrupted_edge_set for e in path_edges):
                    item = (u, fac, float(nx.shortest_path_length(G, u, d, weight="length")))
                    if item not in active_candidates:
                        active_candidates.append(item)
                        
    if len(active_candidates) < 50:
        raise RuntimeError(f"Scenario {scenario_id} could only find {len(active_candidates)} active pairs (needs 50).")
        
    # Sample exactly 50 Active Core pairs
    active_indices = rng.choice(len(active_candidates), size=50, replace=False)
    active_selected = [active_candidates[i] for i in active_indices]
    
    # Stratify Control Context pairs by distance terciles
    control_dists = np.array([c[2] for c in control_candidates])
    q33 = np.percentile(control_dists, 33.33)
    q66 = np.percentile(control_dists, 66.67)
    
    idx_short = [i for i, c in enumerate(control_candidates) if c[2] < q33]
    idx_med = [i for i, c in enumerate(control_candidates) if q33 <= c[2] < q66]
    idx_long = [i for i, c in enumerate(control_candidates) if c[2] >= q66]
    
    n_short = 17
    n_med = 17
    n_long = 16
    
    sel_short = rng.choice(idx_short, size=n_short, replace=False)
    sel_med = rng.choice(idx_med, size=n_med, replace=False)
    sel_long = rng.choice(idx_long, size=n_long, replace=False)
    
    control_selected = (
        [(control_candidates[i], "short") for i in sel_short] +
        [(control_candidates[i], "medium") for i in sel_med] +
        [(control_candidates[i], "long") for i in sel_long]
    )
    
    # Compute exact ground truth for all 100 samples
    samples = []
    
    # Active Core
    for u, fac, orig_dist in active_selected:
        gt = compute_ground_truth_sample(G, G_disrupt, u, fac["snapped_node"])
        
        # Determine distance tier
        tier = "short" if orig_dist < q33 else ("medium" if orig_dist < q66 else "long")
        
        sample_rec = {
            "sample_id": f"{scenario_id}_o{u}_d{fac['snapped_node']}",
            "city": city,
            "scenario_id": scenario_id,
            "split": split,
            "seed": seed,
            "origin_node": u,
            "origin_x": G.nodes[u]["x"],
            "origin_y": G.nodes[u]["y"],
            "destination_node": fac["snapped_node"],
            "destination_x": G.nodes[fac["snapped_node"]]["x"],
            "destination_y": G.nodes[fac["snapped_node"]]["y"],
            "origin_facility_id": None,
            "destination_facility_id": fac["facility_id"],
            "facility_name": fac["name"],
            "facility_amenity": fac["amenity"],
            "facility_snap_distance_m": fac["snap_distance_m"],
            "disrupted_edge_ids": scenario_meta["disrupted_edge_ids"],
            "original_distance_m": gt["original_distance_m"],
            "disrupted_distance_m": gt["disrupted_distance_m"],
            "reachable": gt["reachable"],
            "relative_detour": gt["relative_detour"],
            "sample_type": "active_core",
            "distance_tier": tier,
            "orig_path_hops": gt["orig_path_hops"],
            "disrupt_path_hops": gt["disrupt_path_hops"],
        }
        samples.append(sample_rec)
        
    # Control Context
    for (u, fac, orig_dist), tier in control_selected:
        gt = compute_ground_truth_sample(G, G_disrupt, u, fac["snapped_node"])
        
        sample_rec = {
            "sample_id": f"{scenario_id}_o{u}_d{fac['snapped_node']}",
            "city": city,
            "scenario_id": scenario_id,
            "split": split,
            "seed": seed,
            "origin_node": u,
            "origin_x": G.nodes[u]["x"],
            "origin_y": G.nodes[u]["y"],
            "destination_node": fac["snapped_node"],
            "destination_x": G.nodes[fac["snapped_node"]]["x"],
            "destination_y": G.nodes[fac["snapped_node"]]["y"],
            "origin_facility_id": None,
            "destination_facility_id": fac["facility_id"],
            "facility_name": fac["name"],
            "facility_amenity": fac["amenity"],
            "facility_snap_distance_m": fac["snap_distance_m"],
            "disrupted_edge_ids": scenario_meta["disrupted_edge_ids"],
            "original_distance_m": gt["original_distance_m"],
            "disrupted_distance_m": gt["disrupted_distance_m"],
            "reachable": gt["reachable"],
            "relative_detour": gt["relative_detour"],
            "sample_type": "control_context",
            "distance_tier": tier,
            "orig_path_hops": gt["orig_path_hops"],
            "disrupt_path_hops": gt["disrupt_path_hops"],
        }
        samples.append(sample_rec)
        
    return samples, scenario_meta


def generate_benchmark_dataset(
    mode: str = "pilot",
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute complete dataset generation pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    scenarios_cfg = get_scenario_definitions(mode=mode)
    print(f"Starting {mode.upper()} dataset generation across {len(scenarios_cfg)} scenarios...")
    
    # Load and prepare cities
    cities_data = {}
    for city in ["seattle", "portland"]:
        print(f"Loading cached data for {city}...")
        b_gdf, G_multi, fac_gdf = load_city_data(city)
        G = build_physical_digraph(G_multi)
        mapped_facs, stats = map_facilities_to_network(G, fac_gdf)
        cities_data[city] = {
            "boundary": b_gdf,
            "graph": G,
            "facilities": mapped_facs,
        }
        print(f"[{city}] Canonical DiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; Mapped Facs: {stats['mapped_facilities']}")
        
    all_samples = []
    all_scenario_metadata = []
    
    for i, scfg in enumerate(scenarios_cfg, 1):
        city = scfg["city"]
        sid = scfg["scenario_id"]
        cdata = cities_data[city]
        
        print(f"[{i}/{len(scenarios_cfg)}] Generating {sid} ({city}, {scfg['split']})...")
        samples, meta = generate_scenario_samples(
            G=cdata["graph"],
            boundary_gdf=cdata["boundary"],
            mapped_facilities=cdata["facilities"],
            scenario_cfg=scfg,
        )
        
        all_samples.extend(samples)
        all_scenario_metadata.append(meta)
        
    df_samples = pd.DataFrame(all_samples)
    
    # Persist artifacts
    file_prefix = "pilot" if mode == "pilot" else "benchmark"
    parquet_path = out_path / f"{file_prefix}_samples.parquet"
    json_path = out_path / f"{file_prefix}_samples.json"
    meta_path = out_path / f"{file_prefix}_scenarios_metadata.json"
    
    # Save Parquet (disrupted_edge_ids stored as JSON string for parquet compatibility)
    df_parquet = df_samples.copy()
    df_parquet["disrupted_edge_ids"] = df_parquet["disrupted_edge_ids"].apply(json.dumps)
    df_parquet.to_parquet(parquet_path, index=False)
    
    # Save JSON
    with open(json_path, "w") as fp:
        json.dump(all_samples, fp, indent=2)
        
    # Save Scenario Metadata
    with open(meta_path, "w") as fp:
        json.dump(all_scenario_metadata, fp, indent=2)
        
    # Save Canonical Edge Orders for both cities (DEC-013)
    for city_name in ["seattle", "portland"]:
        c_edges_path = out_path / f"{city_name}_canonical_edges.json"
        canonical_edges = sorted(list(cities_data[city_name]["graph"].edges()))
        with open(c_edges_path, "w") as fp:
            json.dump([[int(u), int(v)] for u, v in canonical_edges], fp)
            
    # Save Edge Masks NPZ (DEC-013)
    masks_dict = {
        m["scenario_id"]: np.array(list(m["edge_disruption_mask_binary"]), dtype=np.int8)
        for m in all_scenario_metadata
    }
    npz_path = out_path / f"{file_prefix}_edge_masks.npz"
    np.savez_compressed(npz_path, **masks_dict)
        
    print(f"\nSuccessfully generated {len(df_samples)} samples across {len(all_scenario_metadata)} scenarios.")
    print(f"Persisted to:\n  - {parquet_path}\n  - {json_path}\n  - {meta_path}\n  - {npz_path}")
    
    return df_samples, all_scenario_metadata


def generate_secondary_spatial_split(
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate the secondary buffered geographic split in Seattle.
    
    Region 1: South/Central Seattle (latitude < 47.65 - 0.0067)
    Region 2: North Seattle (latitude > 47.65 + 0.0067)
    Exclusion Buffer: 1,500 m along the Ship Canal corridor.
    """
    print("\nGenerating secondary buffered geographic split for Seattle...")
    b_gdf, G_multi, fac_gdf = load_city_data("seattle")
    G = build_physical_digraph(G_multi)
    mapped_facs, _ = map_facilities_to_network(G, fac_gdf)
    
    # Ship canal latitude ~ 47.65. Buffer = 1,500 m (~0.0135 deg lat, +-0.0067 deg)
    canal_lat = 47.650
    half_buffer_deg = 0.0067  # ~750m north and south of canal
    
    south_cutoff = canal_lat - half_buffer_deg
    north_cutoff = canal_lat + half_buffer_deg
    
    # Partition origins
    poly = b_gdf.geometry.iloc[0]
    scc = max(nx.strongly_connected_components(G), key=len)
    
    region1_nodes = [
        n for n in scc 
        if poly.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"])) and G.nodes[n]["y"] <= south_cutoff
    ]
    region2_nodes = [
        n for n in scc 
        if poly.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"])) and G.nodes[n]["y"] >= north_cutoff
    ]
    excluded_nodes = [
        n for n in scc 
        if poly.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"])) and south_cutoff < G.nodes[n]["y"] < north_cutoff
    ]
    
    print(f"Secondary Spatial Partitioning:")
    print(f"  Region 1 (South/Central - Train): {len(region1_nodes)} nodes")
    print(f"  Region 2 (North - Spatial Test) : {len(region2_nodes)} nodes")
    print(f"  Exclusion Buffer (Ship Canal)   : {len(excluded_nodes)} nodes")
    
    metadata = {
        "experiment": "E06_secondary_spatial_split",
        "city": "seattle",
        "canal_latitude": canal_lat,
        "buffer_meters": 1500.0,
        "region1_train_nodes": len(region1_nodes),
        "region2_test_nodes": len(region2_nodes),
        "excluded_buffer_nodes": len(excluded_nodes),
        "notes": "Origins in the 1,500m exclusion buffer are completely excluded from both train and test.",
    }
    
    meta_path = Path(output_dir) / "seattle_spatial_split_metadata.json"
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Saved spatial split metadata to {meta_path}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Benchmark Dataset Generator")
    parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="pilot", help="Generation mode (pilot or full)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--with-spatial-split", action="store_true", help="Also generate secondary spatial split metadata")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed (default: scenario-locked seeds)")
    args = parser.parse_args()
    
    df, meta = generate_benchmark_dataset(mode=args.mode, output_dir=args.output_dir)
    if args.with_spatial_split:
        generate_secondary_spatial_split(output_dir=args.output_dir)
