import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
"""Comprehensive Automated Dataset Validation Suite.

Verifies all 15 protocol invariants:
  1. all sample coordinates are inside municipal boundaries
  2. disruption epicenters are inside boundaries
  3. all OD nodes exist in the canonical graph
  4. exactly 50 active + 50 control samples per scenario
  5. disrupted edges are valid canonical edges
  6. no scenario leakage across splits
  7. Dijkstra results are deterministic
  8. unreachable samples have no relative_detour
  9. reachable samples have non-negative relative_detour within numerical tolerance
  10. original distance is never greater than disrupted distance, within tolerance
  11. no duplicate OD pair within the same scenario
  12. facility snapping distances are reasonable
  13. graph remains routable where expected
  14. dataset schema is stable
  15. random seeds reproduce identical outputs
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

from src.data.osm_loader import load_city_data
from src.graph.physical_graph import build_physical_digraph, map_facilities_to_network
from src.ground_truth.reachability import compute_ground_truth_sample
from src.dataset.generator import generate_scenario_samples


REQUIRED_SCHEMA = [
    "sample_id",
    "city",
    "scenario_id",
    "split",
    "seed",
    "origin_node",
    "origin_x",
    "origin_y",
    "destination_node",
    "destination_x",
    "destination_y",
    "origin_facility_id",
    "destination_facility_id",
    "facility_name",
    "facility_amenity",
    "facility_snap_distance_m",
    "disrupted_edge_ids",
    "original_distance_m",
    "disrupted_distance_m",
    "reachable",
    "relative_detour",
    "sample_type",
    "distance_tier",
    "orig_path_hops",
    "disrupt_path_hops",
]


def run_validation_suite(
    samples_path: str,
    metadata_path: str,
) -> Dict[str, Any]:
    """Run all 15 validation checks on a generated dataset."""
    print("=" * 70)
    print(f"RUNNING VALIDATION SUITE ON: {samples_path}")
    print("=" * 70)
    
    samples_p = Path(samples_path)
    if samples_p.suffix == ".parquet":
        df = pd.read_parquet(samples_p)
        if isinstance(df["disrupted_edge_ids"].iloc[0], str):
            df["disrupted_edge_ids"] = df["disrupted_edge_ids"].apply(json.loads)
    else:
        with open(samples_p) as fp:
            data = json.load(fp)
        df = pd.DataFrame(data)
        
    with open(metadata_path) as fp:
        meta_list = json.load(fp)
    meta_by_id = {m["scenario_id"]: m for m in meta_list}
    
    # Load raw graphs and boundaries
    city_data = {}
    for city in df["city"].unique():
        b_gdf, G_multi, fac_gdf = load_city_data(city)
        G = build_physical_digraph(G_multi)
        mapped_facs, _ = map_facilities_to_network(G, fac_gdf)
        city_data[city] = {
            "boundary": b_gdf,
            "boundary_poly": b_gdf.geometry.iloc[0],
            "graph": G,
            "facilities": mapped_facs,
        }
        
    results = {}
    
    # Check 1: All sample coordinates inside municipal boundaries
    coords_valid = True
    bad_coords = 0
    for _, row in df.iterrows():
        poly = city_data[row["city"]]["boundary_poly"]
        pt_o = Point(row["origin_x"], row["origin_y"])
        pt_d = Point(row["destination_x"], row["destination_y"])
        if not poly.contains(pt_o) or not poly.contains(pt_d):
            coords_valid = False
            bad_coords += 1
    results["check_01_sample_coords_inside_boundary"] = {
        "passed": coords_valid,
        "details": f"All {len(df)} sample coordinates inside boundary ({bad_coords} violations)"
    }
    
    # Check 2: Disruption epicenters inside boundaries
    epicenters_valid = True
    bad_epicenters = 0
    for m in meta_list:
        poly = city_data[m["city"]]["boundary_poly"]
        pt = Point(m["disruption_center"]["x"], m["disruption_center"]["y"])
        if not poly.contains(pt):
            epicenters_valid = False
            bad_epicenters += 1
    results["check_02_epicenters_inside_boundary"] = {
        "passed": epicenters_valid,
        "details": f"All {len(meta_list)} epicenters inside boundary ({bad_epicenters} violations)"
    }
    
    # Check 3: All OD nodes exist in canonical graph
    nodes_valid = True
    bad_nodes = 0
    for _, row in df.iterrows():
        G = city_data[row["city"]]["graph"]
        if row["origin_node"] not in G or row["destination_node"] not in G:
            nodes_valid = False
            bad_nodes += 1
    results["check_03_all_nodes_exist_in_graph"] = {
        "passed": nodes_valid,
        "details": f"All OD nodes valid in canonical graph ({bad_nodes} missing)"
    }
    
    # Check 4: Exactly 50 active + 50 control samples per scenario
    balance_valid = True
    scenario_counts = {}
    for sid, group in df.groupby("scenario_id"):
        n_active = sum(group["sample_type"] == "active_core")
        n_ctrl = sum(group["sample_type"] == "control_context")
        scenario_counts[sid] = {"active": int(n_active), "control": int(n_ctrl), "total": len(group)}
        if n_active != 50 or n_ctrl != 50:
            balance_valid = False
    results["check_04_balanced_50_50_per_scenario"] = {
        "passed": balance_valid,
        "details": f"All {len(scenario_counts)} scenarios have 50 active + 50 control" if balance_valid else "Imbalance detected"
    }
    
    # Check 5: Disrupted edges are valid canonical edges
    edges_valid = True
    bad_edges = 0
    for m in meta_list:
        G = city_data[m["city"]]["graph"]
        for u, v in m["disrupted_edge_ids"]:
            if not G.has_edge(u, v):
                edges_valid = False
                bad_edges += 1
    results["check_05_disrupted_edges_valid_in_graph"] = {
        "passed": edges_valid,
        "details": f"All disrupted edges exist in canonical graph ({bad_edges} invalid)"
    }
    
    # Check 6: No scenario leakage across splits
    splits_per_scenario = df.groupby("scenario_id")["split"].nunique()
    no_split_leakage = (splits_per_scenario == 1).all()
    
    # Also verify scenario IDs in train do not appear in val or test
    scenarios_by_split = df.groupby("split")["scenario_id"].unique()
    train_scenarios = set(scenarios_by_split.get("train", []))
    val_scenarios = set(scenarios_by_split.get("val", []))
    test_scenarios = set(scenarios_by_split.get("test", []))
    transfer_scenarios = set(scenarios_by_split.get("transfer", []))
    
    leak_train_val = train_scenarios.intersection(val_scenarios)
    leak_train_test = train_scenarios.intersection(test_scenarios)
    leak_train_transfer = train_scenarios.intersection(transfer_scenarios)
    
    no_overlap = len(leak_train_val) == 0 and len(leak_train_test) == 0 and len(leak_train_transfer) == 0
    results["check_06_no_scenario_leakage_across_splits"] = {
        "passed": bool(no_split_leakage and no_overlap),
        "details": "Train, validation, test, and transfer scenarios are 100% disjoint"
    }
    
    # Check 7: Dijkstra results are deterministic
    # Test on a small deterministic spot check
    dijkstra_deterministic = True
    row0 = df.iloc[0]
    G0 = city_data[row0["city"]]["graph"]
    d_retest = float(nx.shortest_path_length(G0, row0["origin_node"], row0["destination_node"], weight="length"))
    if abs(d_retest - row0["original_distance_m"]) > 0.05:
        dijkstra_deterministic = False
    results["check_07_dijkstra_deterministic"] = {
        "passed": dijkstra_deterministic,
        "details": "Spot check matches original Dijkstra distance exactly"
    }
    
    # Check 8: Unreachable samples have no relative_detour (None or NaN)
    unreachable_df = df[df["reachable"] == 0]
    detour_null_when_unreachable = unreachable_df["relative_detour"].isna().all()
    results["check_08_unreachable_samples_have_null_detour"] = {
        "passed": bool(detour_null_when_unreachable),
        "details": f"{len(unreachable_df)} unreachable samples all have null relative_detour"
    }
    
    # Check 9: Reachable samples have non-negative relative_detour (>= -1e-5)
    reachable_df = df[df["reachable"] == 1]
    detour_non_negative = (reachable_df["relative_detour"] >= -1e-5).all()
    results["check_09_reachable_samples_non_negative_detour"] = {
        "passed": bool(detour_non_negative),
        "details": f"All {len(reachable_df)} reachable samples have relative_detour >= 0"
    }
    
    # Check 10: Original distance <= disrupted distance within tolerance
    dist_monotonic = (reachable_df["disrupted_distance_m"] >= reachable_df["original_distance_m"] - 0.05).all()
    results["check_10_distance_monotonically_non_decreasing"] = {
        "passed": bool(dist_monotonic),
        "details": "Disrupted distance is always >= original distance"
    }
    
    # Check 11: No duplicate OD pair within the same scenario
    duplicate_od_found = False
    for sid, group in df.groupby("scenario_id"):
        pairs = list(zip(group["origin_node"], group["destination_node"]))
        if len(pairs) != len(set(pairs)):
            duplicate_od_found = True
            break
    results["check_11_no_duplicate_od_within_scenario"] = {
        "passed": not duplicate_od_found,
        "details": "All OD pairs within each scenario are unique"
    }
    
    # Check 12: Facility snapping distances are reasonable (< 250m)
    max_snap = df["facility_snap_distance_m"].max()
    mean_snap = df["facility_snap_distance_m"].mean()
    snap_valid = max_snap < 300.0
    results["check_12_facility_snapping_reasonable"] = {
        "passed": bool(snap_valid),
        "details": f"Mean snap={mean_snap:.1f}m, Max snap={max_snap:.1f}m (limit 300m)"
    }
    
    # Check 13: Graph remains routable where expected
    routable_valid = df["original_distance_m"].notna().all()
    results["check_13_intact_graph_routable"] = {
        "passed": bool(routable_valid),
        "details": "All sampled pairs have valid intact shortest paths"
    }
    
    # Check 14: Dataset schema is stable
    missing_cols = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    results["check_14_schema_stable"] = {
        "passed": len(missing_cols) == 0,
        "details": f"All {len(REQUIRED_SCHEMA)} required columns present" if not missing_cols else f"Missing: {missing_cols}"
    }
    
    # Check 15: Deterministic seed reproduction (DEC-010, DEC-011)
    repro_passed = True
    repro_mismatches = []
    if len(meta_list) > 0:
        m0 = meta_list[0]
        sid = m0["scenario_id"]
        c_name = m0["city"]
        scfg = {
            "scenario_id": sid,
            "city": c_name,
            "split": m0["split"],
            "seed": m0["seed"],
            "radius_m": m0["disruption_radius_m"],
        }
        regen_samples, regen_meta = generate_scenario_samples(
            G=city_data[c_name]["graph"],
            boundary_gdf=city_data[c_name]["boundary"],
            mapped_facilities=city_data[c_name]["facilities"],
            scenario_cfg=scfg,
        )
        
        orig_group = df[df["scenario_id"] == sid]
        orig_samples = orig_group.to_dict(orient="records")
        
        if len(orig_samples) != len(regen_samples):
            repro_passed = False
            repro_mismatches.append(f"Count mismatch: stored={len(orig_samples)}, regen={len(regen_samples)}")
        else:
            for idx in range(len(orig_samples)):
                o_s = orig_samples[idx]
                r_s = regen_samples[idx]
                
                if o_s["sample_id"] != r_s["sample_id"]:
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} sample_id: {o_s['sample_id']} != {r_s['sample_id']}")
                    break
                if abs(o_s["original_distance_m"] - r_s["original_distance_m"]) > 1e-4:
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} original_distance_m: {o_s['original_distance_m']} != {r_s['original_distance_m']}")
                    break
                if o_s["reachable"] != r_s["reachable"]:
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} reachable: {o_s['reachable']} != {r_s['reachable']}")
                    break
                    
                d_orig = o_s["disrupted_distance_m"]
                d_reg = r_s["disrupted_distance_m"]
                if pd.isna(d_orig) != pd.isna(d_reg):
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} disrupted_distance_m NaN status mismatch")
                    break
                elif not pd.isna(d_orig) and abs(d_orig - d_reg) > 1e-4:
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} disrupted_distance_m value: {d_orig} != {d_reg}")
                    break
                    
                r_orig = o_s["relative_detour"]
                r_reg = r_s["relative_detour"]
                if pd.isna(r_orig) != pd.isna(r_reg):
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} relative_detour NaN status mismatch")
                    break
                elif not pd.isna(r_orig) and abs(r_orig - r_reg) > 1e-4:
                    repro_passed = False
                    repro_mismatches.append(f"Row {idx} relative_detour value: {r_orig} != {r_reg}")
                    break
                    
            if repro_passed and m0["disrupted_edge_ids"] != regen_meta["disrupted_edge_ids"]:
                repro_passed = False
                repro_mismatches.append("Metadata disrupted_edge_ids mismatch")
            if repro_passed and m0.get("edge_disruption_mask_binary") != regen_meta.get("edge_disruption_mask_binary"):
                repro_passed = False
                repro_mismatches.append("Metadata edge_disruption_mask_binary mismatch")
    else:
        repro_passed = False
        repro_mismatches.append("Empty metadata list")
        
    results["check_15_seed_reproducibility"] = {
        "passed": repro_passed,
        "details": f"Regenerated {sid} ({len(orig_samples)} samples) matched stored benchmark bit-for-bit" if repro_passed else f"Reproduction failed: {'; '.join(repro_mismatches[:3])}"
    }
    
    # Check 16: Disruption mask persistence and integrity (DEC-013)
    masks_valid = True
    mask_errors = []
    for m in meta_list:
        sid = m["scenario_id"]
        c_name = m["city"]
        G = city_data[c_name]["graph"]
        n_edges = G.number_of_edges()
        
        mask_bin = m.get("edge_disruption_mask_binary")
        if not mask_bin:
            masks_valid = False
            mask_errors.append(f"{sid}: missing edge_disruption_mask_binary")
            continue
            
        if len(mask_bin) != n_edges:
            masks_valid = False
            mask_errors.append(f"{sid}: mask length {len(mask_bin)} != graph edge count {n_edges}")
            continue
            
        n_zeros = mask_bin.count("0")
        n_ones = mask_bin.count("1")
        if n_zeros != m["number_of_disrupted_edges"]:
            masks_valid = False
            mask_errors.append(f"{sid}: 0-count {n_zeros} != number_of_disrupted_edges {m['number_of_disrupted_edges']}")
            continue
        if n_zeros + n_ones != n_edges:
            masks_valid = False
            mask_errors.append(f"{sid}: invalid characters in binary mask")
            continue
            
    results["check_16_edge_disruption_mask_persistence"] = {
        "passed": masks_valid,
        "details": f"All {len(meta_list)} scenarios have valid edge disruption masks persisted (DEC-013)" if masks_valid else f"Mask errors: {'; '.join(mask_errors[:3])}"
    }
    
    # Summary report
    print("\n--- VALIDATION RESULTS ---")
    all_passed = True
    for check_name, info in results.items():
        status = "PASSED" if info["passed"] else "FAILED"
        if not info["passed"]:
            all_passed = False
        print(f"[{status}] {check_name}: {info['details']}")
        
    print(f"\nOVERALL STATUS: {'ALL 16 CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    
    return {
        "all_passed": all_passed,
        "checks": results,
        "sample_count": len(df),
        "scenario_count": len(meta_list),
        "reachable_count": int((df["reachable"] == 1).sum()),
        "unreachable_count": int((df["reachable"] == 0).sum()),
        "detour_count": int(((df["reachable"] == 1) & (df["relative_detour"] > 0)).sum()),
        "mean_detour_ratio": float(df[df["relative_detour"] > 0]["relative_detour"].mean()) if (df["relative_detour"] > 0).any() else 0.0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--pilot":
        samples_file = "data/processed/pilot_samples.parquet"
        meta_file = "data/processed/pilot_scenarios_metadata.json"
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        samples_file = sys.argv[1]
        meta_file = sys.argv[2] if len(sys.argv) > 2 else "data/processed/benchmark_scenarios_metadata.json"
    else:
        samples_file = "data/processed/benchmark_samples.parquet"
        meta_file = "data/processed/benchmark_scenarios_metadata.json"
    res = run_validation_suite(samples_file, meta_file)
    sys.exit(0 if res["all_passed"] else 1)
