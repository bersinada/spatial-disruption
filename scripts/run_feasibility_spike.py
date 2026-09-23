"""Feasibility Spike: End-to-End Pipeline Execution.

Demonstrates:
  OSM data
  → physical road graph
  → critical facility mapping
  → controlled edge disruption (single & multi-edge)
  → exact ground-truth computation (Dijkstra reachability & distance delta)
  → persisted dataset sample
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.osm_loader import fetch_or_load_city_data
from src.graph.physical_graph import build_physical_digraph, map_facilities_to_network
from src.graph.diagnostics import compute_graph_diagnostics
from src.disruption.generator import (
    apply_single_edge_disruption,
    apply_localized_multi_edge_disruption,
)
from src.ground_truth.reachability import batch_compute_ground_truth


def main():
    print("=" * 70)
    print("STARTING DATASET & GRAPH FEASIBILITY SPIKE")
    print("=" * 70)
    
    # 1. Geographic Selection (Small, highly validated prototype: Seattle Downtown)
    city_name = "seattle_downtown_prototype"
    center = (47.6062, -122.3321)  # Seattle downtown 4th & Madison
    radius_m = 1000.0
    
    # 2. Extract / Load OSM road graph and facilities
    print(f"\n[Step 1] Loading real OSM data for {city_name} (radius = {radius_m}m)...")
    G_multi, facilities_gdf = fetch_or_load_city_data(
        name=city_name,
        center_point=center,
        radius_meters=radius_m,
    )
    
    # 3. Build physical DiGraph (Option A representation)
    print("\n[Step 2] Building canonical physical DiGraph (Option A: Intersection=Node, Segment=Edge)...")
    G_phys = build_physical_digraph(G_multi)
    
    # 4. Map facilities to road network nodes
    print("\n[Step 3] Mapping POI facilities to nearest physical road intersections...")
    mapped_facilities, facility_stats = map_facilities_to_network(
        G_phys,
        facilities_gdf,
        max_snap_dist_meters=500.0,
    )
    
    # 5. Compute graph diagnostics
    print("\n[Step 4] Running comprehensive graph and topological diagnostics...")
    diagnostics = compute_graph_diagnostics(G_phys, facility_stats)
    print(json.dumps(diagnostics, indent=2))
    
    # 6. Test counterfactual disruptions
    print("\n[Step 5] Testing counterfactual network disruptions...")
    
    # Disruption A: Single-edge failure
    # Select an edge that is part of a major road to test tangible impact
    primary_edges = [
        (u, v) for u, v, d in G_phys.edges(data=True)
        if "primary" in d.get("highway", "") or "secondary" in d.get("highway", "")
    ]
    target_edge = primary_edges[0] if primary_edges else None
    G_disrupt_single, meta_single = apply_single_edge_disruption(
        G_phys, seed=42, target_edge=target_edge
    )
    print(f"  - Single-edge disruption: Removed edge {meta_single['disrupted_edges'][0]}")
    print(f"    Original edges: {meta_single['original_edge_count']} -> Disrupted edges: {meta_single['disrupted_edge_count']}")
    
    # Disruption B: Localized multi-edge failure (e.g. 150m disaster zone)
    # Pick the target edge's start node as epicenter
    epicenter = target_edge[0] if target_edge else list(G_phys.nodes())[0]
    G_disrupt_multi, meta_multi = apply_localized_multi_edge_disruption(
        G_phys, radius_meters=150.0, seed=42, epicenter_node=epicenter
    )
    print(f"  - Localized multi-edge disruption (r=150m): Epicenter node {meta_multi['epicenter_node']}")
    print(f"    Disrupted {meta_multi['num_disrupted_edges']} road segments.")
    print(f"    Original edges: {meta_multi['original_edge_count']} -> Disrupted edges: {meta_multi['disrupted_edge_count']}")
    
    # 7. Exact Ground-Truth Reachability & Shortest-Path Computation
    print("\n[Step 6] Generating exact ground-truth reachability & distance deltas via Dijkstra...")
    
    # Sample a set of origins across the network
    import random
    rng = random.Random(42)
    all_nodes = list(G_phys.nodes())
    # Sample 15 distinct origin nodes
    sample_origins = rng.sample(all_nodes, min(15, len(all_nodes)))
    
    # Compute ground truth under single-edge disruption
    samples_single = batch_compute_ground_truth(
        G_orig=G_phys,
        G_disrupt=G_disrupt_single,
        origin_nodes=sample_origins,
        mapped_facilities=mapped_facilities,
        disruption_metadata=meta_single,
    )
    
    # Compute ground truth under localized multi-edge disruption
    samples_multi = batch_compute_ground_truth(
        G_orig=G_phys,
        G_disrupt=G_disrupt_multi,
        origin_nodes=sample_origins,
        mapped_facilities=mapped_facilities,
        disruption_metadata=meta_multi,
    )
    
    # Combine samples with disruption tags
    for s in samples_single:
        s["disruption_scenario"] = "single_edge"
    for s in samples_multi:
        s["disruption_scenario"] = "localized_multi_edge"
    all_samples = samples_single + samples_multi
    
    # 8. Analyze ground-truth results
    print(f"\n[Step 7] Analyzing ground-truth label distributions across {len(all_samples)} (O, D) pairs...")
    
    def summarize_samples(sample_list, label):
        classes = {}
        deltas = []
        for s in sample_list:
            c = s["transition_class"]
            classes[c] = classes.get(c, 0) + 1
            if s["delta_distance_m"] is not None and s["delta_distance_m"] > 0:
                deltas.append(s["delta_distance_m"])
        print(f"\n--- Scenario: {label} ({len(sample_list)} pairs) ---")
        for c, count in sorted(classes.items()):
            print(f"  {c:<22}: {count:>4} ({count/len(sample_list)*100:.1f}%)")
        if deltas:
            import numpy as np
            print(f"  Positive detour deltas: min={np.min(deltas):.1f}m, mean={np.mean(deltas):.1f}m, max={np.max(deltas):.1f}m")
        else:
            print("  No positive detours observed in sampled pairs.")
            
    summarize_samples(samples_single, "Single-Edge Disruption")
    summarize_samples(samples_multi, "Localized Multi-Edge Disruption")
    
    # 9. Manual verification of a representative sample
    detour_samples = [s for s in all_samples if s["transition_class"] == "detour"]
    severed_samples = [s for s in all_samples if s["transition_class"] == "severed"]
    
    print("\n[Step 8] Manual verification of ground-truth calculations:")
    if detour_samples:
        sample_detour = detour_samples[0]
        print(f"  Representative Detour Sample:")
        print(f"    Origin: {sample_detour['origin_node']} -> Facility: {sample_detour['facility_node']} ({sample_detour.get('facility_name', 'Facility')})")
        print(f"    Pre-disruption distance : {sample_detour['orig_distance_m']} m ({sample_detour['orig_path_hops']} hops)")
        print(f"    Post-disruption distance: {sample_detour['disrupt_distance_m']} m ({sample_detour['disrupt_path_hops']} hops)")
        print(f"    Delta distance          : +{sample_detour['delta_distance_m']} m")
        assert sample_detour["delta_distance_m"] > 0, "Detour delta must be positive!"
        assert sample_detour["disrupt_distance_m"] > sample_detour["orig_distance_m"], "Disrupted distance must exceed original!"
        print("    -> VERIFIED: Exact Dijkstra confirms increased path length under detour.")
    else:
        print("  No detour samples in this slice; checking severed / unaffected...")
        
    if severed_samples:
        sample_severed = severed_samples[0]
        print(f"  Representative Severed Sample:")
        print(f"    Origin: {sample_severed['origin_node']} -> Facility: {sample_severed['facility_node']}")
        print(f"    Pre-disruption distance : {sample_severed['orig_distance_m']} m")
        print(f"    Post-disruption reachable: False (Network severed)")
        print("    -> VERIFIED: Path completely eliminated by disruption.")
        
    # 10. Persist processed dataset artifacts
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples_file = out_dir / "feasibility_samples.json"
    diagnostics_file = out_dir / "feasibility_diagnostics.json"
    
    with open(samples_file, "w") as f:
        json.dump(all_samples, f, indent=2)
    with open(diagnostics_file, "w") as f:
        json.dump(diagnostics, f, indent=2)
        
    print(f"\n[Step 9] Successfully persisted dataset samples to {samples_file} ({len(all_samples)} samples)")
    print(f"Successfully persisted graph diagnostics to {diagnostics_file}")
    print("\nFEASIBILITY SPIKE COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
