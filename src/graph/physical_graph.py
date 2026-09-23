"""Physical Road Graph and Critical Facility Mapping.

Implements Option A canonical representation:
  - Intersection/Junction = Node
  - Road Segment Corridor = Directed Edge
  - Parallel edges collapsed into single canonical corridor edge.
  - Facilities mapped to nearest valid surface road nodes.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import pyproj

FREEWAY_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}


def build_physical_digraph(
    G_multi: nx.MultiDiGraph,
) -> nx.DiGraph:
    """Convert an OSMnx MultiDiGraph to a clean, canonical directed DiGraph.
    
    Parallel directed edges between (u, v) are collapsed into a single corridor edge:
      - length: minimum physical length
      - lanes: sum of parallel lanes
      - bridge: boolean (True if any parallel edge is a bridge)
      - tunnel: boolean (True if any parallel edge is a tunnel)
      - highway: dominant/primary road classification
      - is_disrupted: False (default operational state)
      
    Returns:
        nx.DiGraph with normalized edge and node attributes.
    """
    G = nx.DiGraph()
    G.graph["crs"] = G_multi.graph.get("crs", "EPSG:4326")
    
    # 1. Add nodes with normalized spatial attributes
    for node, data in G_multi.nodes(data=True):
        G.add_node(
            node,
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            street_count=int(data.get("street_count", 0)),
            highway=str(data.get("highway", "")),
        )
        
    # 2. Add edges, collapsing multi-edges to minimum length with aggregated lane counts
    for u, v, k, data in G_multi.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
            
        oneway = bool(data.get("oneway", False))
        lanes = data.get("lanes", "1")
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            lanes_int = int(lanes)
        except (ValueError, TypeError):
            lanes_int = 1
            
        bridge = bool(data.get("bridge", False) and data.get("bridge") != "no")
        tunnel = bool(data.get("tunnel", False) and data.get("tunnel") != "no")
        
        edge_attr = {
            "length": length,
            "highway": str(highway),
            "lanes": lanes_int,
            "oneway": oneway,
            "bridge": bridge,
            "tunnel": tunnel,
            "is_disrupted": False,
        }
        
        if G.has_edge(u, v):
            # Aggregate parallel attributes
            existing = G[u][v]
            # Keep minimum traversal length
            if length < existing["length"]:
                existing["length"] = length
                existing["highway"] = str(highway)
            # Sum parallel lanes
            existing["lanes"] = existing["lanes"] + lanes_int
            # Combine bridge / tunnel flags
            existing["bridge"] = existing["bridge"] or bridge
            existing["tunnel"] = existing["tunnel"] or tunnel
        else:
            G.add_edge(u, v, **edge_attr)
            
    return G


def get_surface_candidate_nodes(G: nx.DiGraph) -> List[Any]:
    """Find road network nodes that connect to surface streets (non-freeway).
    
    Prevents snapping facilities to elevated freeways / viaducts.
    """
    surface_nodes = set()
    for u, v, data in G.edges(data=True):
        hw = data.get("highway", "")
        if hw not in FREEWAY_CLASSES:
            surface_nodes.add(u)
            surface_nodes.add(v)
            
    # If network has only freeways (unlikely), fallback to all nodes
    return list(surface_nodes) if surface_nodes else list(G.nodes())


def map_facilities_to_network(
    G: nx.DiGraph,
    facilities_gdf: gpd.GeoDataFrame,
    max_snap_dist_meters: float = 300.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map POI facilities (hospitals and clinics) to nearest surface road nodes.
    
    Uses geodesic distance via projected coordinates (UTM Zone 10N / EPSG:32610).
    Filters out freeway-only access nodes to prevent snapping to viaducts.
    """
    if facilities_gdf.empty:
        return [], {
            "total_facilities": 0,
            "mapped_facilities": 0,
            "unmapped_facilities": 0,
            "mapping_rate": 0.0,
        }
        
    candidate_node_ids = get_surface_candidate_nodes(G)
    node_coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in candidate_node_ids])
    
    # Projected coordinates via EPSG:32610 (UTM Zone 10N, covers PNW: Seattle & Portland)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    node_x_proj, node_y_proj = transformer.transform(node_coords[:, 0], node_coords[:, 1])
    node_proj_array = np.column_stack([node_x_proj, node_y_proj])
    
    mapped_records = []
    unmapped_count = 0
    snap_distances = []
    
    for idx, row in facilities_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        pt = geom.centroid if geom.geom_type != "Point" else geom
        fx, fy = pt.x, pt.y
        f_proj_x, f_proj_y = transformer.transform(fx, fy)
        
        # Nearest neighbor search in projected Euclidean space (meters)
        dists = np.hypot(node_proj_array[:, 0] - f_proj_x, node_proj_array[:, 1] - f_proj_y)
        nearest_idx = int(np.argmin(dists))
        nearest_dist_m = float(dists[nearest_idx])
        nearest_node = candidate_node_ids[nearest_idx]
        
        facility_name = str(row.get("name", "Unknown Facility"))
        amenity = str(row.get("amenity", "clinic"))
        facility_id = str(row.get("facility_id", idx))
        
        is_mapped = nearest_dist_m <= max_snap_dist_meters
        if not is_mapped:
            unmapped_count += 1
        else:
            snap_distances.append(nearest_dist_m)
            
        rec = {
            "facility_id": facility_id,
            "name": facility_name,
            "amenity": amenity,
            "lon": fx,
            "lat": fy,
            "snapped_node": nearest_node if is_mapped else None,
            "snap_distance_m": round(nearest_dist_m, 2),
            "is_mapped": is_mapped,
        }
        mapped_records.append(rec)
        
    stats = {
        "total_facilities": len(mapped_records),
        "mapped_facilities": len(mapped_records) - unmapped_count,
        "unmapped_facilities": unmapped_count,
        "mapping_rate": float((len(mapped_records) - unmapped_count) / max(len(mapped_records), 1)),
        "mean_snap_distance_m": float(np.mean(snap_distances)) if snap_distances else 0.0,
        "median_snap_distance_m": float(np.median(snap_distances)) if snap_distances else 0.0,
        "max_snap_distance_m": float(np.max(snap_distances)) if snap_distances else 0.0,
    }
    
    return mapped_records, stats
