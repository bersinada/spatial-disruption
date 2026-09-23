"""OSM Data Loader for reproducible road graph and facility extraction."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import geopandas as gpd
import networkx as nx
import osmnx as ox


def load_boundary(city: str, raw_dir: str = "data/raw") -> gpd.GeoDataFrame:
    """Load official municipal boundary GeoDataFrame from cache."""
    boundary_path = Path(raw_dir) / city / "boundary.geojson"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found at {boundary_path}")
    return gpd.read_file(boundary_path)


def load_city_graph(city: str, raw_dir: str = "data/raw") -> nx.MultiDiGraph:
    """Load raw MultiDiGraph from cached GraphML."""
    graph_path = Path(raw_dir) / city / "road_network.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Road network GraphML not found at {graph_path}")
    return ox.load_graphml(graph_path)


def load_city_facilities(city: str, raw_dir: str = "data/raw") -> gpd.GeoDataFrame:
    """Load critical facilities GeoDataFrame (hospitals and clinics) from cache."""
    fac_path = Path(raw_dir) / city / "facilities.geojson"
    if not fac_path.exists():
        raise FileNotFoundError(f"Facilities GeoJSON not found at {fac_path}")
    gdf = gpd.read_file(fac_path)
    # Ensure filtered to hospital and clinic
    gdf = gdf[gdf["amenity"].isin(["hospital", "clinic"])].copy()
    return gdf


def load_city_data(city: str, raw_dir: str = "data/raw") -> Tuple[gpd.GeoDataFrame, nx.MultiDiGraph, gpd.GeoDataFrame]:
    """Load boundary, road network, and facilities for a given city."""
    gdf_boundary = load_boundary(city, raw_dir=raw_dir)
    G_multi = load_city_graph(city, raw_dir=raw_dir)
    gdf_facs = load_city_facilities(city, raw_dir=raw_dir)
    return gdf_boundary, G_multi, gdf_facs
