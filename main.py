import argparse
import json
import numpy as np
import networkx as nx

from Map import Map
from Radar import Radar
from Boundaries import Boundaries
from Location import Location

EPSILON = 1e-4

# -----------------------
# Heuristic for A* (Manhattan × EPSILON)
# -----------------------
def heuristic(a, b):
    (x1, y1), (x2, y2) = a, b
    return (abs(x1 - x2) + abs(y1 - y2)) * EPSILON

# -----------------------
# Load JSON Scenario
# -----------------------
def load_scenario(filepath: str, scenario_key: str):
    with open(filepath, 'r') as f:
        scenarios = json.load(f)

    # Flatten the list of one-item dictionaries into one big dictionary
    all_scenarios = {}
    for entry in scenarios:
        all_scenarios.update(entry)

    if scenario_key not in all_scenarios:
        raise ValueError(f"Scenario '{scenario_key}' not found.")

    scenario = all_scenarios[scenario_key]

    height = scenario['H']
    width = scenario['W']
    boundaries = Boundaries(
        max_lat=scenario['max_lat'],
        min_lat=scenario['min_lat'],
        max_lon=scenario['max_lon'],
        min_lon=scenario['min_lon']
    )

    map_obj = Map(boundaries=boundaries, height=height, width=width)
    map_obj.generate_radars(scenario['n_radars'])

    # Convert POIs (lat, lon) → (i, j)
    lat_range = np.linspace(boundaries.min_lat, boundaries.max_lat, height)
    lon_range = np.linspace(boundaries.min_lon, boundaries.max_lon, width)

    pois = []
    for lat, lon in scenario['POIs']:
        i = np.abs(lat_range - lat).argmin()
        j = np.abs(lon_range - lon).argmin()
        pois.append((i, j))

    return map_obj, pois

# -----------------------
# Path Planner
# -----------------------
def plan_route(graph: nx.DiGraph, pois: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = []
    for i in range(len(pois) - 1):
        start = pois[i]
        goal = pois[i + 1]

        if start not in graph or goal not in graph:
            print(f"[ERROR] POI {start} or {goal} is not in the graph (possibly high detection zone).")
            return []

        try:
            partial = nx.astar_path(graph, start, goal, heuristic=heuristic, weight='weight')
            if i > 0:
                partial = partial[1:]  # avoid repeating nodes
            path.extend(partial)
        except nx.NetworkXNoPath:
            print(f"[ERROR] No path found from {start} to {goal}")
            return []
    return path

# -----------------------
# Print Grid with Path
# -----------------------
def print_path_on_map(path, height, width):
    grid = [['.' for _ in range(width)] for _ in range(height)]
    for (i, j) in path:
        grid[i][j] = 'X'
    for row in grid:
        print(' '.join(row))

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Spy Plane Radar Avoidance Path Planner")
    parser.add_argument("scenario_file", type=str, help="Path to the scenarios.json file")
    parser.add_argument("scenario_key", type=str, help="Scenario key (e.g., scenario_0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Max detection probability allowed")
    args = parser.parse_args()

    print("[INFO] Loading scenario...")
    map_obj, pois = load_scenario(args.scenario_file, args.scenario_key)

    print("[INFO] Computing detection map...")
    prob_map = map_obj.compute_detection_map()

    print("[INFO] Building search graph...")
    graph = map_obj.build_graph(prob_map, threshold=args.threshold)

    print("[INFO] Running A* search between POIs...")
    path = plan_route(graph, pois)

    if path:
        print("[INFO] Final path (X marks the steps):")
        print_path_on_map(path, map_obj.height, map_obj.width)
    else:
        print("[FAILURE] Could not compute a valid route for the provided POIs.")

if __name__ == "__main__":
    main()
