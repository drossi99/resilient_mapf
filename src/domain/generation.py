import pickle
import networkx as nx
from pathlib import Path
import random

from src.utils.map_handler import build_graph
from src.domain.MAPFInstance import MAPFInstance
from src.utils.map_handler import load_map


TEST_INSTANCES_DIR = str(Path("data/test_instances"))
MAX_DST_RANGE = 5

def generate_test_instances(n_instances, n_agents, min_dst, name, map_name):
    grid = load_map(map_name)
    graph = build_graph(grid)

    instances = []
    for i in range(n_instances):
        starts = list()
        goals = list()
        for j in range(n_agents):
            start, goal = extract_start_goal_with_min_distance(
                graph, min_dst, starts, goals
            )
            if start is None or goal is None:
                raise ValueError(
                    f"Impossible to find couples of cells with {min_dst=} in {name}"
                )
            starts.append(start)
            goals.append(goal)
        instance = (map_name, starts, goals)
        instances.append(instance)

    # Save instances with pickle
    with open(f"{TEST_INSTANCES_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(instances, f)

def extract_start_goal_with_min_distance(graph, min_dst, starts, goals):
    max_dst = min_dst + MAX_DST_RANGE
    max_attempts = 100
    nodes = list(graph.nodes)

    for i in range(max_attempts):
        start, goal = random.sample(nodes, 2)
        try:
            dist = nx.shortest_path_length(graph, start, goal)
            if dist >= min_dst and dist <= max_dst:
                if start in starts or goal in goals:
                    continue
                return start, goal
        except nx.NetworkXNoPath:
            continue # No path exists

    for _ in range(len(nodes)):
        start = random.choice(nodes)
        lengths = nx.single_source_shortest_path_length(graph, start)

        candidates = [goal for goal, dist in lengths.items() if dist >= min_dst]
        if candidates:
            goal = random.choice(candidates)
            if start in starts or goal in goals:
                continue
            return start, goal
    return None, None
