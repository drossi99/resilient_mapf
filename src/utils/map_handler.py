import networkx as nx
from pathlib import Path

MAPS_DIR = str(Path("maps/")) + "/"


def load_map(file_path):
    with open(MAPS_DIR + file_path, "r") as file:
        lines = file.readlines()

    map_start = lines.index("map\n") + 1
    grid = [list(line.strip()) for line in lines[map_start:] if line.strip()]
    return grid


def build_graph(grid):
    graph_dict = {}
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == ".":
                vertex = (i, j)
                neighbors = []
                directions = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == ".":
                        cost = 1.414 if dx != 0 and dy != 0 else 1
                        neighbors.append(((ni, nj), cost))
                graph_dict[vertex] = neighbors

    nx_graph = convert_to_nx_graph(graph_dict)
    return nx_graph


def convert_to_nx_graph(graph_dict):
    nx_graph = nx.Graph()
    for vertex, edges in graph_dict.items():
        for neighbor, cost in edges:
            nx_graph.add_edge(vertex, neighbor, weight=cost)
    return nx_graph
