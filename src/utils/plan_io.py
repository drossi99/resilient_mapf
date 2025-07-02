import json
import inspect
from tkinter import messagebox, filedialog

import pickle
import inspect
import datetime
import os

from src.domain.MAPFInstance import RobustnessParams
from pathlib import Path

SOLUTIONS_DIR = str(Path("data/plans"))
SAVE_INSTANCE_INITIAL_DIR = str(Path("data/instances"))
TITLE_SAVE_INSTANCE = "Save MAPF Instance"

def debug_print(message):
    frame = inspect.currentframe().f_back
    file_name = inspect.getfile(frame)
    line_no = frame.f_lineno
    print(f"[DEBUG] {file_name}:{line_no} - {message}")


def save_full_solution_pickle(
    solution,
    mapf_instance,
    robustness_params,
    grid_name,
    filename="fullsolution",
):
    # remove extension from filename
    edited_grid_name = os.path.basename(grid_name).split(".")[0]

    filename = create_solution_name(
        filename, edited_grid_name, robustness_params, len(mapf_instance.starts)
    )
    data = {
        "solution": solution,
        "mapf_instance": mapf_instance,
        "robustness_params": robustness_params,
        "grid_name": grid_name,
    }
    with open(f"{SOLUTIONS_DIR}/{filename}", "wb") as f:
        pickle.dump(data, f)


def create_solution_name(filename, grid_name, robustness_params, num_ag):
    current_date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{filename}_{grid_name}_{robustness_params.k}k_{robustness_params.m}m_{robustness_params.h}h_{robustness_params.selected_failure_types}_{num_ag}ag_{current_date_time}.pkl"


def load_full_solution_pickle(filename="full_solution.pkl"):
    debug_print(f"Loading full solution from {filename} with pickle")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return (
        data["solution"],
        data["mapf_instance"],
        data["robustness_params"],
        data["grid_name"]
    )


def save_instance_to_file(
    map_name,
    start_positions,
    goal_positions,
    robustness_params,
    parent_window=None,
):
    instance = {
        "grid": map_name,
        "start_positions": start_positions,
        "goal_positions": goal_positions,
        "k": robustness_params.k,
        "m": robustness_params.m,
        "h": robustness_params.h,
        "selected_failtypes": robustness_params.selected_failure_types,
    }
    file_path = filedialog.asksaveasfilename(
        title=TITLE_SAVE_INSTANCE,
        initialdir=SAVE_INSTANCE_INITIAL_DIR,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        parent=parent_window,
    )
    if file_path:
        try:
            with open(file_path, "w") as f:
                json.dump(instance, f)
            return True
        except Exception as e:
            debug_print(f"Error saving instance: {e}")
    return False

def load_instance_from_file(filename):
    try:
        debug_print(f"Loading instance from {filename}")
        with open(filename, "r") as f:
            data = json.load(f)
        grid = data.get("grid")
        starts = [tuple(pos) for pos in data.get("start_positions", [])]
        goals = [tuple(pos) for pos in data.get("goal_positions", [])]
        k = data.get("k")
        m = data.get("m")
        h = data.get("h")
        selected_failtypes = data.get("selected_failtypes")
        robustness_params = RobustnessParams(k, m, h, selected_failtypes)
        return grid, starts, goals, robustness_params
    except (FileNotFoundError, json.JSONDecodeError):
        debug_print(f"Failed to load instance from {filename}")
        messagebox.showerror("Error", "Failed to load instance file.")
        return None, None, None, None
