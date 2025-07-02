import os
import csv

def build_solutions_csv(instances, robustness_params, solutions, selected_set, timing_stats, timed_out_flags):
    directory = ""
    filename = os.path.join(directory, "results.csv")
    file_exists = os.path.exists(filename)

    fieldnames = [
        "test_case_name", "map", "total cells", "percentage of available cells",
        "R_up", "R_down", "success", "timelimit_reached", "k", "m", "h",
        "k-res_solution_cost", "0-res_sol_cost", "delta %", "failure types",
        "time_resplan", "calls_compute_plan_cbs", "time_compute_plan_cbs", "mean_compute_plan_cbs",
        "calls_low_level_search_cbs", "time_low_level_search_cbs", "mean_low_level_search_cbs",
        "calls_safe_interval_table", "time_safe_interval_table", "mean_safe_interval_table"
    ]


    if not file_exists:
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for instance, solution, stats, timeout_flag in zip(instances, solutions, timing_stats, timed_out_flags):
        map_name = instance[0]
        total_cells, percentage_avail_cells = compute_map_info(map_name)

        def safe_stat(fn, key):
            return stats.get(fn, {}).get(key, 0)

        row = {
            "test_case_name": selected_set,
            "map": map_name,
            "total cells": total_cells,
            "percentage of available cells": percentage_avail_cells,
            "R_up": len(solution.R_up),
            "R_down": len(solution.R_down),
            "success": solution.tau_states is not None,
            "timelimit_reached": timeout_flag,
            "k": robustness_params.k,
            "m": robustness_params.m,
            "h": robustness_params.h,
            "k-res_solution_cost": solution.resilient_cost if solution.tau_states is not None else None,
            "0-res_sol_cost": solution.cbs_cost if solution.cbs_cost is not None else None,
            "delta %": round(100 * (solution.resilient_cost - solution.cbs_cost) / solution.cbs_cost, 2) if solution.resilient_cost is not None else None,
            "failure types": robustness_params.selected_failure_types,
            "time_resplan": safe_stat("resplan_mapf", "cumtime"),
            "calls_compute_plan_cbs": safe_stat("compute_plan_cbs", "ncalls"),
            "time_compute_plan_cbs": safe_stat("compute_plan_cbs", "cumtime"),
            "mean_compute_plan_cbs": safe_stat("compute_plan_cbs", "mean_time"),
            "calls_low_level_search_cbs": safe_stat("low_level_search_cbs", "ncalls"),
            "time_low_level_search_cbs": safe_stat("low_level_search_cbs", "cumtime"),
            "mean_low_level_search_cbs": safe_stat("low_level_search_cbs", "mean_time"),
            "calls_safe_interval_table": safe_stat("build_safe_interval_table", "ncalls"),
            "time_safe_interval_table": safe_stat("build_safe_interval_table", "cumtime"),
            "mean_safe_interval_table": safe_stat("build_safe_interval_table", "mean_time")
        }

        with open(filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writerow(row)

def compute_map_info(map_name):
    import networkx as nx
    from src.utils.map_handler import load_map, build_graph

    grid = load_map(map_name)
    graph = build_graph(grid)

    total_cells = sum(len(row) for row in grid)
    free_cells = graph.number_of_nodes()
    percentage_avail_cells = round(100 * free_cells / total_cells, 2)

    return total_cells, percentage_avail_cells
