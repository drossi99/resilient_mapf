from src.utils.map_handler import build_graph

SOLVER_ALGORITHM = "resplan"

def solve_mapf(
    mapf_instance, robustness_params, search_params
):
    if SOLVER_ALGORITHM == "resplan":
        from src.domain.solver.resplan_solver import resplan_mapf

        return resplan_mapf(
            mapf_instance, robustness_params, search_params
        )
    else:
        raise ValueError(f"Solving algorithm not recognized: {SOLVER_ALGORITHM}")
