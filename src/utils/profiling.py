import signal
import cProfile
import pstats
import io

from src.domain.solver.solver_manager import solve_mapf

DEBUG = True
class TimeoutException(Exception):
    pass

def debug_print(message):
    if DEBUG:
        print(message)

def timeout_handler(signum, frame):
    raise TimeoutException()

def profile_code(func, *args, stdout_redirector=None, **kwargs):
    import cProfile, pstats, io, sys

    # Create a new profiler for each instance
    pr = cProfile.Profile()

    original_stdout = sys.stdout
    if stdout_redirector is not None:
        sys.stdout = stdout_redirector

    try:
        pr.clear()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
    finally:
        # Reset the profiler
        sys.setprofile(None)
        if stdout_redirector is not None:
            sys.stdout = original_stdout

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    stats_dict = extract_stats(ps)

    return result, stats_dict

def solve_mapf_with_profiling(mapf_instance, robustness_params, search_params):
    import cProfile
    import pstats
    import io
    import threading

    print(f"[DEBUG] Profiling in thread: {threading.current_thread().name}")
    pr = cProfile.Profile()

    try:
        pr.enable()
        result = solve_mapf(mapf_instance, robustness_params, search_params)
        pr.disable()

        print(f"[DEBUG] Profiling completed, stats count: {len(pr.stats) if hasattr(pr, 'stats') else 0}")

    except Exception as e:
        pr.disable()
        print(f"[DEBUG] Exception in solve_mapf: {e}")
        raise

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")

    def extract_stats_local(ps):
        stats_dict = {}
        target_functions = [
            "resplan_mapf",
            "compute_plan_cbs",
            "low_level_search_cbs",
            "build_safe_interval_table"
        ]

        if not ps.stats:
            return stats_dict

        if ps.stats:
            first_key = next(iter(ps.stats))
            first_value = ps.stats[first_key]
            print(f"[DEBUG] First value structure: {type(first_value)} - {first_value}")

        for func_key, stat_data in ps.stats.items():
            try:
                if isinstance(stat_data, tuple):
                    if len(stat_data) == 4:
                        ncalls, tottime, cumtime, callers = stat_data
                    elif len(stat_data) == 5:
                        # Format: (cc, nc, tt, ct, callers)
                        cc, nc, tottime, cumtime, callers = stat_data
                        ncalls = nc
                    else:
                        print(f"[DEBUG] Unexpected tuple length: {len(stat_data)} for {func_key}")
                        continue
                else:
                    print(f"[DEBUG] Unexpected data type: {type(stat_data)} for {func_key}")
                    continue

                if isinstance(func_key, tuple) and len(func_key) >= 3:
                    func_name = func_key[2]
                else:
                    func_name = str(func_key)

                for target in target_functions:
                    if target in func_name:
                        if target not in stats_dict:
                            stats_dict[target] = {
                                'ncalls': 0,
                                'tottime': 0.0,
                                'cumtime': 0.0,
                                'mean_time': 0.0
                            }

                        stats_dict[target]['ncalls'] += ncalls
                        stats_dict[target]['tottime'] += tottime
                        stats_dict[target]['cumtime'] += cumtime

            except Exception as e:
                print(f"[DEBUG] Error processing {func_key}: {e}")
                continue

        # Calcola i tempi medi
        for func_name in stats_dict:
            if stats_dict[func_name]['ncalls'] > 0:
                stats_dict[func_name]['mean_time'] = (
                        stats_dict[func_name]['cumtime'] / stats_dict[func_name]['ncalls']
                )

        return stats_dict

    stats_dict = extract_stats_local(ps)
    print(f"[DEBUG] Extracted stats: {stats_dict}")

    return result, stats_dict

def extract_stats(ps):
    functions_of_interest = {
        "resplan_mapf": None,
        "compute_plan_cbs": None,
        "low_level_search_cbs": None,
        "build_safe_interval_table": None,
    }

    extracted = {
        key: {"ncalls": 0, "tottime": 0.0, "cumtime": 0.0}
        for key in functions_of_interest
    }

    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        filename, lineno, funcname = func
        if funcname in functions_of_interest:
            extracted[funcname] = {
                "ncalls": nc,
                "tottime": round(tt, 3),
                "cumtime": round(ct, 3),
                "mean_time": round(ct / nc, 5) if nc else 0,
            }
    return extracted