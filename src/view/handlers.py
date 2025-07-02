import os
import sys
import threading
import time
from tkinter import ttk as tctk
import customtkinter as ctk
from pathlib import Path

import tkinter.messagebox as messagebox
from tkinter import filedialog
import pickle

from src.domain.generation import TEST_INSTANCES_DIR
from src.utils.map_handler import load_map
from src.utils.plan_io import (
    save_instance_to_file,
    load_instance_from_file,
    save_full_solution_pickle,
    load_full_solution_pickle,
)
from src.domain.MAPFInstance import MAPFInstance, RobustnessParams, SearchParams
from src.view.grid_visualizer import GridVisualizer
from src.utils.map_handler import load_map, build_graph
from src.utils.profiling import solve_mapf_with_profiling
from src.domain.solver.solver_manager import solve_mapf


TIME_LIMIT = 5000
SAVE_INSTANCE_INITIAL_DIR = str(Path("data/instances"))
LOAD_PLAN_INITIAL_DIR = str(Path("data/plans"))
DATA_DIR = str(Path("data"))
TITLE_SIMULATE = "MAPF Visualization"
SIZE_SIMULATE = "1500x1200"
TITLE_RUN_TEST = "Starting Test"
SIZE_RUN_TEST = "400x400"

def configure_handlers(root, state, console_textbox):
    def update_agent_display():
        state.agent_listbox.delete(0, "end")
        for i, (start, goal) in enumerate(state.agent_list):
            state.agent_listbox.insert("end", f"Agent {i+1}: Start {start} -> Goal {goal}")

    def add_agent():
        coords = [e.get() for e in state.coord_entries]
        if all(val.isdigit() for val in coords):
            s = (int(coords[0]), int(coords[1]))
            g = (int(coords[2]), int(coords[3]))
            state.agent_list.append((s, g))
            update_agent_display()
            for e in state.coord_entries:
                e.delete(0, "end")
        else:
            messagebox.showerror("Error", "Coordinates must be integers.")

    def remove_agent():
        try:
            index = state.agent_listbox.curselection()[0]
            state.agent_list.pop(index)
            update_agent_display()
        except IndexError:
            messagebox.showerror("Error", "Select an agent to remove.")

    def handle_save_instance():
        if not state.map_var.get():
            messagebox.showerror("Error", "Select a map before saving.")
            return
        try:
            k = int(state.k_var.get())
            h = int(state.h_var.get())
            m = int(state.m_var.get())
        except ValueError:
            messagebox.showerror("Error", "k, h, and m must be integers.")
            return

        selected_failtypes = [ftype for ftype, var in state.fail_type_vars.items() if var.get()]
        try:
            starts = [s for s, _ in state.agent_list]
            goals = [g for _, g in state.agent_list]
            robustness_params = RobustnessParams(k, m, h, selected_failtypes)
            success = save_instance_to_file(
                state.map_var.get(), starts, goals, robustness_params, parent_window=root
            )
            if success:
                messagebox.showinfo("Success", "Instance saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def handle_load_instance():
        path = filedialog.askopenfilename(initialdir=SAVE_INSTANCE_INITIAL_DIR, filetypes=[("JSON files", "*.json")])
        if path:
            grid, starts, goals, rp = load_instance_from_file(path)
            state.map_var.set(grid)
            state.agent_list.clear()
            for s, g in zip(starts, goals):
                state.agent_list.append((tuple(s), tuple(g)))
            update_agent_display()
            state.k_var.set(str(rp.k))
            state.h_var.set(str(rp.h))
            state.m_var.set(str(rp.m))
            for ftype in state.fail_type_vars:
                state.fail_type_vars[ftype].set(ftype in rp.selected_failure_types)

    def handle_simulate_plan():
        path = filedialog.askopenfilename(initialdir=LOAD_PLAN_INITIAL_DIR,filetypes=[("Pickle files", "*.pkl")])
        if path:
            solution, mapf_instance, robustness_params, grid_name = load_full_solution_pickle(path)
            if solution:
                grid = load_map(grid_name)
                vis_root = ctk.CTkToplevel()
                vis_root.title(TITLE_SIMULATE)
                vis_root.geometry(SIZE_SIMULATE)
                app = GridVisualizer(vis_root, grid, mapf_instance, robustness_params, solution)

    def handle_generate_instances():
        try:
            n = int(state.num_instances_var.get())
            a = int(state.num_agents_var.get())
            d = int(state.min_dist_var.get())
            name = state.name_set_var.get()
            mappa = state.map_gen_var.get()

            from src.domain.generation import generate_test_instances
            generate_test_instances(n, a, d, name, mappa)
            messagebox.showinfo("Success", f"{n} instances generated for map {mappa}.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def handle_solve():

        if not state.map_var.get() or not state.agent_list:
            messagebox.showerror("Error", "Select a map and define at least one agent.")
            return
        try:
            grid = load_map(state.map_var.get())
            starts = [s for s, _ in state.agent_list]
            goals = [g for _, g in state.agent_list]
            k = int(state.k_var.get())
            h = int(state.h_var.get())
            m = int(state.m_var.get())
            failtypes = [ftype for ftype, var in state.fail_type_vars.items() if var.get()]

            mapf_instance = MAPFInstance(build_graph(grid), starts, goals)
            initial_failed_actions = tuple(frozenset() for _ in starts)
            initial_failures = [0] * len(starts)
            r_up, r_down, predecessors, resilient_node_macroactions = set(), set(), dict(), dict()

            robustness_params = RobustnessParams(k, m, h, failtypes)
            sp = SearchParams(initial_failed_actions, initial_failures, r_up, r_down, predecessors, resilient_node_macroactions)
            solution = solve_mapf(mapf_instance, robustness_params, sp)

            if solution.tau_states:
                save_full_solution_pickle(solution, mapf_instance, robustness_params, state.map_var.get())
                messagebox.showinfo("Success", f"{k}-resilient plan found successfully.")
                vis_root = ctk.CTkToplevel()
                vis_root.title(TITLE_SIMULATE)
                vis_root.geometry(SIZE_SIMULATE)
                GridVisualizer(vis_root, grid, mapf_instance, robustness_params, solution)
            else:
                messagebox.showinfo("No solution", "No resilient plan found.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during planning: {e}")

    def copy_console():
        content = console_textbox.get("0.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append(content)

    def clear_console():
        console_textbox.delete("1.0", "end")

    # Assign buttons
    state.add_button.configure(command=add_agent)
    state.remove_button.configure(command=remove_agent)
    state.save_instance_btn.configure(command=handle_save_instance)
    state.load_instance_btn.configure(command=handle_load_instance)
    state.simulate_plan_btn.configure(command=handle_simulate_plan)
    state.generate_instances_btn.configure(command=handle_generate_instances)
    state.solve_button.configure(command=handle_solve)
    state.copy_button.configure(command=copy_console)
    state.clear_button.configure(command=clear_console)


def handle_run_test(root, state):
    test_win = ctk.CTkToplevel(root)
    test_win.title(TITLE_RUN_TEST)
    test_win.geometry(SIZE_RUN_TEST)
    test_win.configure(bg="#f0f0f0")

    main_test_frame = ctk.CTkFrame(test_win)
    main_test_frame.pack(fill="both", expand=True, padx=15, pady=15)

    ctk.CTkLabel(main_test_frame, text="Instance set:").grid(row=0, column=0, sticky="w", pady=5)
    test_sets = sorted(os.listdir(TEST_INSTANCES_DIR)) if os.path.exists(TEST_INSTANCES_DIR) else []
    test_set_combo = tctk.Combobox(main_test_frame, values=test_sets, width=25)
    test_set_combo.grid(row=0, column=1, pady=5, padx=(10, 0))

    ctk.CTkLabel(main_test_frame, text="k:").grid(row=1, column=0, sticky="w", pady=5)
    k_entry = tctk.Entry(main_test_frame, width=10)
    k_entry.grid(row=1, column=1, pady=5, padx=(10, 0), sticky="w")

    ctk.CTkLabel(main_test_frame, text="m:").grid(row=2, column=0, sticky="w", pady=5)
    m_entry = tctk.Entry(main_test_frame, width=10)
    m_entry.grid(row=2, column=1, pady=5, padx=(10, 0), sticky="w")

    ctk.CTkLabel(main_test_frame, text="h:").grid(row=3, column=0, sticky="w", pady=5)
    h_entry = tctk.Entry(main_test_frame, width=10)
    h_entry.grid(row=3, column=1, pady=5, padx=(10, 0), sticky="w")



    ctk.CTkLabel(main_test_frame, text="Failure types:").grid(row=4, column=0, columnspan=2, sticky="w", pady=(15, 5))
    for i, (ftype, var) in enumerate(state.fail_type_vars.items()):
        ctk.CTkCheckBox(main_test_frame, text=ftype, variable=var).grid(row=5 + i, column=0, columnspan=2, sticky="w", pady=2)

    def _run_test_in_thread(selected_set, robustness_params, instances, selected_failtypes):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        from src.utils.map_handler import build_graph
        from src.domain.solver.Solution import Solution
        from src.utils.sperimental_analysis import build_solutions_csv

        successes, timeouts = 0, 0
        solutions, timing_stats, timed_out_flags = [], [], []

        for idx, instance in enumerate(instances):
            try:
                print(f"\n*** Processing instance {idx+1}/{len(instances)}...")
                map_name, starts, goals = instance
                grid = load_map(map_name)
                mapf_instance = MAPFInstance(build_graph(grid), starts, goals)
                init_failures = [0] * len(starts)
                init_failed_actions = tuple(frozenset() for _ in starts)
                r_up, r_down, pred, res_macro = set(), set(), dict(), dict()
                search_params = SearchParams(init_failed_actions, init_failures, r_up, r_down, pred, res_macro)

                sol, stat, timed_out = None, {}, False

                # Esegui nel thread con profiling integrato
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        solve_mapf_with_profiling,
                        mapf_instance,
                        robustness_params,
                        search_params
                    )
                    try:
                        sol, stat = future.result(timeout=TIME_LIMIT)
                    except TimeoutError:
                        print(f"⚠️ Timeout reached for instance {idx}")
                        timeouts += 1
                        timed_out = True

                timed_out_flags.append(timed_out)
                if sol and sol.tau_states:
                    selected_set_name = selected_set.split(".")[0]
                    save_full_solution_pickle(sol, mapf_instance, robustness_params, map_name, filename=f"{selected_set_name}_inst{idx}")
                    successes += 1
                else:
                    sol = Solution(None, set(), set(), dict(), dict())

                solutions.append(sol)
                timing_stats.append(stat)

            except Exception as e:
                print(f"❌ Error on instance {idx}: {e}")
                solutions.append(Solution(None, set(), set(), dict(), dict()))
                timing_stats.append(stat)
                timed_out_flags.append(timed_out)

        build_solutions_csv(instances, robustness_params, solutions, selected_set, timing_stats, timed_out_flags)
        print(f"\n !! Test completed: {selected_set} with:\n\tk={robustness_params.k}\n\tm={robustness_params.m}\n\t{robustness_params.h}\n\t{robustness_params.selected_failure_types}\n\n{successes}/{len(instances)} successful, {timeouts} timeouts.")
        messagebox.showinfo("Test completed", f"{selected_set} with:\n\tk={robustness_params.k}\n\tm={robustness_params.m}\n\t{robustness_params.h}\n\t{robustness_params.selected_failure_types}\n\n{successes}/{len(instances)} successful, {timeouts} timeouts\n\nResults saved to CSV.")

    def _handle_start_test():
        selected_set = test_set_combo.get()
        if not selected_set:
            messagebox.showerror("Error", "Select an instance set.")
            return

        try:
            k = int(k_entry.get())
            h = int(h_entry.get())
            m = int(m_entry.get())
        except ValueError:
            messagebox.showerror("Error", "k, h, and m must be integers.")
            return

        selected_failtypes = [ftype for ftype, var in state.fail_type_vars.items() if var.get()]
        if not selected_failtypes:
            messagebox.showerror("Error", "Select at least one failure type.")
            return

        filepath = os.path.join(DATA_DIR, "test_instances", selected_set)
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"No such file: {filepath}")
            return

        with open(filepath, "rb") as f:
            instances = pickle.load(f)

        robustness_params = RobustnessParams(k, m, h, selected_failtypes)
        thread = threading.Thread(target=_run_test_in_thread, args=(selected_set, robustness_params, instances, selected_failtypes), daemon=True)
        thread.start()
        test_win.destroy()

    ctk.CTkButton(main_test_frame, text="Start test", command=_handle_start_test).grid(row=9, column=0, columnspan=2, pady=20)