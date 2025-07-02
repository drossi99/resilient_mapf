import customtkinter as ctk
from tkinter import ttk
import tkinter.ttk as tctk

from src.domain.MAPFInstance import MAPFInstance, RobustnessParams, SearchParams
from src.domain.solver.solver_manager import solve_mapf
from src.domain.solver.resplan_solver import compute_affected_actions

DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)

def apply_action(state, action_vector, fail_indices, grid):
    new_state = []
    moves = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "up_left": (-1, -1),
        "up_right": (-1, 1),
        "down_left": (1, -1),
        "down_right": (1, 1),
        "wait": (0, 0),
    }
    for i, pos in enumerate(state):
        if i in fail_indices:
            new_state.append(pos)
            debug_print(f"\n\tapply_action: agent {i} FAILS, remains at {pos}")
        else:
            dr, dc = moves.get(action_vector[i], (0, 0))
            new_row = pos[0] + dr
            new_col = pos[1] + dc
            if (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid[0])
                and grid[new_row][new_col] != "@"
            ):
                new_pos = (new_row, new_col)
            else:
                new_pos = pos
            new_state.append(new_pos)
            debug_print(
                f"\napply_action: agent {i} with action '{action_vector[i]}' moves to {new_pos}"
            )
    return tuple(new_state)

class GridVisualizer:
    def __init__(
        self,
        root,
        grid,
        mapf_instance,
        robustness_params,
        solution,
    ):

        self.root = root
        self.grid = grid
        self.mapf_instance = mapf_instance

        self.robust_strategy = solution.tau_states
        self.k = robustness_params.k
        self.h = robustness_params.h
        self.m = robustness_params.m
        self.paths = (
            self.extract_nominal_paths() if solution.tau_states is not None else None
        )
        self.plan = solution.tau_states
        self.current_step = 0
        self.show_full_paths = ctk.BooleanVar(value=False)
        self.cell_size = 20
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        self.selected_agent = 0
        self.conflicting_cells = []
        self.dynamic_obstacles = []
        self.failed_actions = [set() for _ in self.mapf_instance.starts]
        self.R_up = solution.R_up
        self.R_down = solution.R_down
        self.predecessors = solution.predecessors
        self.resilient_node_macroactions = solution.resilient_node_macroactions
        self.failure_events = []
        self.affected_cells = set()
        self.failed_cells = set()
        self.selected_failure_type = ctk.StringVar(value="topw")
        self.selected_failure_types = robustness_params.selected_failure_types
        self.failed_agents = [0] * len(self.mapf_instance.starts)

        self.agent_colors = [
            "#3498DB",
            "#9B59B6",
            "#F1C40F",
            "#E67E22",
            "#1ABC9C",
            "#8E44AD",
            "#D35400",
            "#27AE60",
        ]

        main_frame = ctk.CTkFrame(root)
        main_frame.pack(expand=True, fill=ctk.BOTH)

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(
            side=ctk.LEFT,
            expand=True,
            fill=ctk.BOTH,
        )

        right_frame = ctk.CTkFrame(main_frame, width=300)
        right_frame.pack(side=ctk.RIGHT, fill=ctk.Y)
        right_frame.pack_propagate(False)

        self.canvas = ctk.CTkCanvas(left_frame, bg="white", highlightthickness=0)
        self.canvas.pack(expand=True, fill=ctk.BOTH)

        control_frame = ctk.CTkFrame(left_frame)
        control_frame.pack(pady=10)

        # Operations buttons
        self.reset_button = ctk.CTkButton(
            control_frame, text="⏮ Reset", command=self.reset
        )
        self.reset_button.grid(row=0, column=0, padx=5)
        self.zoom_out_button = ctk.CTkButton(
            control_frame, text="🔍- Zoom Out", command=self.zoom_out
        )
        self.zoom_out_button.grid(row=0, column=1, padx=5)
        self.zoom_in_button = ctk.CTkButton(
            control_frame, text="🔍+ Zoom In", command=self.zoom_in
        )
        self.zoom_in_button.grid(row=0, column=2, padx=5)
        self.step_back_button = ctk.CTkButton(
            control_frame, text="⏴ Previous Step", command=self.previous_step
        )
        self.step_back_button.grid(row=0, column=3, padx=5)
        self.step_button = ctk.CTkButton(
            control_frame, text="⏵ Next Step", command=self.next_step
        )
        self.step_button.grid(row=0, column=4, padx=5)
        self.play_button = ctk.CTkButton(
            control_frame, text="▶ Play", command=self.play_all
        )
        self.play_button.grid(row=0, column=5, padx=5)
        self.toggle_paths = ctk.CTkCheckBox(
            control_frame,
            text="Show Full Paths",
            variable=self.show_full_paths,
            command=self.draw_grid,
        )
        self.toggle_paths.grid(row=0, column=6, padx=5)

        # Failure simulation controls
        self.agent_selector_label = ctk.CTkLabel(right_frame, text="Select agent:")
        self.agent_selector_label.pack(pady=5)
        self.agent_selector = tctk.Combobox(
            right_frame,
            values=[f"Agent {i+1}" for i in range(len(self.mapf_instance.starts))],
        )
        self.agent_selector.current(0)
        self.agent_selector.bind("<<ComboboxSelected>>", self.on_agent_selected)
        self.agent_selector.pack(pady=5)

        failure_type_label = ctk.CTkLabel(right_frame, text="Failure type:")
        failure_type_label.pack(pady=(15, 2))

        # Radio buttons for fail type selection
        self.selected_failure_type = ctk.StringVar(value="topw")  # default value

        failure_types = [
            ("Topological weak", "topw"),
            ("Topological strong", "tops"),
            ("Individual", "individual"),
            ("High level action", "high-level"),
        ]

        for text, value in failure_types:
            if value in self.selected_failure_types:
                rb = ctk.CTkRadioButton(
                    right_frame,
                    text=text,
                    variable=self.selected_failure_type,
                    value=value,
                )
                rb.pack(anchor=ctk.W, padx=20)

        self.select_agent_delay_button = ctk.CTkButton(
            right_frame,
            text="Simulate failure",
            command=self.apply_selected_agent_failure,
        )
        self.select_agent_delay_button.pack(pady=5)

        # Budget info
        self.stats_label = ctk.CTkLabel(
            left_frame, text="", anchor="center", font=("Arial", 14, "bold")
        )
        self.stats_label.pack(pady=10)
        self.k_label = ctk.CTkLabel(
            left_frame,
            text=f"k left: {self.k}, max failable agents: {self.m}, max failures for each agent: {self.h}, failures of each agent: {self.failed_agents}",
            anchor="center",
            font=("Arial", 12),
        )
        self.k_label.pack(pady=5)

        # Plan log
        paths_label = ctk.CTkLabel(
            right_frame, text="Paths:", font=("Arial", 12, "bold")
        )
        paths_label.pack(pady=5)
        self.paths_text = ctk.CTkTextbox(
            right_frame, width=70, height=30, font=("Courier", 10)
        )
        self.paths_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Failure log
        failure_log_label = ctk.CTkLabel(
            right_frame, text="Failure Log:", font=("Arial", 12, "bold")
        )
        failure_log_label.pack(pady=5)

        self.failure_log_text = ctk.CTkTextbox(
            right_frame, width=70, height=15, font=("Courier", 10)
        )
        self.failure_log_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.tooltip = ctk.CTkLabel(
            root,
            text="",
            font=("Arial", 10),
            fg_color="yellow",
            text_color="black",
            corner_radius=4
        )
        self.tooltip.place_forget()

        self.canvas.bind("<ButtonPress-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", self.stop_pan)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.show_tooltip)

        self.update_cell_size()
        self.draw_grid()
        self.update_paths_display()

    def on_agent_selected(self, event):
        selected_text = self.agent_selector.get()
        try:
            agent_id = int(selected_text.split()[1]) - 1
            self.selected_agent = agent_id
            debug_print(f"\nSelected agent: {agent_id + 1}")
        except Exception as e:
            debug_print(f"Error in the selection of agent: {e}")

    def update_cell_size(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        rows, cols = len(self.grid), len(self.grid[0])
        self.cell_size = min(canvas_width // cols, canvas_height // rows)

    def extract_nominal_paths(self):
        num_agents = len(self.mapf_instance.starts)
        paths = [[self.mapf_instance.starts[i]] for i in range(num_agents)]
        current_state = tuple(self.mapf_instance.starts)

        for macro in self.robust_strategy:
            action_vec = tuple(a for a, _ in macro)
            current_state = apply_action(current_state, action_vec, set(), self.grid)
            for i in range(num_agents):
                paths[i].append(current_state[i])

        return paths


    def draw_grid(self):
        self.canvas.delete("all")
        rows, cols = len(self.grid), len(self.grid[0])
        cell_roles = {}

        for idx, pos in enumerate(self.mapf_instance.starts):
            cell_roles.setdefault(pos, []).append(("start", idx + 1))
        for idx, pos in enumerate(self.mapf_instance.goals):
            cell_roles.setdefault(pos, []).append(("goal", idx + 1))

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                x1 = j * self.cell_size + self.offset_x
                y1 = i * self.cell_size + self.offset_y
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if cell == "@":
                    color = "#2C3E50"
                elif cell == "T":
                    color = "#1E8449"
                elif cell == ".":
                    color = "#ECF0F1"
                elif cell == "E":
                    color = "#3498DB"
                elif cell == "S":
                    color = "#9B59B6"
                else:
                    color = "#ECF0F1"
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="#BDC3C7"
                )

        margin = self.cell_size * 0.1
        diameter = self.cell_size * 0.8
        mini_diameter = diameter * 0.5
        for (row, col), roles in cell_roles.items():
            if len(roles) > 1:
                offset = -mini_diameter / 2
                for role, agent in roles:
                    fill_color = "#27AE60" if role == "start" else "#E74C3C"
                    x1 = col * self.cell_size + margin + self.offset_x + offset
                    y1 = row * self.cell_size + margin + self.offset_y + offset
                    x2 = x1 + mini_diameter
                    y2 = y1 + mini_diameter
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill=fill_color, outline="#000000"
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=f"{role[0].upper()}{agent}",
                        fill="white",
                        font=("Arial", int(diameter // 4)),
                    )
                    offset += mini_diameter + 2
            else:
                role, agent = roles[0]
                fill_color = "#27AE60" if role == "start" else "#E74C3C"
                x1 = col * self.cell_size + margin + self.offset_x
                y1 = row * self.cell_size + margin + self.offset_y
                x2 = x1 + diameter
                y2 = y1 + diameter
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=fill_color, outline="#000000"
                )
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=f"{role[0].upper()}{agent}",
                    fill="white",
                    font=("Arial", int(diameter // 4)),
                )

        self.mark_conflicting_cells()

        if self.show_full_paths.get() and self.paths is not None:
            for agent_idx, path in enumerate(self.paths):
                path_color = self.agent_colors[agent_idx % len(self.agent_colors)]
                offset = (agent_idx % 3 - 1) * 3
                points = []
                for t in range(min(len(path), self.current_step + 1)):
                    pos = path[t]
                    x = (
                        int(pos[1]) * self.cell_size
                        + self.offset_x
                        + offset
                        + self.cell_size // 2
                    )
                    y = (
                        int(pos[0]) * self.cell_size
                        + self.offset_y
                        + offset
                        + self.cell_size // 2
                    )
                    points.extend([x, y])
                if len(points) > 2:
                    self.canvas.create_line(
                        points, fill=path_color, width=2, smooth=False, splinesteps=0
                    )

        for agent_idx, path in enumerate(self.paths if self.paths is not None else []):
            margin = self.cell_size * 0.1
            diameter = self.cell_size * 0.8
            if self.current_step < len(path):
                pos = path[self.current_step]
                x1 = pos[1] * self.cell_size + margin + self.offset_x
                y1 = pos[0] * self.cell_size + margin + self.offset_y
                x2 = x1 + diameter
                y2 = y1 + diameter
                color = self.agent_colors[agent_idx % len(self.agent_colors)]
                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="#F39C12")
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(agent_idx + 1),
                    fill="white",
                    font=("Arial", int(diameter // 4)),
                )
            elif len(path) > 0:
                pos = path[-1]
                x1 = pos[1] * self.cell_size + margin + self.offset_x
                y1 = pos[0] * self.cell_size + margin + self.offset_y
                x2 = x1 + diameter
                y2 = y1 + diameter
                color = self.agent_colors[agent_idx % len(self.agent_colors)]
                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="#F39C12")
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(agent_idx + 1),
                    fill="white",
                    font=("Arial", int(diameter // 4)),
                )

        max_steps = max(len(path) for path in self.paths) - 1 if self.paths else 0
        self.stats_label.configure(text=f"Step {self.current_step}/{max_steps}")
        self.k_label.configure(
            text=f"k left: {self.k}, max failable agents: {self.m}, max failures for each agent: {self.h}, failures of each agent: {self.failed_agents}"
        )

        self.update_paths_display()
        # Mark conflicting cells
        for row, col in self.failed_cells:
            center_x = col * self.cell_size + self.offset_x + self.cell_size / 2
            center_y = row * self.cell_size + self.offset_y + self.cell_size / 2
            r = self.cell_size * 0.05
            self.canvas.create_oval(
                center_x - r,
                center_y - r,
                center_x + r,
                center_y + r,
                fill="red",
                outline="black",
            )

        # Mark affected cells
        for row, col in self.affected_cells:
            center_x = col * self.cell_size + self.offset_x + self.cell_size / 2
            center_y = row * self.cell_size + self.offset_y + self.cell_size / 2
            r = self.cell_size * 0.07
            self.canvas.create_oval(
                center_x - r,
                center_y - r,
                center_x + r,
                center_y + r,
                fill="orange",
                outline="black",
            )

    def update_paths_display(self):
        self.paths_text.delete(1.0, ctk.END)
        if self.paths is None:
            self.paths_text.insert(ctk.END, "No path available.")
            return

        for agent_idx, path in enumerate(self.paths):
            self.paths_text.insert(ctk.END, f"Agent {agent_idx + 1} Path:\n")
            for step, pos in enumerate(path):
                # Evidenzia il passo corrente
                if step == self.current_step:
                    self.paths_text.insert(
                        ctk.END, f"Step {step}: ({pos[0]}, {pos[1]}) <- Current\n"
                    )
                else:
                    self.paths_text.insert(
                        ctk.END, f"Step {step}: ({pos[0]}, {pos[1]})\n"
                    )
            self.paths_text.insert(ctk.END, "\n")

    def handle_click(self, event):
        col = (event.x - self.offset_x) // self.cell_size
        row = (event.y - self.offset_y) // self.cell_size
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            self.k_label.configure(text=f"Cell Info: Row={row}, Col={col}")
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def pan(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.draw_grid()

    def stop_pan(self, event):
        self.is_panning = False

    def zoom_in(self):
        self.cell_size += 2
        self.draw_grid()

    def zoom_out(self):
        if self.cell_size > 2:
            self.cell_size -= 2
            self.draw_grid()

    def previous_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.draw_grid()

    def next_step(self):
        max_steps = max(len(path) for path in self.paths) - 1 if self.paths else 0
        if self.current_step < max_steps:
            self.current_step += 1
            self.draw_grid()

    def play_all(self):
        max_steps = max(len(path) for path in self.paths) - 1 if self.paths else 0
        for _ in range(self.current_step, max_steps):
            self.next_step()
            self.root.update_idletasks()
            self.root.after(100)

    def on_resize(self, event):
        self.update_cell_size()
        self.draw_grid()

    def show_tooltip(self, event):
        col = (event.x - self.offset_x) // self.cell_size
        row = (event.y - self.offset_y) // self.cell_size
        tooltip_text = f"Row: {row}, Col: {col}"
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            self.tooltip.configure(text=tooltip_text)
            self.tooltip.place(x=event.x + 10, y=event.y + 10)
        else:
            self.tooltip.place_forget()
        info = []
        for idx, pos in enumerate(self.mapf_instance.starts):
            if (row, col) == pos:
                info.append(f"\nStart of Agent {idx + 1}")
        for idx, pos in enumerate(self.mapf_instance.goals):
            if (row, col) == pos:
                info.append(f"\nGoal of Agent {idx + 1}")
        self.tooltip.configure(text=f"{tooltip_text}{', '.join(info)}")

    def reset(self):
        self.current_step = 0
        self.draw_grid()

    def apply_selected_agent_failure(self):
        ag = self.selected_agent
        t = self.current_step

        if t >= len(self.paths[ag]):
            debug_print("⚠ End of path: no action to fail.")
            return

        # macroaction
        macro_entry = self.robust_strategy[t]
        macroaction = (
            macro_entry if isinstance(macro_entry[0], tuple) else macro_entry[0]
        )
        failed_act = macroaction[ag]
        macroaction_to_execute = list(macroaction)
        macroaction_to_execute[ag] = ("wait", failed_act[1])

        # state before failure
        current_state = tuple(path[t] for path in self.paths)
        next_state = tuple(path[t + 1] for path in self.paths)
        next_state = list(next_state)
        next_state[ag] = current_state[ag]

        fail_type = self.selected_failure_type.get()

        debug_print(
            f"***\n\t>>> agent {ag+1} failed – type '{fail_type}' – action {failed_act}\n***"
        )

        affected = compute_affected_actions(
            failed_act,
            ag,
            fail_type,
            current_state,
            self.mapf_instance.graph,
            self.mapf_instance.goals,
        )

        for i in range(len(self.failed_actions)):
            self.failed_actions[i].update(affected[i])

        # update budget
        self.k -= 1
        if self.k < 0:
            debug_print("\n\t Allowed failures exceeded!")
            return

        # recall solver
        failed_tuple = tuple(frozenset(s) for s in self.failed_actions)
        self.failed_agents[ag] = self.failed_agents[ag] + 1

        if self.failed_agents[ag] > self.h:
            debug_print(f"\n\t Agent {ag+1} exceeded max failures ({self.h}).")
            return
        new_mapf_instance = MAPFInstance(self.mapf_instance.graph, next_state, self.mapf_instance.goals)
        robustness_params = RobustnessParams(self.k, self.m, self.h, self.selected_failure_types)
        search_params = SearchParams(failed_tuple, self.failed_agents, self.R_up, self.R_down, self.predecessors, self.resilient_node_macroactions)

        new_solution = solve_mapf(
            new_mapf_instance,
            robustness_params,
            search_params,
        )
        # Merge existing plan with new solution
        if new_solution.tau_states:
            debug_print("***\n\tNew robust plan computed!\n***")
            prefix = self.robust_strategy[:t] + [macroaction_to_execute]
            self.R_up = new_solution.R_up
            self.R_down = new_solution.R_down
            self.predecessors = new_solution.predecessors
            self.resilient_node_macroactions = new_solution.resilient_node_macroactions
            self.robust_strategy = prefix + new_solution.tau_states
            self.paths = self.extract_nominal_paths()
            self.update_paths_display()
        else:
            debug_print("\n\tNo robust plan found after failing.")

        self.failure_events.append(
            {
                "step": self.current_step,
                "agent": ag,
                "action": failed_act,
                "fail_type": fail_type,
                "affected_actions": affected,
            }
        )

        # Add failed and affected cells
        self.failed_cells.add(failed_act[1])
        for agent_affected_actions in affected:
            for act in agent_affected_actions:
                self.affected_cells.add(
                    act[1]
                )
        self.update_failure_log()
        self.draw_grid()


    def mark_conflicting_cells(self):
        for cell in self.conflicting_cells:
            if isinstance(cell, tuple) and len(cell) == 2:
                row, col = cell
                center_x = col * self.cell_size + self.offset_x + self.cell_size / 2
                center_y = row * self.cell_size + self.offset_y + self.cell_size / 2
                self.canvas.create_text(
                    center_x,
                    center_y,
                    text="!",
                    fill="red",
                    font=("Arial", int(self.cell_size * 0.8), "bold"),
                )

    def update_failure_log(self):
        self.failure_log_text.delete(1.0, ctk.END)
        self.failure_log_text.insert(ctk.END, f"Remaining k: {self.k}\n")
        self.failure_log_text.insert(
            ctk.END, f"Failures of each agents: {self.failed_agents}\n"
        )
        self.failure_log_text.insert(ctk.END, "-" * 30 + "\n")
        for event in self.failure_events:
            self.failure_log_text.insert(
                ctk.END,
                f"Step {event['step']} | Agent {event['agent']+1} | Fail: {event['fail_type']}\n",
            )
            self.failure_log_text.insert(
                ctk.END, f"Action: {event['action'][0]} at {event['action'][1]}\n"
            )
            self.failure_log_text.insert(ctk.END, f"Affected Cells:\n")
            for affected_set in event["affected_actions"]:
                for act in affected_set:
                    self.failure_log_text.insert(ctk.END, f" - {act[0]} in {act[1]}\n")
            self.failure_log_text.insert(ctk.END, "-" * 30 + "\n")
