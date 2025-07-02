import os
import customtkinter as ctk
import tkinter.ttk as tctk
import tkinter

from src.utils.map_handler import MAPS_DIR
from src.view.handlers import handle_run_test

def build_layout(root, state):

    ######## MAIN CONTAINER ########
    main_container = ctk.CTkFrame(root)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)

    main_container.grid_rowconfigure(0, weight=0)
    main_container.grid_rowconfigure(1, weight=1)
    main_container.grid_rowconfigure(2, weight=1)
    main_container.grid_columnconfigure(0, weight=2, minsize=500)
    main_container.grid_columnconfigure(1, weight=1)
    main_container.grid_columnconfigure(2, weight=1)

    ## Create instance ##
    crea_frame_container = ctk.CTkFrame(main_container, border_width=1, border_color="#A0A0A0", corner_radius=8)
    crea_frame_container.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=(0, 5), pady=(0, 5))

    ctk.CTkLabel(crea_frame_container, text="Create instance", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    crea_frame = ctk.CTkFrame(crea_frame_container, fg_color="transparent")
    crea_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Coordinates
    coord_frame = ctk.CTkFrame(crea_frame)
    coord_frame.pack(fill="x", pady=(0, 10))
    coord_labels = ["rowS", "colS", "rowG", "colG"]
    state.coord_entries = []

    for i, label in enumerate(coord_labels):
        ctk.CTkLabel(coord_frame, text=label).grid(row=0, column=i, padx=2)
        entry = tctk.Entry(coord_frame, width=8)
        entry.grid(row=1, column=i, padx=2)
        state.coord_entries.append(entry)

    # Add/Remove agent buttons
    buttons_frame = ctk.CTkFrame(crea_frame)
    buttons_frame.pack(fill="x", pady=(5, 10))
    state.add_button = ctk.CTkButton(buttons_frame, text="Add agent")
    state.add_button.pack(side="left", padx=(0, 5))
    state.remove_button = ctk.CTkButton(buttons_frame, text="Remove agent")
    state.remove_button.pack(side="left")

    # Agents list
    state.agent_listbox = tkinter.Listbox(crea_frame, height=8, width=35)
    state.agent_listbox.pack(fill="both", expand=True, pady=(0, 10))

    # Parameters
    settings_frame = ctk.CTkFrame(crea_frame)
    settings_frame.pack(fill="x", pady=(0, 10))

    ctk.CTkLabel(settings_frame, text="Map").grid(row=0, column=0, sticky="w")
    map_list = sorted(os.listdir(MAPS_DIR)) if os.path.exists(MAPS_DIR) else []
    map_combo = tctk.Combobox(settings_frame, textvariable=state.map_var, values=map_list, width=30)
    map_combo.grid(row=0, column=1, padx=(5, 0))

    ctk.CTkLabel(settings_frame, text="total k").grid(row=1, column=0, sticky="w")
    tctk.Entry(settings_frame, textvariable=state.k_var, width=8).grid(row=1, column=1, padx=(5, 0), sticky="w")

    ctk.CTkLabel(settings_frame, text="m: max failable agents").grid(row=2, column=0, sticky="w")
    tctk.Entry(settings_frame, textvariable=state.m_var, width=8).grid(row=2, column=1, padx=(5, 0), sticky="w")


    ctk.CTkLabel(settings_frame, text="h: max faults for each agent").grid(row=3, column=0, sticky="w")
    tctk.Entry(settings_frame, textvariable=state.h_var, width=8).grid(row=3, column=1, padx=(5, 0), sticky="w")


    for i, (text, var) in enumerate(state.fail_type_vars.items()):
        ctk.CTkCheckBox(settings_frame, text=text, variable=var).grid(row=i, column=4, padx=(20, 0), sticky="w")

    # Save instance button
    save_btn_frame = ctk.CTkFrame(crea_frame, fg_color="transparent")
    save_btn_frame.pack(fill="x", pady=(10, 0))
    state.save_instance_btn = ctk.CTkButton(save_btn_frame, text="Save instance")
    state.save_instance_btn.pack()

    ## Load instance ##
    carica_frame_container = ctk.CTkFrame(main_container, border_width=1, border_color="#A0A0A0", corner_radius=10)
    carica_frame_container.grid(row=0, column=1, sticky="new", padx=5, pady=(0, 5))

    ctk.CTkLabel(carica_frame_container, text="Load instance from file", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    carica_frame = ctk.CTkFrame(carica_frame_container, fg_color="transparent")
    carica_frame.pack(fill="both", expand=True, padx=10, pady=10)
    ctk.CTkLabel(carica_frame, text="Load an existing instance.\nOnce loaded, you can edit it or solve it").pack(expand=True, fill="x")
    state.load_instance_btn = ctk.CTkButton(carica_frame, text="Load instance")
    state.load_instance_btn.pack(pady=(10, 0))

    ## Simulate plan ##
    simula_frame_container = ctk.CTkFrame(main_container, border_width=1, border_color="#A0A0A0", corner_radius=10)
    simula_frame_container.grid(row=0, column=2, sticky="new", padx=5, pady=(0, 5))

    ctk.CTkLabel(simula_frame_container, text="Load and simulate plan", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    simula_frame = ctk.CTkFrame(simula_frame_container, fg_color="transparent")
    simula_frame.pack(fill="both", expand=True, padx=10, pady=10)
    ctk.CTkLabel(simula_frame, text="Load a saved plan, previously produced.\nThe simulation window will be displayed").pack(expand=True, fill="x")
    state.simulate_plan_btn = ctk.CTkButton(simula_frame, text="Simulate")
    state.simulate_plan_btn.pack(pady=(10, 0))

    ## Generate test instances ##
    generate_frame_container = ctk.CTkFrame(main_container, border_width=1, border_color="#A0A0A0", corner_radius=10)
    generate_frame_container.grid(row=1, column=1, rowspan=3, columnspan=3, sticky="new", padx=(5, 0), pady=5)

    ctk.CTkLabel(generate_frame_container, text="Generate test instances", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    generate_frame = ctk.CTkFrame(generate_frame_container, fg_color="transparent")
    generate_frame.pack(fill="both", expand=True, padx=10, pady=10)
    gen_grid = ctk.CTkFrame(generate_frame)
    gen_grid.pack(expand=True)

    ctk.CTkLabel(gen_grid, text="Map").grid(row=0, column=0, sticky="w", padx=(0, 10))
    map_list = sorted(os.listdir(MAPS_DIR)) if os.path.exists(MAPS_DIR) else []
    tctk.Combobox(gen_grid, textvariable=state.map_gen_var, values=map_list, width=20).grid(row=0, column=1, padx=(0, 20))

    ctk.CTkLabel(gen_grid, text="# Instances").grid(row=1, column=0, sticky="w", padx=(0, 10))
    tctk.Entry(gen_grid, textvariable=state.num_instances_var, width=10).grid(row=1, column=1, padx=(0, 20))

    ctk.CTkLabel(gen_grid, text="# Agents").grid(row=2, column=0, sticky="w", padx=(0, 10))
    tctk.Entry(gen_grid, textvariable=state.num_agents_var, width=10).grid(row=2, column=1, padx=(0, 20))

    ctk.CTkLabel(gen_grid, text="Min distance").grid(row=3, column=0, sticky="w", padx=(0, 10))
    tctk.Entry(gen_grid, textvariable=state.min_dist_var, width=10).grid(row=3, column=1, padx=(0, 20))

    ctk.CTkLabel(gen_grid, text="Set name").grid(row=4, column=0, sticky="w", padx=(0, 10))
    tctk.Entry(gen_grid, textvariable=state.name_set_var, width=20).grid(row=4, column=1, padx=(0, 20))

    state.generate_instances_btn = ctk.CTkButton(gen_grid, text="Generate instances")
    state.generate_instances_btn.grid(row=5, column=0, columnspan=2, pady=20)

    ## Console output ##
    console_frame_container = ctk.CTkFrame(main_container, border_width=1, border_color="#A0A0A0", corner_radius=10)
    console_frame_container.grid(row=2, column=1, columnspan=3, sticky="nsew", padx=(5, 0), pady=(0, 10))

    ctk.CTkLabel(console_frame_container, text="Console output", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    console_textbox = ctk.CTkTextbox(console_frame_container, height=150, font=("Courier", 10), wrap="word", fg_color="#f0f0f0", border_width=0, corner_radius=8)
    console_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    state.copy_button = ctk.CTkButton(console_frame_container, text="Copy All")
    state.copy_button.place(relx=0.65, rely=0.95, anchor="e")
    state.clear_button = ctk.CTkButton(console_frame_container, text="Clear Console")
    state.clear_button.place(relx=0.98, rely=0.95, anchor="e")

    ## Run/test buttons ##
    buttons_frame = ctk.CTkFrame(main_container)
    buttons_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=(0, 5), pady=(5, 0))

    state.solve_button = ctk.CTkButton(buttons_frame, text="Solve k-R MAPF", font=("Arial", 14, "bold"), height=40)
    state.solve_button.pack(side="left", padx=(10, 5), pady=10, fill="x", expand=True)

    state.run_test_button = ctk.CTkButton(buttons_frame, text="Run Test", font=("Arial", 14, "bold"), height=40)
    state.run_test_button.pack(side="left", padx=5, pady=10, fill="x", expand=True)
    state.run_test_button.configure(command=lambda: handle_run_test(root, state))

    return console_textbox
