class MAPFInstance:
    def __init__(self, graph, starts, goals):
        self.graph = graph
        self.starts = starts
        self.goals = goals

class RobustnessParams:
    def __init__(self, k, m, h, selected_failure_types):
        self.k = k
        self.m = m
        self.h = h
        self.selected_failure_types = selected_failure_types

class SearchParams:
    def __init__(self, initial_failed_actions, initial_failures, r_up, r_down, predecessors, resilient_node_macroactions):
        self.initial_failed_actions = initial_failed_actions
        self.initial_failures = initial_failures
        self.r_up = r_up
        self.r_down = r_down
        self.predecessors = predecessors
        self.resilient_node_macroactions = resilient_node_macroactions
