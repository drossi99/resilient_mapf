class Solution:
    def __init__(
        self,
        tau_states,
        R_up,
        R_down,
        predecessors,
        resilient_node_macroactions,
        cost=None,
        cbs_cost=None,
    ):
        # self.pi_actions = pi_actions
        self.tau_states = tau_states
        self.R_up = R_up
        self.R_down = R_down
        self.predecessors = predecessors
        self.resilient_node_macroactions = resilient_node_macroactions
        self.resilient_cost = cost
        self.cbs_cost = cbs_cost
