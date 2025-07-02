import heapq
from collections import defaultdict
from collections import deque

from copy import deepcopy
import heapq


HIGH_LEVEL_MOVES = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
    (-1, 1): "up_right",
    (-1, -1): "up_left",
    (1, 1): "down_right",
    (1, -1): "down_left",
    (0, 0): "wait",
}
MAX_ITERATIONS = 1000

class CBSNode:
    def __init__(self, constraints=None, solution=None, cost=float("inf")):
        self.constraints = constraints if constraints is not None else set()
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost  # For heapq to sort CBSNodes by cost

    def compute_low_level_solution(self, starts, goals, graph, failed_actions, h_maps):
        paths = []
        total_cost = 0
        for i, start in enumerate(starts):
            goal = goals[i]
            try:
                path, cost = low_level_search_cbs(
                    i,
                    start,
                    goal,
                    self.constraints,
                    graph,
                    failed_actions,
                    h_maps[i],
                    paths=paths,
                )
            except:
                print(
                    f"Low-level search failed for agent {i} with start {start} and goal {goal}"
                )
                return False
            if path is None:
                return False
            paths.append(path)
            total_cost += cost
            print(f"\n\tFound: {path=} with {self.constraints=} for agent {i}")
        self.solution = paths
        self.cost = total_cost
        return True

class SIPPSNode:
    def __init__(
            self,
            v,
            interval,
            interval_id,
            g,
            heuristic_map,
            Os_vertex,
            Os_edge,
            Os_target,
            T,
            Tprime,
            is_goal=False,
            parent=None,
            cfuture=0,
    ):
        self.v = v
        self.low, self.high = interval
        self.interval_id = interval_id
        # arrival time = low
        self.g = g
        self.c = self.compute_c_value(parent, Os_vertex, Os_edge, Os_target, cfuture)

        if is_goal:
            self.h = 0
        else:
            d = heuristic_map[v]
            if self.c == 0:
                # No soft collision
                self.h = max(d, Tprime - self.low)
            else:
                # At least one soft collision
                self.h = max(d, T - self.low)

        self.f = self.g + self.h
        self.is_goal = is_goal
        self.parent = parent

    def __lt__(self, other):
        return (self.c, self.f) < (other.c, other.f)

    def is_identical(self, other):
        return (self.v, self.interval_id, self.is_goal) == (
            other.v,
            other.interval_id,
            other.is_goal,
        )

    def dominates_weakly(self, other):
        if self.is_identical(other):
            if self.low <= other.low and self.high >= other.high:
                if self.c <= other.c:
                    return True
        return False

    def identity(self):
        return (self.v, self.interval_id, self.is_goal)

    def compute_c_value(self, parent, Os_vertex, Os_edge, Os_target, cfuture):
        cv = 0
        ce = 0

        cv = int(
            any(
                t_obs >= self.low and t_obs < self.high
                for (v_obs, t_obs) in Os_vertex.union(Os_target)
                if v_obs == self.v
            )
        )
        ce = int(((parent.v, self.v), self.low) in Os_edge) if parent else 0
        c = (parent.c if parent else 0) + cv + ce + cfuture

        return c


def compute_plan_cbs(starts, goals, failed_actions, states_down, graph):
    open_list = []
    closed_tau_set = set()  # Avoids repeated exploration of tau states
    visited_constraints = set()

    root = CBSNode()

    if tuple(starts) in states_down:
        return None, None

    h_maps = [heuristic(graph, goal) for goal in goals]

    if not root.compute_low_level_solution(
        starts, goals, graph, failed_actions, h_maps
    ):
        return None, None

    heapq.heappush(open_list, root)

    max_iterations = MAX_ITERATIONS
    iteration = 0

    while open_list:
        if iteration < max_iterations:
            iteration += 1

            node = heapq.heappop(open_list)
            pi, tau = build_solution(node)
            tau_key = tuple(map(tuple, tau))

            # Solution already explored
            if tau_key in closed_tau_set:
                continue
            closed_tau_set.add(tau_key)

            conflict = detect_conflict(node.solution)

            if conflict is None:
                found_down_state = False

                # Check if solution contains a down state
                for t, state in enumerate(tau):
                    if state in states_down:
                        found_down_state = True

                        # Add constraints for each agent in the down state
                        for agent, pos in enumerate(state):
                            if (agent, pos, t, "vertex") in node.constraints:
                                continue

                            constraint_key = (agent, pos, t, "vertex")
                            if constraint_key in visited_constraints:
                                continue
                            visited_constraints.add(constraint_key)

                            constraints_new = deepcopy(node.constraints)
                            constraints_new.add(constraint_key)
                            child = CBSNode(constraints_new)

                            if child.compute_low_level_solution(
                                starts, goals, graph, failed_actions, h_maps
                            ):
                                heapq.heappush(open_list, child)

                        break # Only first down state is handled

                if not found_down_state:
                    return pi, tau

            # Conflicts are handled
            else:
                if conflict[-1] == "vertex":
                    ai, aj, vertex, timestep, confl_type = conflict

                    # Child node for ai
                    constraints_ai = deepcopy(node.constraints)
                    constraint_ai = (ai, vertex, timestep, "vertex")
                    constraints_ai.add(constraint_ai)
                    child_ai = CBSNode(constraints_ai)

                    if child_ai.compute_low_level_solution(
                        starts, goals, graph, failed_actions, h_maps
                    ):
                        pi_ai, tau_ai = build_solution(child_ai)
                        tau_ai_key = tuple(map(tuple, tau_ai))
                        if tau_ai_key not in closed_tau_set:
                            heapq.heappush(open_list, child_ai)

                    # Child node for aj
                    constraints_aj = deepcopy(node.constraints)
                    constraint_aj = (aj, vertex, timestep, "vertex")
                    constraints_aj.add(constraint_aj)
                    child_aj = CBSNode(constraints_aj)

                    if child_aj.compute_low_level_solution(
                        starts, goals, graph, failed_actions, h_maps
                    ):
                        pi_aj, tau_aj = build_solution(child_aj)
                        tau_aj_key = tuple(map(tuple, tau_aj))
                        if tau_aj_key not in closed_tau_set:
                            heapq.heappush(open_list, child_aj)
                elif conflict[-1] == "edge":
                    ai, aj, (u, v), timestep, confl_type = conflict

                    # Child node for ai
                    constraints_ai = deepcopy(node.constraints)
                    constraint_ai = (ai, (u, v), timestep, "edge")
                    constraints_ai.add(constraint_ai)
                    child_ai = CBSNode(constraints_ai)

                    if child_ai.compute_low_level_solution(
                        starts, goals, graph, failed_actions, h_maps
                    ):
                        pi_ai, tau_ai = build_solution(child_ai)
                        tau_ai_key = tuple(map(tuple, tau_ai))
                        if tau_ai_key not in closed_tau_set:
                            heapq.heappush(open_list, child_ai)

                    # Child node for aj
                    constraints_aj = deepcopy(node.constraints)
                    constraint_aj = (aj, (v, u), timestep, "edge")
                    constraints_aj.add(constraint_aj)
                    child_aj = CBSNode(constraints_aj)

                    if child_aj.compute_low_level_solution(
                        starts, goals, graph, failed_actions, h_maps
                    ):
                        pi_aj, tau_aj = build_solution(child_aj)
                        tau_aj_key = tuple(map(tuple, tau_aj))
                        if tau_aj_key not in closed_tau_set:
                            heapq.heappush(open_list, child_aj)

        else:
            print("\n\tNo solution found within max iterations.")
            break
    return None, None


def low_level_search_cbs(
    agent_id,
    start,
    goal,
    constraints,
    graph,
    failed_actions,
    heuristic_map,
    max_time=1000,
    paths=None,
):
    Oh_vertex, Oh_edge, Oh_target = set(), set(), set()
    Os_vertex, Os_edge, Os_target = set(), set(), set()

    # Hard constraints are populated
    for c in constraints:
        if c[0] != agent_id:
            continue
        if c[-1] == "vertex":
            Oh_vertex.add((c[1], c[2]))
        elif c[-1] == "edge":
            u, v = c[1]
            t = c[2]
            Oh_edge.add(((u, v), t))

    # Soft constraints are populated
    for path in paths or []:
        for t in range(len(path) - 1):
            Os_edge.add(((path[t], path[t + 1]), t))
        for t, v in enumerate(path):
            Os_vertex.add((v, t))

    # Map modifications due to failed actions
    modified = deepcopy(graph)
    if not modified.is_directed():
        modified = modified.to_directed()
    inv = {v: k for k, v in HIGH_LEVEL_MOVES.items()}
    for act, cell in failed_actions[agent_id]:
        if act not in inv:
            continue
        dr, dc = inv[act]
        to_cell = (cell[0] + dr, cell[1] + dc)
        if modified.has_edge(cell, to_cell):
            modified.remove_edge(cell, to_cell)

    # Construction of safe interval table
    T_table = build_safe_interval_table(
        modified, Oh_vertex, Oh_target, Os_vertex, Os_target, max_time
    )
    if start not in T_table or goal not in modified:
        return None, float("inf")

    hard_times = [t for (v, t) in Oh_vertex | Oh_edge | Oh_target if v == goal]
    T = max(hard_times) + 1 if hard_times else 0
    soft_times = [t for (v, t) in Os_vertex | Os_target if v == goal]
    Tprime = (max(hard_times + soft_times) + 1) if (hard_times or soft_times) else 0

    root_int = T_table[start][0]
    root = SIPPSNode(
        start,
        root_int,
        0,
        root_int[0],
        heuristic_map,
        Os_vertex,
        Os_edge,
        Os_target,
        T,
        Tprime,
    )
    open_list, closed_list = [], []
    heapq.heappush(open_list, root)

    while open_list:
        n = heapq.heappop(open_list)

        if n.is_goal:
            return extract_path(n), n.g
        if n.v == goal and n.low >= T:
            cf = sum(
                1 for t in range(n.low, max_time) if (goal, t) in Os_vertex | Os_target
            )
            if cf == 0:
                return extract_path(n), n.g
            else:
                goal_node = SIPPSNode(
                    goal,
                    (n.low, n.high),
                    n.interval_id,
                    n.low,
                    heuristic_map,
                    Os_vertex,
                    Os_edge,
                    Os_target,
                    T,
                    Tprime,
                    is_goal=True,
                    parent=n,
                    cfuture=cf,
                )
                insert_node(goal_node, open_list, closed_list)

        I = set()
        rng = set(range(n.low + 1, n.high))
        for w in modified.successors(n.v):
            for idx, (lo, hi) in enumerate(T_table[w]):
                if rng & set(range(lo, hi)):
                    I.add((w, idx))
        for idx, (lo, hi) in enumerate(T_table[n.v]):
            if lo == n.high:
                I.add((n.v, idx))

        for v, idx in I:
            lo, hi = T_table[v][idx]

            # Avoid hard constraints
            t_start = max(n.g + 1, lo)
            t_hard = None
            for t in range(t_start, hi):
                if ((n.v, v), t) not in Oh_edge and (v, t) not in Oh_vertex:
                    t_hard = t
                    break
            if t_hard is None:
                continue

            # Avoid soft constraints
            t_soft = None
            for t in range(t_hard, hi):  # t ∈ [t_hard, hi)
                if ((n.v, v), t) not in Os_edge:
                    t_soft = t
                    break
            if t_soft is None:
                continue

            if t_soft > t_hard:
                n1 = SIPPSNode(
                    v,
                    (lo, t_soft),
                    idx,
                    t_hard,
                    heuristic_map,
                    Os_vertex,
                    Os_edge,
                    Os_target,
                    T,
                    Tprime,
                    parent=n,
                )
                insert_node(n1, open_list, closed_list)

                n2 = SIPPSNode(
                    v,
                    (t_soft, hi),
                    idx,
                    t_soft,
                    heuristic_map,
                    Os_vertex,
                    Os_edge,
                    Os_target,
                    T,
                    Tprime,
                    parent=n,
                )
                insert_node(n2, open_list, closed_list)
            else:
                # No soft collision, t_hard is arrival time
                n3 = SIPPSNode(
                    v,
                    (lo, hi),  # [lo, hi)
                    idx,
                    t_hard,  # arrival time reale
                    heuristic_map,
                    Os_vertex,
                    Os_edge,
                    Os_target,
                    T,
                    Tprime,
                    parent=n,
                )
                insert_node(n3, open_list, closed_list)

        closed_list.append(n)
    return None, float("inf")


def build_solution(node):
    # Build joint states and macroactions
    if node.solution is None:
        return [], []

    paths = node.solution
    max_len = max(len(p) for p in paths)
    num_agents = len(paths)

    extended_paths = [path + [path[-1]] * (max_len - len(path)) for path in paths]

    joint_states = []
    for t in range(max_len):
        state_t = tuple(extended_paths[i][t] for i in range(num_agents))
        joint_states.append(state_t)

    return extract_pi_from_tau(joint_states), joint_states


def heuristic(graph, goal):
    h_map = {node: float("inf") for node in graph.nodes}
    h_map[goal] = 0
    heap = [(0, goal)]

    while heap:
        cost, current = heapq.heappop(heap)

        if cost > h_map[current]:
            continue

        for neighbor in graph.neighbors(current):
            edge_weight = graph[current][neighbor].get("weight", 1)
            new_cost = cost + edge_weight

            if new_cost < h_map[neighbor]:
                h_map[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))
    return h_map


def build_safe_interval_table(
    graph, Oh_vertex, Oh_target, Os_vertex, Os_target, max_time=1000
):
    T = defaultdict(list)

    for v in graph.nodes:
        time_status = []
        for t in range(max_time):
            is_hard = (v, t) in Oh_vertex or (v, t) in Oh_target
            is_soft = (v, t) in Os_vertex or (v, t) in Os_target
            time_status.append((is_hard, is_soft))

        t = 0
        while t < max_time:
            if time_status[t][0]:
                t += 1
                continue

            soft_flag = time_status[t][1]
            start = t
            t += 1
            while (
                t < max_time
                and not time_status[t][0]
                and time_status[t][1] == soft_flag
            ):
                t += 1
            end = t
            T[v].append((start, end))

    return dict(T)


def insert_node(n, open_list, closed_dict):
    key = n.identity()
    n_list = set()

    for node in set(open_list).union(closed_dict):
        if n.is_identical(node):
            n_list.add(node)

    for q in n_list:
        if q.dominates_weakly(n):
            return

        elif n.dominates_weakly(q):
            if q in open_list:
                open_list.remove(q)
            if q in closed_dict:
                closed_dict.remove(q)

        elif n.low < q.high and q.low < n.high:
            if n.low < q.low:
                n.high = q.low
            else:
                q.high = n.low

            if n.high <= n.low:
                return

            if q.high <= q.low:
                if q in open_list:
                    open_list.remove(q)
                if q in closed_dict:
                    closed_dict.remove(q)
    heapq.heappush(open_list, n)


def extract_path(n):
    path_nodes = []
    node = n
    while node is not None:
        path_nodes.append(node)
        node = node.parent
    path_nodes.reverse()

    path = []
    current_time = path_nodes[0].low

    for i in range(len(path_nodes)):
        node = path_nodes[i]
        while current_time < node.low:
            path.append(path[-1] if path else node.v)
            current_time += 1

        path.append(node.v)
        current_time += 1

    return path


def extract_pi_from_tau(joint_states):
    macroactions = []

    for t in range(len(joint_states) - 1):
        current = joint_states[t]
        next_ = joint_states[t + 1]
        macroaction = []

        for i in range(len(current)):
            from_cell = current[i]
            to_cell = next_[i]
            action_name = infer_action(from_cell, to_cell)
            macroaction.append((action_name, from_cell))

        macroactions.append(tuple(macroaction))

    return macroactions


def detect_conflict(paths):
    max_len = max(len(p) for p in paths)

    # Vertex conflicts
    for t in range(max_len):
        positions = {}
        for i, path in enumerate(paths):
            pos = path[t] if t < len(path) else path[-1]

            if pos in positions:
                return (positions[pos], i, pos, t, "vertex")
            positions[pos] = i

    # Edge conflicts
    for t in range(max_len - 1):
        for i in range(len(paths)):
            if t + 1 >= len(paths[i]):
                continue

            pos_i_t = paths[i][t]
            pos_i_t1 = paths[i][t + 1]

            for j in range(i + 1, len(paths)):
                if t + 1 >= len(paths[j]):
                    continue

                pos_j_t = paths[j][t]
                pos_j_t1 = paths[j][t + 1]

                # Verifica se si scambiano posizione
                if pos_i_t == pos_j_t1 and pos_i_t1 == pos_j_t:
                    return (i, j, (pos_i_t, pos_i_t1), t, "edge")  # Conflitto di edge

    return None


def infer_action(from_cell, to_cell):
    drow = to_cell[0] - from_cell[0]
    dcol = to_cell[1] - from_cell[1]

    if drow == 0 and dcol == 0:
        return "wait"
    elif drow == 0 and dcol == 1:
        return "right"
    elif drow == 0 and dcol == -1:
        return "left"
    elif drow == 1 and dcol == 0:
        return "down"
    elif drow == -1 and dcol == 0:
        return "up"
    elif drow == 1 and dcol == 1:
        return "down_right"
    elif drow == 1 and dcol == -1:
        return "down_left"
    elif drow == -1 and dcol == 1:
        return "up_right"
    elif drow == -1 and dcol == -1:
        return "up_left"
    else:
        return "invalid"
