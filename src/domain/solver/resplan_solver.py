import networkx as nx
import math
from queue import PriorityQueue

from src.utils.profiling import debug_print
import itertools
from src.domain.solver.Solution import Solution
from src.domain.MAPFInstance import MAPFInstance
from copy import deepcopy
from src.domain.solver.computeplan import compute_plan_cbs

MAX_ITERATIONS = 500


class Node:
    def __init__(self, state, k, failed_actions, failure_agents):
        self.state = state
        self.k = k
        self.failed_actions = tuple(frozenset(s) for s in failed_actions)
        self.failure_agents = tuple(failure_agents)

    def get_signature(self):
        return (self.state, self.k, self.failed_actions, self.failure_agents)

    def __eq__(self, other):
        return isinstance(other, Node) and self.get_signature() == other.get_signature()

    def __hash__(self):
        return hash(self.get_signature())


    def get_failable_agents(self, m, h):
        failable_agents = []
        num_agents = len(self.failure_agents)

        agents_with_failures = sum(
            1 for failures in self.failure_agents if failures > 0
        )

        for i in range(num_agents):
            agent_failures = self.failure_agents[i]
            if agent_failures < h and (agent_failures > 0 or agents_with_failures < m):
                failable_agents.append(i)

        return failable_agents

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

def resplan_mapf(
    mapf_instance,
    robustness_parames,
    search_params,
):
    map_graph = mapf_instance.graph
    starts = tuple(mapf_instance.starts)
    goals = tuple(mapf_instance.goals)

    N = len(starts)
    k = robustness_parames.k
    m = robustness_parames.m
    h = robustness_parames.h
    selected_failure_types = robustness_parames.selected_failure_types

    initial_failed_actions = search_params.initial_failed_actions
    initial_failures = search_params.initial_failures
    R_up = search_params.r_up
    R_down = search_params.r_down
    predecessors = search_params.predecessors
    resilient_node_macroactions = search_params.resilient_node_macroactions

    instance_k = k

    s_0 = tuple(starts)
    starting_failed_agents = initial_failures
    root = Node(s_0, k, initial_failed_actions, initial_failures)
    predecessors.setdefault(root.get_signature(), [])
    non_resilient_plan = None

    open = [root]

    while open:
        current_node = open.pop()
        print("\npopped node:", current_node.get_signature())

        if current_node.get_signature() in {
            n.get_signature() for n in R_up
        } or current_node.get_signature() in {n.get_signature() for n in R_down}:
            print("\nNode already processed")
            continue

        if is_goal(current_node.state, goals):
            R_up.add(current_node)
            print(f"\nNode {current_node.get_signature()}: goal node -> added to R_up")
            continue

        if len(set(current_node.state)) < len(current_node.state):
            R_down.add(current_node)
            print(
                f"\nNode {current_node.get_signature()}: has duplicate states -> added to R_down"
            )
            continue

        success, macroaction = rcheck(
            current_node, R_up, mapf_instance, N, selected_failure_types, m, h
        )
        if success:
            R_up.add(current_node)
            resilient_node_macroactions[current_node.get_signature()] = macroaction
            print(
                f"Node {current_node.get_signature()} is resilient: added to R_up with macroaction {macroaction}"
            )
            continue

        states_down = get_non_resilient_states(R_down)

        pi, tau = compute_plan_cbs(
            tuple(current_node.state),
            goals,
            current_node.failed_actions,
            states_down,
            mapf_instance.graph,
        )

        # No path -> node is non-resilient
        if pi is None:
            R_down.add(current_node)
            print(f"\nNode {current_node.get_signature()} is not resilient: no pi found")
            continue

        if root.get_signature() == current_node.get_signature():
            non_resilient_plan = pi

        parent_node = current_node
        temp_state = current_node.state
        if current_node.k > 0:
            if (
                parent_node.get_signature() not in {n.get_signature() for n in R_up}
                and parent_node.get_signature()
                not in {n.get_signature() for n in R_down}
                and parent_node.get_signature() not in {n.get_signature() for n in open}
            ):
                open.append(parent_node)

            for i in range(1, len(tau)):
                macroaction = pi[i - 1]
                success_node = Node(
                    tau[i],
                    current_node.k,
                    current_node.failed_actions,
                    current_node.failure_agents,
                )
                if (
                    success_node.get_signature()
                    not in {n.get_signature() for n in R_up}
                    and success_node.get_signature()
                    not in {n.get_signature() for n in R_down}
                    and success_node.get_signature()
                    not in {n.get_signature() for n in open}
                ):
                    open.append(success_node)
                predecessors.setdefault(success_node.get_signature(), []).append(
                    (parent_node, macroaction, (None, None))
                )

                for j in current_node.get_failable_agents(m, h):
                    action_to_fail = macroaction[j]
                    if action_to_fail[0] != "wait":
                        for failure_type in selected_failure_types:
                            affected_actions = compute_affected_actions(
                                action_to_fail,
                                j,
                                failure_type,
                                current_node.state,
                                mapf_instance.graph,
                                mapf_instance.goals,
                            )
                            if affected_actions is not None:
                                new_failed_actions = update_failed_actions(
                                    current_node.failed_actions, affected_actions
                                )
                                new_state = list(tau[i])
                                new_state[j] = current_node.state[j]
                                new_state = tuple(new_state)

                                fail_node = Node(
                                    new_state,
                                    current_node.k - 1,
                                    new_failed_actions,
                                    tuple(
                                        (
                                            current_node.failure_agents[i] + 1
                                            if i == j
                                            else current_node.failure_agents[i]
                                        )
                                        for i in range(N)
                                    ),
                                )

                                sig_fail = fail_node.get_signature()
                                predecessors.setdefault(sig_fail, []).append(
                                    (parent_node, macroaction, (j, failure_type))
                                )

                                if sig_fail in {n.get_signature() for n in R_down}:
                                    R_down.add(parent_node)
                                    update_non_resilient_soft(current_node, R_down, k)
                                    print(
                                        f"Node {parent_node} is not resilient: {sig_fail} from {macroaction} already in R_down"
                                    )

                                if (
                                    sig_fail not in {n.get_signature() for n in R_up}
                                    and sig_fail
                                    not in {n.get_signature() for n in R_down}
                                    and sig_fail
                                    not in {n.get_signature() for n in open}
                                ):
                                    open.append(fail_node)

                parent_node = success_node
            last_node = Node(
                tau[len(pi)],
                current_node.k,
                current_node.failed_actions,
                current_node.failure_agents,
            )
            R_up.add(last_node)
            predecessors.setdefault(last_node.get_signature(), []).append(
                (parent_node, pi[-1], (None, None))
            )
        else:
            prev_node = Node(
                tau[0], 0, current_node.failed_actions, current_node.failure_agents
            )
            R_up.add(prev_node)

            for i in range(1, len(tau)):
                new_node = Node(
                    tau[i], 0, current_node.failed_actions, current_node.failure_agents
                )
                R_up.add(new_node)
                predecessors.setdefault(new_node.get_signature(), []).append(
                    (prev_node, pi[i - 1], (None, None))
                )
                prev_node = new_node

    print("\n--- Search Completed. Extracting Solution ---")
    print(f"\nSize of R_up: {len(R_up)}")
    print(f"\nSize of R_down: {len(R_down)}")
    print(f"\nSize of predecessors: {len(predecessors)}")

    initial_node_sig = Node(
        s_0, instance_k, initial_failed_actions, initial_failures
    ).get_signature()
    print(f"\nInitial node signature: {initial_node_sig}")

    start_node_in_R_up = False
    r_up_signatures = {n.get_signature() for n in R_up}
    if initial_node_sig in r_up_signatures:
        start_node_in_R_up = True


    if start_node_in_R_up:
        print("\nInitial node is in R_up.")
        goal_nodes = get_goal_nodes(R_up, goals)
        print(f"\nFound {len(goal_nodes)} goal nodes in R_up.")

        if not goal_nodes:
            print(
                "Starting node is resilient, but no goal nodes found."
            )
            return Solution(
                None, R_up, R_down, predecessors, resilient_node_macroactions
            )

        print(f"\nPlan extraction attempt")
        pi_k_resilient = extract_solution_from_predecessors(
            goals,
            predecessors,
            initial_node_sig,
            R_up,
            resilient_node_macroactions,
            mapf_instance,
        )

        if pi_k_resilient is not None:
            print("\nResilient plan found:")
            if not pi_k_resilient and is_goal(s_0, goals):
                print("\n\tStart node is the goal node)")
            elif pi_k_resilient:
                for step, macroaction in enumerate(pi_k_resilient):
                    print(f"\n  Step {step}: {macroaction}")

            return Solution(
                pi_k_resilient,
                R_up,
                R_down,
                predecessors,
                resilient_node_macroactions,
                compute_plan_cost_by_macroactions(pi_k_resilient, mapf_instance.graph),
                compute_plan_cost_by_macroactions(
                    non_resilient_plan, mapf_instance.graph
                ),
            )
        else:
            print(
                "No plan extracted"
            )
            return Solution(
                None, R_up, R_down, predecessors, resilient_node_macroactions
            )

    else:
        print(
            "Starting node is not resilient, no plan extracted"
        )
        return Solution(None, R_up, R_down, predecessors, resilient_node_macroactions)


def get_goal_nodes(R_up, goals):
    goal_nodes = []
    for node in R_up:
        if is_goal(node.state, goals):
            goal_nodes.append(node)
    return goal_nodes

def extract_solution_from_predecessors(
    goals,
    predecessors,
    initial_node_sig,
    R_up,
    resilient_node_macroactions,
    mapf_instance,
):
    plan = []
    r_up_signatures = {n.get_signature() for n in R_up}

    while not is_goal(initial_node_sig[0], goals):
        found = False

        # Use macroactions from rcheck
        if initial_node_sig in resilient_node_macroactions:
            macroaction = resilient_node_macroactions[initial_node_sig]
            new_state = try_apply_macroaction(
                initial_node_sig[0], macroaction, mapf_instance.graph
            )
            new_node = Node(
                new_state, initial_node_sig[1], initial_node_sig[2], initial_node_sig[3]
            )

            if new_node.get_signature() in r_up_signatures:
                plan.append(macroaction)
                initial_node_sig = new_node.get_signature()
                found = True

        # Otherwise look for predecessors
        if not found:
            for child, parents in predecessors.items():
                for node, macroaction, failure_info in parents:
                    if node.get_signature() == initial_node_sig:
                        if failure_info == (None, None) and child in r_up_signatures:
                            plan.append(macroaction)
                            initial_node_sig = child
                            found = True
                            break
                if found:
                    break

        if not found:
            print(f"\nERROR: No resilient path found from {initial_node_sig}")
            return None

    return plan

def get_non_resilient_states(R_down):
    non_resilient_states = set()
    for r_down_node in R_down:
        non_resilient_states.add(r_down_node.state)
    return non_resilient_states

def is_goal(state, goals):
    return tuple(state) == tuple(goals)

def rcheck(current_node, R_up, mapf_instance, N, selected_failure_types, m, h):
    r_up_signatures = {n.get_signature() for n in R_up}

    for macroaction in compute_applicable_macroactions(
        current_node.state, current_node.failed_actions, mapf_instance.graph
    ):
        success_state = try_apply_macroaction(
            current_node.state, macroaction, mapf_instance.graph
        )
        success_node = Node(
            success_state,
            current_node.k,
            current_node.failed_actions,
            current_node.failure_agents,
        )

        if success_node.get_signature() not in r_up_signatures:
            continue

        all_failures_ok = True

        if current_node.k > 0:
            for j in current_node.get_failable_agents(m, h):
                for fail_type in selected_failure_types:
                    action_to_fail = macroaction[j]
                    if action_to_fail[0] != "wait":
                        affected_actions = compute_affected_actions(
                            action_to_fail,
                            j,
                            fail_type,
                            current_node.state,
                            mapf_instance.graph,
                            mapf_instance.goals,
                        )
                        if affected_actions is not None:
                            new_failed_actions = update_failed_actions(
                                current_node.failed_actions, affected_actions
                            )
                            new_state = list(success_state)
                            new_state[j] = current_node.state[j]
                            new_state = tuple(new_state)
                            fail_node = Node(
                                new_state,
                                current_node.k - 1,
                                new_failed_actions,
                                tuple(
                                    (
                                        current_node.failure_agents[i] + 1
                                        if i == j
                                        else current_node.failure_agents[i]
                                    )
                                    for i in range(N)
                                ),
                            )

                            if fail_node.get_signature() not in r_up_signatures or len(
                                set(new_state)
                            ) < len(new_state):
                                all_failures_ok = False

                if not all_failures_ok:
                    break
        if all_failures_ok:
            return True, macroaction

    return False, None


def compute_applicable_highlevel_actions(state, G):
    applicable_highlevel_actions = tuple(set() for _ in range(len(state)))

    for i, pos in enumerate(state):
        for neighbor in G.neighbors(pos):
            dr = neighbor[0] - pos[0]
            dc = neighbor[1] - pos[1]
            move = HIGH_LEVEL_MOVES.get((dr, dc))
            if move:
                applicable_highlevel_actions[i].add(move)

        applicable_highlevel_actions[i].add("wait")

    return applicable_highlevel_actions


def compute_applicable_macroactions(state, failed_actions, G):
    applicable_highlevel_actions = compute_applicable_highlevel_actions(state, G)
    applicable_lowlevel_actions = compute_applicable_lowlevel_actions(
        state, applicable_highlevel_actions, failed_actions
    )

    applicable_macroactions = set()

    # cartesian product for low-level actions
    for macroaction in itertools.product(*applicable_lowlevel_actions):
        applicable_macroactions.add(macroaction)

    return applicable_macroactions

def compute_applicable_lowlevel_actions(
    state, applicable_highlevel_actions, failed_actions
):
    applicable_lowlevel_actions = tuple(set() for _ in range(len(state)))

    for i in range(len(state)):
        row, col = state[i]
        for action in applicable_highlevel_actions[i]:
            new_lowlevel_action = (action, (row, col))
            if new_lowlevel_action not in failed_actions[i]:
                applicable_lowlevel_actions[i].add(new_lowlevel_action)

    return applicable_lowlevel_actions

def try_apply_macroaction(current_state, macroaction, G):
    next_state_list = list(current_state)
    num_agents = len(current_state)

    for i in range(num_agents):
        action_str, agent_pos = macroaction[i]
        if agent_pos != current_state[i]:
            print(
                f"ERROR: Mismatch in apply macroaction pos {agent_pos} vs state {current_state[i]}"
            )
            return None

        if action_str == "wait":
            next_state_list[i] = current_state[i]
        else:
            delta = next(k for k, v in HIGH_LEVEL_MOVES.items() if v == action_str)
            target_pos = (
                current_state[i][0] + delta[0],
                current_state[i][1] + delta[1],
            )

            if G.has_node(target_pos):
                next_state_list[i] = target_pos
            else:
                print(
                    f"WARNING: Invalid move attempted in try_apply_macroaction for agent {i}: {action_str} from {agent_pos} to {target_pos}"
                )
                return None

    return tuple(next_state_list)

def update_failed_actions(current_failed, affected_actions):
    new_failed = list(current_failed)  # list for modification
    for i in range(len(current_failed)):
        new_failed[i] = current_failed[i].union(set(affected_actions[i]))
    return tuple(new_failed)


def compute_all_macroactions(possible_individual_actions):
    return list(itertools.product(*possible_individual_actions))


def compute_macroaction_cost(macroaction, G):
    cost = 0
    for action in macroaction:
        if action[0] == "wait":
            continue
        if "_" in action[0]:
            cost += math.sqrt(2)
        else:
            cost += 1
    return cost


def compute_plan_cost_by_macroactions(pi, G):
    if not pi:
        return 0
    total_cost = 0
    for macroaction in pi:
        total_cost += compute_macroaction_cost(macroaction, G)
    return total_cost



def compute_affected_actions(
    failed_action, failed_agent_index, fail_type, current_state, G, goals
):
    num_agents = len(current_state)
    affected = [set() for _ in range(num_agents)]
    a_H, from_pos = failed_action

    delta = next((d for d, name in HIGH_LEVEL_MOVES.items() if name == a_H), None)
    if delta is None:
        return tuple(affected)

    to_pos = (from_pos[0] + delta[0], from_pos[1] + delta[1])

    if fail_type in {"tops"} and to_pos in goals:
        return tuple([set() for _ in range(num_agents)])

    if fail_type == "topw":
        for i in range(num_agents):
            affected[i].add(failed_action)

    elif fail_type == "individual":
        affected[failed_agent_index].add(failed_action)

    elif fail_type == "high-level":
        if delta:
            for node in G.nodes:
                to_node = (node[0] + delta[0], node[1] + delta[1])
                if to_node in G[node]:
                    affected[failed_agent_index].add((a_H, node))

    elif fail_type == "tops":
        for u in G.nodes:
            for v in G.neighbors(u):
                dr, dc = v[0] - u[0], v[1] - u[1]
                move_str = HIGH_LEVEL_MOVES.get((dr, dc))
                if move_str is None:
                    continue
                if u == to_pos or v == to_pos:
                    for i in range(num_agents):
                        affected[i].add((move_str, u))
        for i in range(num_agents):
            affected[i].add(failed_action)

    return tuple(affected)


def update_non_resilient_soft(current_node, R_down, max_k):
    state = current_node.state
    k = current_node.k
    failed_actions = current_node.failed_actions
    failure_agents = current_node.failure_agents
    num_agents = len(failed_actions)
    r_down_signatures = {n.get_signature() for n in R_down}

    if current_node.get_signature() not in r_down_signatures:
        R_down.add(current_node)
        print(f"\nAdded non-resilient node: {current_node.get_signature()}")

    r_down_signatures = {n.get_signature() for n in R_down}
    for agent_idx in range(num_agents):
        failures = failed_actions[agent_idx]
        for action_to_remove in failures:
            new_failed = list(failed_actions)
            new_failed[agent_idx] = new_failed[agent_idx] - {action_to_remove}
            new_failed_tuple = tuple(frozenset(s) for s in new_failed)
            new_k = k + 1

            if new_k <= max_k:
                new_node = Node(state, new_k, new_failed_tuple, failure_agents)
                if new_node.get_signature() not in r_down_signatures:
                    R_down.add(new_node)
                    print(
                        f"Added non-resilient node: {new_node.get_signature()} with reduced failure {action_to_remove} for agent {agent_idx}"
                    )