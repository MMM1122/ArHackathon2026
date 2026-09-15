"""Amazon Robotics Hackathon: coordinated, bounded fleet lookahead.

Team name: Danger Stranger
Email address: testing

Only this file is needed. No files, future arrivals, or engine internals are
read by the driver. Fill in the two identity fields before submitting.

The dispatcher combines cached Dijkstra distances, FIFO-aware batch assignment,
discounted delivery rewards, and a beam search over legal fleet moves. Its
small simulator reserves destination slots at DEPARTURE and commits moves in
unit-ID order, just like the engine. Only the next move is executed; new pods
and the actual floor state are incorporated at the next time step.
"""

import heapq
import itertools
import math
import time
from typing import Optional

from ar_hackathon.models.graph_state import GraphState


# Both limits apply: deterministic work cap for ordinary cases, wall-clock
# guard for larger inputs. The engine's one-second timeout is not our budget.
_SEARCH_SECONDS = 0.035
_MAX_EXPANSIONS = 600
_BEAM_WIDTH = 6
_SEARCH_DEPTH = 9
_JOINT_LIMIT = 24
_INF = float("inf")
_cache = None


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class _Graph:
    def __init__(self, state, signature):
        self.signature = signature
        self.ids = [n.id for n in state.nodes]
        self.index = {node: i for i, node in enumerate(self.ids)}
        self.cap = [n.capacity for n in state.nodes]
        self.kind = [n.node_type for n in state.nodes]
        self.adj = [[] for _ in self.ids]
        self.edge = {}
        self.edge_cap = []
        for e in state.edges:
            a, b = self.index[e.from_node], self.index[e.to_node]
            eid = len(self.edge_cap)
            self.edge_cap.append(e.capacity)
            # The engine subtracts one per tick, including the departure tick.
            duration = max(1, int(math.ceil(e.weight)))
            pairs = [(a, b)] + ([(b, a)] if e.bidirectional else [])
            for u, v in pairs:
                if (u, v) not in self.edge:  # get_edge uses the first match.
                    self.edge[u, v] = (eid, duration)
                    self.adj[u].append((v, eid, duration))
        self.dist = {}
        self.moves = {}
        self.last_time = -1
        self.last_id = None

    def distances(self, start):
        if start not in self.dist:
            d = [_INF] * len(self.ids)
            d[start] = 0
            heap = [(0, start)]
            while heap:
                cost, u = heapq.heappop(heap)
                if cost != d[u]:
                    continue
                for v, _, duration in self.adj[u]:
                    new = cost + duration
                    if new < d[v]:
                        d[v] = new
                        heapq.heappush(heap, (new, v))
            self.dist[start] = d
        return self.dist[start]


class _Planner:
    def __init__(self, graph, state):
        self.g = graph
        self.now = state.current_time_step
        robots = sorted(state.drive_units, key=lambda u: u.id)
        self.robot_ids = [u.id for u in robots]
        self.capacity = [u.capacity for u in robots]
        pods = sorted(state.active_pods, key=lambda p: (p.entry_time, p.id))
        pindex = {p.id: i for i, p in enumerate(pods)}
        self.dest = [graph.index[p.destination_station] for p in pods]
        self.reward = [math.exp(-(self.now - p.entry_time) / 50.0)
                       for p in pods]
        self.travel_price = sum(self.reward) * 0.000001
        waiting = [0] * len(graph.ids)
        for i, p in enumerate(pods):
            if p.carried_by is None and p.current_node is not None:
                waiting[graph.index[p.current_node]] |= 1 << i
        units = []
        for u in robots:
            units.append((graph.index[u.current_node],
                          graph.index[u.transit_destination] if u.in_transit else -1,
                          max(1, int(math.ceil(u.transit_remaining_time)))
                          if u.in_transit else 0,
                          sum(1 << pindex[p] for p in u.carrying if p in pindex)))
        # World: (relative next tick, units, waiting masks, earned reward, travel).
        # Unit: (source/current node, inbound destination, remaining ticks, cargo).
        self.root = (0, tuple(units), tuple(waiting), 0.0, 0)
        self.route_cache = {}
        self.estimate_cache = {}

    def _route(self, start, mask):
        """Small exact delivery-order search; bounded greedy order for big loads."""
        key = start, mask
        if key in self.route_cache:
            return self.route_cache[key]
        grouped = {}
        for p in _bits(mask):
            dest = self.dest[p]
            grouped[dest] = grouped.get(dest, 0.0) + self.reward[p]
        if not grouped:
            return 0.0, 0, start, start
        if len(grouped) <= 4:
            orders = itertools.permutations(sorted(grouped))
        else:
            left = set(grouped)
            order, u = [], start
            while left:
                d = self.g.distances(u)
                v = max(sorted(left), key=lambda x: grouped[x]
                        * math.exp(-d[x] / 50.0) / max(1, d[x]))
                left.remove(v)
                order.append(v)
                u = v
            orders = [order]
        best = None
        for order in orders:
            elapsed, value, u = 0, 0.0, start
            for v in order:
                elapsed += self.g.distances(u)[v]
                value += grouped[v] * math.exp(-max(0, elapsed - 1) / 50.0)
                u = v
            candidate = (value, -elapsed, -order[0])
            if best is None or candidate > best[0]:
                best = (candidate, (value, elapsed, u, order[0]))
        result = best[1]
        self.route_cache[key] = result
        return result

    def _parking(self, start):
        """Stage near storage, preferring places without a node capacity limit."""
        d = self.g.distances(start)
        storage = [i for i, kind in enumerate(self.g.kind) if kind == "storage"]
        staging = [i for i in storage if self.g.cap[i] is None and d[i] < _INF]
        if staging:
            return min(staging, key=lambda i: (d[i], i))
        candidates = [i for i in range(len(d))
                      if self.g.cap[i] is None and self.g.kind[i] != "station"
                      and d[i] < _INF]
        if not candidates:
            candidates = [i for i in range(len(d))
                          if self.g.kind[i] != "station" and d[i] < _INF]
        if not candidates:
            return start
        return min(candidates, key=lambda i: (
            d[i] + 1.1 * min((self.g.distances(i)[s] for s in storage), default=0),
            self.g.kind[i] != "storage", i))

    def _earliest(self, units, robot, start, available):
        """Earliest travel using current reservations and estimated releases.

        A stationary blocker needs at least one tick to yield. Charging that
        delay in the terminal forecast prevents the search from preferring an
        imaginary immediate dock entry over actually moving its blocker away.
        """
        node_release = [[] for _ in self.g.ids]
        edge_release = [[] for _ in self.g.edge_cap]
        for i, (pos, dest, rem, _) in enumerate(units):
            if i == robot:
                continue
            node_release[dest if rem else pos].append(rem + 1 if rem else 1)
            if rem:
                edge_release[self.g.edge[pos, dest][0]].append(rem)
        nf, ef = [], []
        for releases, capacity in zip(node_release, self.g.cap):
            nf.append(sorted(releases)[len(releases) - capacity]
                      if capacity is not None and len(releases) >= capacity else 0)
        for releases, capacity in zip(edge_release, self.g.edge_cap):
            ef.append(sorted(releases)[len(releases) - capacity]
                      if capacity is not None and len(releases) >= capacity else 0)
        d = [_INF] * len(self.g.ids)
        d[start] = available
        heap = [(available, start)]
        while heap:
            elapsed, u = heapq.heappop(heap)
            if elapsed != d[u]:
                continue
            for v, eid, duration in self.g.adj[u]:
                arrival = max(elapsed, ef[eid], nf[v]) + duration
                if arrival < d[v]:
                    d[v] = arrival
                    heapq.heappush(heap, (arrival, v))
        return d

    def estimate(self, world):
        """Score a projected full schedule, and assign each robot its first goal.

        Assign FIFO batches, not arbitrary pods: automatic pickups do not let us
        choose a later pod at the same source. Later trips account for robot
        availability so one idle robot cannot serve every queue simultaneously.
        This forecast ignores future traffic; the search resolves near traffic.
        """
        tick, units, waiting, earned, travel = world
        key = tick, units, waiting
        cached = self.estimate_cache.get(key)
        if cached is not None:
            future, goals = cached
            return earned + future - travel * self.travel_price, goals
        schedule, goals = [], []
        future = 0.0
        for i, (pos, dest, rem, cargo) in enumerate(units):
            start = dest if rem else pos
            value, duration, end, first = self._route(start, cargo)
            delay = 0
            if cargo and duration < _INF:
                delay = self._earliest(units, i, start, rem)[first] - rem - self.g.distances(start)[first]
            future += value * math.exp(-(tick + rem + delay) / 50.0)
            schedule.append([tick + rem + delay + duration, end])
            goals.append(first if cargo else None)
        pending = list(waiting)
        while any(pending):
            best = None
            for source, mask in enumerate(pending):
                if not mask:
                    continue
                for i, (available, start) in enumerate(schedule):
                    distance = self.g.distances(start)[source]
                    if distance == _INF:
                        continue
                    batch = 0
                    for p in itertools.islice(_bits(mask), self.capacity[i]):
                        batch |= 1 << p
                    value, duration, end, _ = self._route(source, batch)
                    if duration == _INF:
                        continue
                    arrival = available + distance
                    score = value * math.exp(-arrival / 50.0)
                    # A small fixed service overhead balances short trips and
                    # worthwhile batches; availability includes earlier jobs.
                    priority = score / max(4, arrival - tick + duration)
                    candidate = (priority, -arrival, -i, -source)
                    if best is None or candidate > best[0]:
                        best = candidate, i, source, batch, score, arrival + duration, end
            if best is None:
                break  # Unreachable tasks must not trap the dispatcher.
            _, i, source, batch, score, finish, end = best
            future += score
            pending[source] ^= batch
            if goals[i] is None:
                goals[i] = source
            schedule[i] = [finish, end]
        for i, unit in enumerate(units):
            if goals[i] is None:
                goals[i] = self._parking(unit[1] if unit[2] else unit[0])
        result = future, tuple(goals)
        self.estimate_cache[key] = result
        return earned + future - travel * self.travel_price, result[1]

    def _occupancy(self, units):
        nodes = [0] * len(self.g.ids)
        edges = [0] * len(self.g.edge_cap)
        for pos, dest, rem, _ in units:
            nodes[dest if rem else pos] += 1
            if rem:
                edges[self.g.edge[pos, dest][0]] += 1
        return nodes, edges

    def _choices(self, i, units, goal, nodes, edges):
        pos, _, rem, cargo = units[i]
        if rem:
            return [(0.0, None)]
        # A pod picked up at its own destination must remain for delivery.
        if any(self.dest[p] == pos for p in _bits(cargo)):
            return [(0.0, None)]
        old = self.g.distances(pos)[goal]
        choices = []
        for v, eid, duration in self.g.adj[pos]:
            if self.g.edge_cap[eid] is not None and edges[eid] >= self.g.edge_cap[eid]:
                continue
            if self.g.cap[v] is not None and nodes[v] >= self.g.cap[v]:
                continue
            distance = self.g.distances(v)[goal]
            extra = duration + distance - old if old < _INF else duration
            # Keep unreachable detours available only as last-resort yielding.
            if not math.isfinite(extra):
                extra = 1000000.0
            choices.append((extra + duration * 0.001, v))
        choices.sort()
        # Include waiting even if several moves look attractive. A dock slot
        # cannot be assumed to clear later in this callback round.
        wait_cost = 0.7 if pos != goal else 0.0
        if self.g.kind[pos] == "station" and not cargo:
            wait_cost = 3.0
        return sorted(choices[:3] + [(wait_cost, None)],
                      key=lambda x: (x[0], -1 if x[1] is None else x[1]))

    def _service(self, units, waiting, tick, earned):
        units, waiting = list(units), list(waiting)
        for i, (pos, dest, rem, cargo) in enumerate(units):
            if rem:
                continue
            for p in list(_bits(cargo)):
                if self.dest[p] == pos:
                    cargo ^= 1 << p
                    earned += self.reward[p] * math.exp(-tick / 50.0)
            free = self.capacity[i] - bin(cargo).count("1")
            for p in itertools.islice(_bits(waiting[pos]), free):
                waiting[pos] ^= 1 << p
                cargo |= 1 << p
            units[i] = (pos, dest, rem, cargo)
        return tuple(units), tuple(waiting), earned

    def advance(self, world, units, added_travel):
        tick, _, waiting, earned, travel = world
        # Jump over ticks without decisions, but never over an idle robot's
        # opportunity to act. There are no assumed future pod arrivals.
        delta = min(u[2] for u in units) if all(u[2] for u in units) else 1
        moved = []
        for pos, dest, rem, cargo in units:
            if rem and rem <= delta:
                moved.append((dest, -1, 0, cargo))
            elif rem:
                moved.append((pos, dest, rem - delta, cargo))
            else:
                moved.append((pos, dest, rem, cargo))
        units, waiting, earned = self._service(moved, waiting, tick + delta - 1, earned)
        # At the next tick the engine services standing robots before polling.
        units, waiting, earned = self._service(units, waiting, tick + delta, earned)
        return tick + delta, units, waiting, earned, travel + added_travel

    def successors(self, world, limit):
        _, units, _, _, _ = world
        _, goals = self.estimate(world)
        nodes, edges = self._occupancy(units)
        partial = [(0.0, units, nodes, edges, (), 0)]
        for i in range(len(units)):
            expanded = []
            for cost, us, ns, es, actions, travel in partial:
                for penalty, move in self._choices(i, us, goals[i], ns, es):
                    if move is None:
                        expanded.append((cost + penalty, us, ns, es,
                                         actions + (None,), travel))
                    else:
                        pos, _, _, cargo = us[i]
                        eid, duration = self.g.edge[pos, move]
                        new_units, new_nodes, new_edges = list(us), ns[:], es[:]
                        new_units[i] = (pos, move, duration, cargo)
                        new_nodes[pos] -= 1
                        new_nodes[move] += 1
                        new_edges[eid] += 1
                        expanded.append((cost + penalty, tuple(new_units), new_nodes,
                                         new_edges, actions + (move,), travel + duration))
            partial = heapq.nsmallest(limit, expanded, key=lambda x: x[0])
        for _, us, _, _, actions, travel in partial:
            yield self.advance(world, us, travel), actions

    def plan(self):
        deadline = time.perf_counter() + _SEARCH_SECONDS
        # Always prepare a legal coordinated fallback before doing any search.
        fallback_world, fallback = next(self.successors(self.root, 1))
        if not self.reward or _MAX_EXPANSIONS <= 0:
            return fallback
        beam = [(self.root, None)]
        answer = fallback
        expanded = 0
        for _ in range(_SEARCH_DEPTH):
            candidates = {}
            for world, first in beam:
                for successor, actions in self.successors(world, _JOINT_LIMIT):
                    value, _ = self.estimate(successor)
                    initial = actions if first is None else first
                    key = successor[:3]
                    old = candidates.get(key)
                    if old is None or value > old[0]:
                        candidates[key] = (value, successor, initial)
                    expanded += 1
                    if expanded >= _MAX_EXPANSIONS or time.perf_counter() >= deadline:
                        break
                if expanded >= _MAX_EXPANSIONS or time.perf_counter() >= deadline:
                    break
            if not candidates:
                break
            best = heapq.nlargest(_BEAM_WIDTH, candidates.values(), key=lambda x: x[0])
            answer = best[0][2]
            beam = [(world, initial) for _, world, initial in best]
            if expanded >= _MAX_EXPANSIONS or time.perf_counter() >= deadline:
                break
        # Compare against a complete greedy continuation as well. An optimistic
        # terminal estimate alone can prefer a queue that never actually clears.
        if answer != fallback:
            if self._rollout(answer) <= self._rollout(fallback):
                answer = fallback
        # With no moving robots or future arrivals in the observable state,
        # total inaction cannot release a blocked aisle or dock.
        if (all(move is None for move in answer)
                and not any(u[2] for u in self.root[1])
                and any(move is not None for move in fallback)):
            answer = fallback
        return answer

    def _rollout(self, actions):
        units, travel = list(self.root[1]), 0
        for i, move in enumerate(actions):
            if move is not None:
                pos, _, _, cargo = units[i]
                _, duration = self.g.edge[pos, move]
                units[i] = (pos, move, duration, cargo)
                travel += duration
        world = self.advance(self.root, tuple(units), travel)
        for _ in range(28):
            if not any(world[2]) and not any(u[3] for u in world[1]):
                break
            world, _ = next(self.successors(world, 1))
        return self.estimate(world)[0]


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """Return one adjacent node ID, or None, using only the provided state."""
    global _cache
    signature = (tuple((n.id, n.capacity, n.node_type) for n in state.nodes),
                 tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional)
                       for e in state.edges),
                 tuple(sorted((u.id, u.capacity) for u in state.drive_units)))
    now = state.current_time_step
    if (_cache is None or _cache.signature != signature
            or now < _cache.last_time
            or (now == _cache.last_time and _cache.last_id is not None
                and drive_unit_id <= _cache.last_id)):
        _cache = _Graph(state, signature)
    if now != _cache.last_time:
        planner = _Planner(_cache, state)
        actions = planner.plan()
        _cache.moves = {uid: _cache.ids[v] if v is not None else None
                        for uid, v in zip(planner.robot_ids, actions)}
        _cache.last_time = now
    _cache.last_id = drive_unit_id
    move = _cache.moves.get(drive_unit_id)
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit or move is None:
        return None
    # Earlier robot callbacks have already committed their moves. Never return
    # a stale reservation as though it were an available edge or destination.
    edge = state.get_edge(unit.current_node, move)
    node = state.get_node(move)
    if edge is None or (edge.capacity is not None and
                        state.edge_occupancy(unit.current_node, move) >= edge.capacity):
        return None
    if node is not None and node.capacity is not None and state.node_occupancy(move) >= node.capacity:
        return None
    return move
