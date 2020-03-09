import argparse
import json
import networkx
import re
import networkx.algorithms.flow
import copy
import ics
from collections import defaultdict
import arrow
from copy import deepcopy
from pathlib import Path
import math
from collections import deque

global HUBS
global NEARNESS_MAP
global DAYS_OF_THE_WEEK
global WEEKLY_LIMIT
global DAILY_LIMIT

HUBS = []
NEARNESS_MAP = []
DAYS_OF_THE_WEEK = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
WEEKLY_LIMIT = 7
DAILY_LIMIT = 1

conflict_information = {}

def get_neighbors_of_avail(graph, node, next_conflict):
    nodes = graph.nodes()
    location = conflict_information[node]['location']
    nodes = [n for n in nodes if 'slot' in n]
    if next_conflict:
        nodes = [n for n in nodes if (n[0] in NEARNESS_MAP[location] 
                                   or n[0] in NEARNESS_MAP[next_conflict])]
    else:
        nodes = [n for n in nodes if n[0] in NEARNESS_MAP[location]]
    return [n for n in nodes if node[1] == n[1]]


def save_conflict_information(conflict_node, free_slot):
    conflict_before = free_slot[0]
    time_when_available = free_slot[1]
    location = free_slot[2]
    amt_free_time = free_slot[3]
    conflict_information[conflict_node] = {
                        'conflict_before':conflict_before,
                        'time_when_available':time_when_available,
                        'location':location,
                        'amt_free_time':amt_free_time
                        }


hours_min_pat = re.compile('((?P<hours>\d*)h)?((?P<minutes>\d*)m)?')
def interpret_hours_minutes(hour_minutes_string):
    match = hours_min_pat.match(hour_minutes_string)
    assert match
    hours = match.group('hours')
    minutes = match.group('minutes')
    return (int(hours)*60 if hours else 0) + (int(minutes) if minutes else 0)


def make_graph(json, filter_days=()):
    graph = networkx.DiGraph()
    graph.add_node("source")
    graph.add_node("sink")
    slots = [
        (hub, day, 'slot') 
        for day in DAYS_OF_THE_WEEK 
        for hub in HUBS 
        if day not in filter_days
    ] # Generate slots
    graph.add_nodes_from(slots)                                                             # Add slots to graph
    graph.add_edges_from( [(slot, 'sink', {'capacity' : 1}) for slot in slots] )            # Add edges from slots to sink
    graph.add_nodes_from(json.keys())                                                       # Add member nodes
    graph.add_edges_from([ ('source', member, {'capacity' : WEEKLY_LIMIT})
                            for member in json.keys() ])                                    # Add edges for source to members

    for member, daily_availability in json.items():
        for day in daily_availability:
            for free_slot in daily_availability[day]:
                if free_slot == []:                             # Allows some more freedom of inputs
                    break
                graph.add_node((member, day, 'day'))
                graph.add_edge(member, (member, day, 'day'), capacity=DAILY_LIMIT)
                time_when_available = free_slot[1]
                amt_free_time = interpret_hours_minutes(free_slot[3])
                next_conflict = len(free_slot) == 5 and free_slot[4]
                if amt_free_time >= 35:
                    amt_of_locations = int(amt_free_time/35)
                    new_node = (member, day, time_when_available, 'avail')
                    graph.add_node(new_node)
                    save_conflict_information(new_node, free_slot)  
                    graph.add_edge((member, day, 'day'), new_node, capacity=amt_of_locations)
                    neighbors = get_neighbors_of_avail(graph, new_node, next_conflict)
                    graph.add_edges_from( [(new_node, neighbor, {'capacity' : 1}) for neighbor in neighbors] )
    return graph


def generate_visualization(g):
    with open('vis.dot', 'w+') as f:
        f.write('digraph{\n')
        for node in g.nodes():
            out = ("".join(node) if type(node) != str else node).replace(" ", "").replace(":","")
            f.write(str(out) + '[label="'+str(node)+'"]\n')
        for edge in g.edges():
            f.write(
                (str(edge[0]) if type(edge[0]) == str else "".join(edge[0])).replace(" ", "").replace(":","")
                + ' -> ' +
                (str(edge[1]) if type(edge[1]) == str else "".join(edge[1])).replace(" ", "").replace(":","") + '\n')
        f.write('}')


def compute_assignments(graph):
    _, flow_dict = networkx.algorithms.flow.maximum_flow(graph,'source','sink')
    flow_dict = {k:v for k,v in flow_dict.items() if 'avail' in k}
    assignments = {node : {k:v for k,v in dctnry.items() if v != 0} for node, dctnry in flow_dict.items()}
    assignments = {k:v for k,v in assignments.items() if v != {}}
    final_assignments = {}
    for node,flows in assignments.items():
        name = node[0]
        if not name in final_assignments:
            final_assignments[name] = []
        conflict_info = conflict_information[node]
        assignment = {}
        assignment['day'] = node[1]
        assignment['time'] = node[2]
        assignment['prev_conflict'] = conflict_info['conflict_before']
        assignment['prev_location'] = conflict_info['location']
        assignment['amt_free_time'] = conflict_info['amt_free_time']
        assignment['location'] = []
        for slot_node in flows.keys(): 
            assignment['location'].append(slot_node[0])
        final_assignments[name].append(assignment)
    return final_assignments, flow_dict

def print_list(l, des, dictionary=False):
    print('\n'+des+'\n')
    if dictionary:
        for k, v in l.items():
            print(str(k) + "   " + str(v))
    else:
        for i in l:
            print(i)


def compute_slots_left_open(graph, flow_dict, filter_days=()):
    # flatten into no sources, just dest and flows
    # ((dest, flow) for dest, flow in dest_dict.items())

    disconnected_slots = {node for node in graph.nodes() if 'slot' in node}
    for edge in graph.edges():
        if edge[1] in disconnected_slots:
            disconnected_slots.remove(edge[1])
    slots_flat = [dest_dict for src, dest_dict in flow_dict.items()]
    # print_list(slots_flat, 'flat')
    slots_flatter = []
    for dest_dict in slots_flat:
        slots_flatter += [ (src, flow) for src, flow in dest_dict.items()]
    # print_list(slots_flatter, 'flatter')
    total_into_slot = {}
    for slot, flow in slots_flatter:
        if slot in total_into_slot:
            total_into_slot[slot] += flow
        else:
            total_into_slot[slot] = flow
    # print_list(total_into_slot, 'totals', True)
    open_slots = {slot for slot in total_into_slot if total_into_slot[slot] == 0}
    return [slot for slot in open_slots.union(disconnected_slots) if not slot[1] in filter_days]


def print_assignments(group_assignments):
    for name, assignments in group_assignments.items():
        rep = ''
        rep += '###### '+name+' ######\n'
        for assignment in assignments:
            rep += 'On ' + assignment['day'] + ' at ' + assignment['time']
            rep += (' after ' + assignment['prev_conflict'] if assignment['prev_conflict'] != 'n/a' else '') 
            rep += (' at '+ assignment['prev_location'] if assignment['prev_conflict'] != 'n/a' else '')
            rep += ', ' + name + ' is responsible for quartercarding '
            last_i = len(assignment['location'])-1
            for i in range(len(assignment['location'])):
                if i == last_i:
                    rep += assignment['location'][i] + '.\n'
                elif i == last_i-1 and len(assignment['location']) >= 3:
                    rep += assignment['location'][i] + ', and '
                elif i == last_i-1 and len(assignment['location']) <= 3:
                    rep += assignment['location'][i] + ' and '
                else:
                    rep += assignment['location'][i] + ', '
        rep += '\n'
        print(rep)


def generate_globals(location_file):
    with open(location_file, 'r') as f:
        data = json.load(f)
    HUBS = data['hubs']
    for group in HUBS.keys():
        HUBS[group] = set(HUBS[group])
    NEARNESS_MAP = data['nearness_map']
    for loc in NEARNESS_MAP.keys():
        NEARNESS_MAP[loc] = set(NEARNESS_MAP[loc])
        near_locs = copy.deepcopy(NEARNESS_MAP[loc])
        for near_loc in NEARNESS_MAP[loc]:
            if near_loc in HUBS: # expand group
                near_locs.remove(near_loc)
                near_locs = near_locs.union(HUBS[near_loc])
        NEARNESS_MAP[loc] = near_locs  
    # Allows you to set the starting location to nothing
    NEARNESS_MAP[''] = []
    new_HUBS = set()
    for _, loc_set in HUBS.items():
        new_HUBS = new_HUBS.union(loc_set)
    HUBS = new_HUBS
    return HUBS, NEARNESS_MAP

class Node(object):
    pass

class SuperSource(Node):
    def __repr__(self):
        return "SuperSource"

class SuperSink(Node):
    def __repr__(self):
        return "SuperSink"

class MemberDayNode(Node):
    def __init__(self, member, day):
        super().__init__()
        self.member = member
        self.day = day

    def __repr__(self):
        return str(self.member) + str(self.day)

class MemberNode(Node):
    def __init__(self, member):
        super().__init__()
        self.member = member

    def __repr__(self):
        return str(self.member)

class MemberDayTimeNode(Node):
    def __init__(self, member, day, begin, end, begin_location, end_location = None):
        super().__init__()
        self.member = member
        self.day = day
        self.begin = begin
        self.end = end
        self.begin_location = begin_location
        self.end_location = end_location

    def set_end_location(self, end_location):
        self.end_location = end_location

    def __repr__(self):
        return (
            "[" + str(self.member) + str(self.day) + ", "+ str(self.begin.format("HH:mm")) + "-" 
            + str(self.end.format("HH:mm")) + ", " + str(self.begin_location) + ", " + self.end_location + "]"
        )

class LocationDayNode(Node):
    def __init__(self, location, day):
        super().__init__()
        self.location = location
        self.day = day

    def __repr__(self):
        return str(self.location) + str(self.day)

class Edge(object):
    def __init__(self, destination, capacity=None):
        self.destination = destination
        self.capacity = capacity
        self.flow = 0
        self.cost = 0

    def set_flow(self, flow):
        self.flow = flow
    
    def set_capacity(self, capacity):
        self.capacity = capacity
    
    def set_cost(self, cost):
        self.cost = cost

    def __repr__(self):
        return "->" + str(self.destination) + "/" + str(self.cost) + "/" + str(self.flow)

class Graph(object):
    def __init__(self):
        self.adjacencies = defaultdict(set)

    def add_subgraph(self, subgraph):
        for node, edges in subgraph.adjacencies.items():
            self.adjacencies[node].update(edges)

    def add_node(self, node):
        self.adjacencies[node] = set()

    def add_edge(self, node, edge):
        self.adjacencies[node].add(edge)

    def remove_edge(self, node, edge):
        self.adjacencies[node].remove(edge)

    def get_edges_leaving(self, node):
        return self.adjacencies[node]

    def get_neighbors(self, node):
        return {edge.destination for edge in self.adjacencies[node]}

    def get_source(self):
        for node in self.adjacencies:
            if isinstance(node, SuperSource):
                return node

    def get_sink(self):
        for node in self.adjacencies:
            if isinstance(node, SuperSink):
                return node
            for neighbor in self.get_neighbors(node):
                if isinstance(neighbor, SuperSink):
                    return neighbor

    def get_edge(self, node, dest, consider_backwards=True):
        for edge in self.adjacencies[node]:
            if edge.destination == dest:
                return edge
        # Will first look for a forward edge, then a backwards one
        if consider_backwards:
            for edge in self.adjacencies[dest]:
                if edge.destination == node:
                    return edge
        return None

    def find_augmenting_path(self):
        parents, distances = self.bellman_ford()
        return self.get_path_from_parent_dict(self.get_sink(), parents)

    def get_path_from_parent_dict(self, end, discovery_dict):
        node = end
        path = deque()
        path.appendleft(node)
        while self.get_source() not in path:
            path.appendleft(discovery_dict[node])
            node = discovery_dict[node]
            if node == None:
                return None
        return path, [self.get_edge(path[i], path[i+1]) for i in range(len(path)-1)]

    def augment_path(self, augmenting_path_nodes, augmenting_path_edges):
        bottleneck = min(edge.capacity for edge in augmenting_path_edges)
        for i in range(len(augmenting_path_nodes)-1):
            node_s = augmenting_path_nodes[i]
            node_d = augmenting_path_nodes[i+1]
            augmentable_edge = self.get_edge(node_s, node_d)
            augmentable_edge.set_flow(augmentable_edge.flow + bottleneck)

    @property
    def total_edges(self):
        return sum(len(edges) for edges in self.adjacencies.values())

    @property
    def total_flow(self):
        source = self.get_source()
        return sum(edge.flow for edge in self.adjacencies[source])

    def maximize_flow(self):
        prev_total = -1
        while True:
            res = self.get_residual_graph()
            augmentations = res.find_augmenting_path()
            # While you can add flow
            if not augmentations:
                break
            self.augment_path(*augmentations)
        # cost_network = self.get_cost_network()
        # negative_cost_cycle = cost_network.bellman_ford()
        # print(cost_network)
        # print("\n\n\n\n\n")
        # while negative_cost_cycle:
        #     raise RuntimeError("CRAPO")

    def get_residual_graph(self):
        residual_graph = Graph()
        for node, edges in self.adjacencies.items():
            for edge in edges:
                if edge.flow > 0:
                    e = Edge(node)
                    e.set_capacity(edge.flow)
                    e.set_cost(0-edge.cost)
                    residual_graph.add_edge(edge.destination, e)
                if edge.flow < edge.capacity:
                    e = Edge(edge.destination)
                    e.set_capacity(edge.capacity-edge.flow)
                    # This cost is meaningless to the residual network
                    # but useful when creating the cost_network
                    e.set_cost(edge.cost)
                    residual_graph.add_edge(node, e)
        return residual_graph

    def get_cost_network(self):
        cost_network = Graph()
        residual_graph = self.get_residual_graph()
        for node, edges in self.adjacencies.items():
            for edge in edges:
                e = Edge(edge.destination)
                e.set_cost(edge.cost)
                cost_network.add_edge(node, e)
        for node, edges in residual_graph.adjacencies.items():
            for edge in edges:
                if not self.get_edge(
                    node, edge.destination, consider_backwards=False
                ):
                    e = Edge(edge.destination)
                    e.set_cost(0-edge.cost)
                    cost_network.add_edge(node, e)
        return cost_network


    def bellman_ford(self): 
        dist = {node : float("Inf") for node in self.adjacencies}
        dist[self.get_source()] = 0
        # Since sink doesn't have any outgoing edges
        dist[self.get_sink()] = float("Inf")
        parent = {node : None for node in self.adjacencies}

        # Step 2: Relax all edges |V| - 1 times. A simple shortest  
        # path from src to any other vertex can have at-most |V| - 1  
        # edges
        for _ in range(len(self.adjacencies)-1):
            # Update dist value and parent index of the adjacent vertices of 
            # the picked vertex. Consider only those vertices which are still in 
            # queue 
            for u, edges in self.adjacencies.items():
                for edge in edges:
                    v = edge.destination
                    cost = edge.cost
                    if dist[u] != float("Inf") and dist[u] + cost < dist[v]: 
                        dist[v] = dist[u] + cost 
                        parent[v] = u
        
        # Step 3: check for negative-weight cycles.  The above step  
        # guarantees shortest distances if graph doesn't contain  
        # negative weight cycle.  If we get a shorter path, then there 
        # is a cycle. 
        for u, edges in self.adjacencies.items():
            for edge in edges:
                v = edge.destination
                cost = edge.cost
                if dist[u] != float("Inf") and dist[u] + cost < dist[v]: 
                    raise RuntimeError("Graph contains negative weight cycle")
        return parent, dist

    def __repr__(self):
        repr_string = ""
        for node, edges in self.adjacencies.items():
            repr_string += repr(node) + " : " + repr(edges) + "\n"
        return repr_string

class QuarterCardGraph(Graph):
    def __init__(self, conflicts, daily_limit, weekly_limit, hub_list):
        super().__init__()
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        for conflict_subgraph in (conflict.get_freetime() for conflict in conflicts):
            self.add_subgraph(conflict_subgraph)
        self.add_super_source()
        sink = SuperSink()
        self.add_node(sink)
        self.add_locations(hub_list, sink)
        self.add_edges_from_freetime_to_slots()
        self.add_capacities_to_edges()
        self.add_costs_to_edges()
        
    def add_locations(self, hub_list, sink):
        for location in hub_list:
            for day in DAYS_OF_THE_WEEK:
                node = LocationDayNode(location, day)
                self.add_node(node)
                self.add_edge(node, Edge(sink))

    def add_super_source(self):
        source = SuperSource()
        self.add_node(source)
        for node in list(self.adjacencies.keys()):
            if isinstance(node, MemberNode):
                self.add_edge(source, Edge(node))

    def add_edges_from_freetime_to_slots(self):
        freetimes = (node for node in self.adjacencies.keys() if isinstance(node, MemberDayTimeNode))
        location_day_nodes = [node for node in self.adjacencies.keys() if isinstance(node, LocationDayNode)]
        for mdt_node in freetimes:
            for ldt_node in location_day_nodes:
                if self.viable_assignment(mdt_node, ldt_node):
                    self.add_edge(mdt_node, Edge(ldt_node))
            
    def viable_assignment(self, mdt_node, ldt_node):
        if ldt_node.day == mdt_node.day:
            if self.assignment_within_walking_range(mdt_node, ldt_node):
                return True
        return False

    def assignment_within_walking_range(self, mdt_node, ldt_node):
        """This will contain all of the logic for determining if hub is close enough"""
        return True

    def add_capacities_to_edges(self):
        for node, edges in self.adjacencies.items():
            if isinstance(node, SuperSource):
                self.set_edges_capacity(edges, self.weekly_limit)
            elif isinstance(node, MemberNode):
                self.set_edges_capacity(edges, self.daily_limit)
            elif isinstance(node, MemberDayNode):
                for edge in edges:
                    destination_node = edge.destination
                    time_difference = destination_node.end - destination_node.begin
                    time_difference_minutes = time_difference.seconds / 60
                    # For every 40 minutes you have you can qcard a location
                    edge.set_capacity(math.floor(time_difference_minutes / 40))
            elif isinstance(node, MemberDayTimeNode) or isinstance(node, LocationDayNode):
                self.set_edges_capacity(edges, 1)

    def set_edges_capacity(self, set_of_edges, capacity):
        for edge in set_of_edges:
            edge.set_capacity(capacity)

    def add_costs_to_edges(self):
        relevant_area = [node for node in self.adjacencies if isinstance(node, MemberDayTimeNode)]
        nodes_sorted_by_time = sorted(
            relevant_area, key=lambda node : self.time_of_day_m(node.end)
            )
        acc = 1
        curr = 1
        prev_node = None
        for node in nodes_sorted_by_time:
            if not prev_node or self.time_of_day_m(prev_node.end) != self.time_of_day_m(node.end):
                curr = acc
            self.set_edges_cost(self.adjacencies[node], curr)
            prev_node = node
            acc += curr * len(self.adjacencies[node])
        # TODO: MAKE THIS USEFUL

    def time_of_day_m(self, arrow):
        return (int(arrow.format("HH"))*60) + int(arrow.format("mm"))

    def set_edges_cost(self, set_of_edges, cost):
        for edge in set_of_edges:
            edge.set_cost(cost)

class Conflicts(ics.Calendar):
    @classmethod
    def from_ics_file(cls, path, member, home, location_resolver):
        cal = cls(open(path, "r").read())
        cal.member = member
        cal.home = home
        cal.location_resolver = location_resolver
        return cal

    def get_freetime_for_one_day(self, day):
        # filter out events not occuring on specified day
        events = [e for e in self.events if e.begin.format("dddd") == day]
        graph = Graph()
        day_node = MemberDayNode(self.member, day)
        graph.add_node(day_node)
        if events:
            #
            timeline = ics.Calendar(events=events).timeline
            # start at 9am
            prev_arrow = events[0].begin.replace(hour=9, minute=0, second=0, microsecond=0)
            # end at 6pm
            final_arrow = events[0].begin.replace(hour=18, minute=0, second=0, microsecond=0)
            prev_location = self.home
            for event in timeline:
                if prev_arrow >= event.begin:
                    prev_arrow = max(prev_arrow, event.end)
                    continue
                elif prev_arrow > final_arrow:
                    break
                else:
                    event_location = location_resolver.resolve_location(event.location)
                    freetime_node = MemberDayTimeNode(
                        self.member, day, prev_arrow, event.begin, prev_location, event_location
                    )
                    graph.add_node(freetime_node)
                    graph.add_edge(day_node, Edge(freetime_node))
                    prev_arrow = event.end
                    prev_location = event_location
        if prev_arrow < final_arrow:
            freetime_node = MemberDayTimeNode(
                self.member, day, prev_arrow, final_arrow, prev_location, self.home
            )
            graph.add_node(freetime_node)
            graph.add_edge(day_node, Edge(freetime_node))
        return graph
        
    def get_freetime(self):
        graph = Graph()
        member_node = MemberNode(self.member)
        graph.add_node(member_node)
        for day in DAYS_OF_THE_WEEK:
            daily_freetime_graph = self.get_freetime_for_one_day(day)
            graph.add_subgraph(daily_freetime_graph)
        md_nodes = (node for node in graph.adjacencies if isinstance(node, MemberDayNode))
        for md_node in md_nodes:
            graph.add_edge(member_node, Edge(md_node))
        return graph

class LocationResolver(object):
    def __init__(self, path_to_location_data):
        with open(path_to_location_data, "r") as fd:
            self.location_data = json.loads(fd.read())

    def resolve_location(self, location):
        """ Longest prefix match to remove room numbers from location """
        matching_locations = list(self.location_data.keys())
        word_i = 0
        while len(matching_locations) > 1:
            matching_locations = [
                potential_loc for potential_loc in matching_locations 
                if self.current_word_matches(potential_loc, word_i, location)
            ]
            word_i += 1
        if len(matching_locations) < 1:
            raise RuntimeError(
                str(location) + " has no unique prefix match. Add one to location data file")
        return matching_locations[0]

    def current_word_matches(self, potential_match, word_i, location):
        split_location = location.split(" ")
        split_match = potential_match.split(" ")
        try:
            if split_match[word_i] == split_location[word_i]:
                return True
            else:
                return False
        except IndexError:
            return False

if __name__ == "__main__":
    conflicts = []
    location_resolver = LocationResolver(Path("new_location_data.json"))
    for path in Path("schedules").iterdir():
        if path.name.endswith(".ics"):
            conflicts.append(
                Conflicts.from_ics_file(
                    path, path.name[:-4], home="Collegetown", location_resolver=location_resolver
                )
            )
            g = QuarterCardGraph(conflicts, 3, 14, ("Gates", "Carpenter"))
            g.maximize_flow()
            # for k, v in g.bellman_ford().items():
            #     print(str(k) + " " + str(v))
            for k, v in g.adjacencies.items():
                if isinstance(k, MemberDayTimeNode):
                    edges = {edge for edge in v if edge.flow > 0}
                    if edges:
                        print(str(k) + "   " + str(edges))
            # print(g)
        
        

    """
    You get the full graph, then you can make a duplicate and add in slots, running max flow
    after the addition of each slot.
    """

    
    # parser = argparse.ArgumentParser(description='Automate Quarter Card Scheduling')
    # parser.add_argument('-w','--weekly_limit', help='Limit pub by week. Defaults to 7', action='store', nargs=1)
    # parser.add_argument('-d','--daily_limit', help='Limit pub by day. Defaults to 1', action='store', nargs=1)
    # parser.add_argument('JSON_campus_map', help='A json file specifying buildings that are close')
    # parser.add_argument('JSON_conflicts')
    # args = parser.parse_args()

    # HUBS, NEARNESS_MAP = generate_globals(args.JSON_campus_map)
    # with open(args.JSON_conflicts, 'r') as f:
    #     data = json.load(f)

    # best_assignments = None
    # best_openslots = None
    # best_graph = None
    # best_openslots_num = len(HUBS) * len(DAYS_OF_THE_WEEK)
    # best_daily_limit = -1
    # best_weekly_limit = -1
    # for daily_lim in range(1, int(args.daily_limit[0]) or 1):
    #     for week_lim in range(1, int(args.weekly_limit[0]) or 7):  
    #         DAILY_LIMIT = daily_lim
    #         WEEKLY_LIMIT = week_lim
    #         g = make_graph(data, filter_days=('Saturday','Sunday'))
    #         assignments, flow_dict = compute_assignments(g)
    #         open_slots = compute_slots_left_open(g, flow_dict)
    #         if len(open_slots) < best_openslots_num:
    #             best_assignments = assignments
    #             best_openslots = open_slots
    #             best_openslots_num = len(open_slots)
    #             best_graph = g
    #             best_daily_limit = daily_lim
    #             best_weekly_limit = week_lim
    
    # print(
    #     f"Best daily limit -> {best_daily_limit}\n"
    #     f"Best weekly limit -> {best_weekly_limit}\n"
    # )
    # print_assignments(best_assignments)
    # generate_visualization(best_graph)
    # for slot in best_openslots:
    #     print(slot[0] + " will not be quarter carded on " + slot[1])