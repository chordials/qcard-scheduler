class graph(object):
    def __init__(self, residual=False):
        self.nodes = {}
        self.residual = residual
    
    def add_node(self, name, node):
        self.nodes[name] = node

    def create_residual_graph(self):
        """Returns a instance of graph representing the residual graph of self"""
        pass
    
    def find_augmenting_path(self):
        assert self.residual


class node(object):
    def __init__(self, graph, name):
        graph.add_node(name, self)
        node.neighbors = {}

    def add_edge(self, dest, capacity):
        if dest in self.neighbors:
            edge = self.neighbors[dest]
            edge.capacity += capacity
        else:
            self.neighbors[dest] = edge(capacity)


class edge(object):
    def __init__(self, capacity):
        self.flow = 0
        self.capacity = capacity
    
    def set_flow(self, new_flow):
        assert new_flow <= self.capacity and new_flow >= 0
        self.flow = new_flow
    
    def add_capacity(self, delta_capacity):
        assert self.capacity + delta_capacity >= 0
        # delete self
        self.capacity += delta_capacity
