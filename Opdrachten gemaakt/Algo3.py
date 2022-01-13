# Python program to print topological sorting of a DAG
from collections import defaultdict


# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of stack
        print(stack)


thisdict = {
    'a': 6,
    'b': 3,
    'c': 4,
    'd': 10,
    'e': 7,
    'f': 8,
    'g': 8


}

g = Graph(7)
g.addEdge(thisdict["a"], thisdict["c"])
g.addEdge(thisdict["a"], thisdict["f"])
g.addEdge(thisdict["b"], thisdict["d"])
g.addEdge(thisdict["b"], thisdict["e"])
g.addEdge(thisdict["c"], thisdict["e"])
g.addEdge(thisdict["f"], thisdict["g"])
g.addEdge(thisdict["e"], thisdict["g"])

print("Following is a Topological Sort of the given graph")
g.topologicalSort()
