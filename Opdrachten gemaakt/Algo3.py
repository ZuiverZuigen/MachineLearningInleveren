# Python program to print topological sorting of a DAG
from collections import defaultdict



# List to keep track of the ticks needed
tickList = [6]

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices


    # function to add an edge to graph
    def addEdge(self, u, v, tick):
        self.graph[u].append(v)
        tickList.append(tick)

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


#g.addedge werkt als volgt, De eerste twee cijfers zijn de letters: A = 1, B = 2 Etc.
#Het derde wat erachter staat is het aantal ticks die de knoop nodig heeft om te doorlopen.

g = Graph(8)
g.addEdge(1, 6, 6);
g.addEdge(1, 3, 6);
g.addEdge(2, 4, 3);
g.addEdge(2, 5, 3);
g.addEdge(3, 5, 7);
g.addEdge(5, 7, 7);
g.addEdge(6, 7, 8);

print("Following is a Topological Sort of the given graph")
g.topologicalSort()
print('Following is the processing time of the given graph', sum(tickList))