import heapq
import weakref

"""
@author: Ashka Stephen
Implement search algorithms and search problem definitions
"""

class SearchProblem:
    """
    This class outlines the structure of a search problem.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        raise NotImplementedError()
    
    def get_end_state(self):
        """
        Returns the end goal for the search problem
        """
        raise NotImplementedError()

    def is_goal_state(self, state):
        """
        Returns True if the state is a valid goal state
        """
        raise NotImplementedError()

    def get_successors(self, state):
        raise NotImplementedError()

    def eval_heuristic(self,state):
        """Evaluates the heuristic function at a given state.  Default
        implementation returns 0 (trivial heuristic)."""
        return 0

class SearchNode:
    """Attributes:
    - state: a state object (problem dependent)
    - parent: a reference to the parent SearchNode or None.  If not None,
      this is a weak reference so that search trees are deleted upon last
      reference to the root.
    - paction: the action taken to arrive here from the parent (problem
      dependent)
    - children: a list of children
    """
    def __init__(self, state, parent=None, paction=None, arccost=1):
        """Initializes a SearchNode with a given state.
        """
        self.state = state
        self.parent = None
        if parent is not None:
            self.parent = weakref.proxy(parent)
            parent.children.append(self)
        self.paction = paction
        self.cost_from_start = 0
        if parent is not None:
            self.cost_from_start = parent.cost_from_start + arccost
        self.children = []

    def is_leaf(self):
        """Returns true if leaf node"""
        return len(self.children) == 0

    def get_depth(self):
        """Returns depth of node"""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1

    def path_from_root(self):
        """Returns the path from the root to this node"""
        if self.parent is None:
            return [self]
        p = self.parent.path_from_root()
        p.append(self)
        return p
        
def breadth_first_search(problem):
    root = SearchNode(problem.get_start_state())
    q = [root]
    qState = [root.state]
    visited = []

    while len(q) > 0:
        n = q.pop(0)
        visited.append(n.state)
        succ, act = problem.get_successors(n.state)
        for (s,a) in zip(succ,act):
            c = SearchNode(s,n,a)
            if (c.state not in visited) and (c not in q) and (c.state not in qState):
                q.append(c)
                qState.append(c.state)

            if problem.is_goal_state(s):
                return [n.paction for n in c.path_from_root() if n.parent != None]

    print "No path found!"
    return []


def greedy_search(problem):
    root = SearchNode(problem.get_start_state())
    endGoalState = SearchNode(problem.get_end_state())
    heap = []
    heapq.heappush(heap,(problem.eval_heuristic(root.state, endGoalState.state), root))

    qState = [root.state]
    visited = []
    
    while len(heap) > 0:
        item = heapq.heappop(heap)
        n = item[1]
        visited.append(n.state)
        succ, act = problem.get_successors(n.state)
        for (s,a) in zip(succ,act):
            c = SearchNode(s,n,a)
            if (c.state not in visited) and (c not in heap) and (c.state not in qState):
                heapq.heappush(heap, (problem.eval_heuristic(c.state, endGoalState.state), c))
                qState.append(c.state)
            if problem.is_goal_state(s):
                return [n.paction for n in c.path_from_root() if n.parent != None]

    print "No path found!"
    return []
    start = problem.get_start_state()
    raise NotImplementedError()


class MazeProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.
    """
    def __init__(self, grid):
        """
        Stores maze grid.
        """
        self.grid = grid

    def get_start_state(self):
        """
        Returns the start state
        """
        for i,row in enumerate(self.grid):
            for j,val in enumerate(row):
                if val=='E':
                    return (i,j)
        raise ValueError("No player start state?")


    def get_end_state(self):
        """
        Returns the end goal state
        """
        for i,row in enumerate(self.grid):
            for j,val in enumerate(row):
                if val=='R':
                    return (i,j)
        raise ValueError("No player end state?")

    def is_goal_state(self, state):
        """
        Returns whether this search state is a goal state of the problem"
        """
        return self.grid[state[0]][state[1]] == 'R'
        
    def get_successors(self, state):
        """
        Return value: (succ,act) where
        - succ: a list of successor states
        - act: a list of actions, one for each successor
        """
        successors = []
        actions = []
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        acts = ['n','s','e','w']
        for d,a in zip(dirs,acts):
            nstate = (state[0]+d[0],state[1]+d[1])
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    if (nstate[0]>=0) and (nstate[1] >=0) and (nstate[0]< len(self.grid)) and (nstate[1]<len(self.grid[i])):
                        if ((self.grid[nstate[0]][nstate[1]] == 'd') or (self.grid[nstate[0]][nstate[1]] == 'R')):
                            successors.append(nstate)
                            actions.append(a)
                            break
                        break
                    break
                break
        return successors, actions

    def eval_heuristic(self, state, goal):
        '''This is the heuristic that will be used for greedy search - Manhattan Distance'''
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        raise NotImplementedError()

def pretty_print_grid(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print grid[i][j],
        print ""
