"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # Stack for DFS
    stack = Stack()
    stack.push((problem.startingState(), []))
    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()

        if state in visited:
            continue

        visited.add(state)
        problem._numExpanded += 1  # ✅ Increment expanded nodes

        if problem.isGoal(state):
            return path

        for successor, action, _ in problem.successorStates(state):
            if successor not in visited:
                stack.push((successor, path + [action]))

    return []
    
    
from collections import deque

def breadthFirstSearch(problem):
    """
    Performs Breadth-First Search (BFS) to find the shortest path.
    """

    # Initialize queue with (state, path)
    frontier = deque([(problem.startingState(), [])])  # FIFO Queue
    explored = set()

    while frontier:
        state, path = frontier.popleft()

        if problem.isGoal(state):  # If goal reached, return path
            return path

        if state not in explored:
            explored.add(state)
            problem._numExpanded += 1  # ✅ Correctly increment expanded nodes

            for successor, action, _ in problem.successorStates(state):
                if successor not in explored:
                    frontier.append((successor, path + [action]))

    return []  # No solution found


    
    
from pacai.util.priorityQueue import PriorityQueue


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # Priority queue for UCS (ordered by path cost)
    priority_queue = PriorityQueue()

    # Push the initial state into the priority queue with cost 0
    priority_queue.push((problem.startingState(), [], 0), 0)

    # Dictionary to track visited states and their lowest cost
    visited = {}

    while not priority_queue.isEmpty():
        state, path, cost = priority_queue.pop()

        # If this state has already been visited at a lower cost, skip it
        if state in visited and visited[state] <= cost:
            continue

        # Mark state as visited with its cost
        visited[state] = cost

        # Goal check
        if problem.isGoal(state):
            return path  # Return the solution path

        # Expand the current state and push successors with updated cost
        for successor, action, step_cost in problem.successorStates(state):
            new_cost = cost + step_cost
            priority_queue.push((successor, path + [action], new_cost), new_cost)

    return []  # No solution found
    
    



def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # Priority queue for A* (ordered by f = g + h)
    priority_queue = PriorityQueue()

    # Push the initial state into the priority queue with cost 0 and heuristic
    start_state = problem.startingState()
    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))

    # Dictionary to track visited states and their lowest cost
    visited = {}

    while not priority_queue.isEmpty():
        state, path, cost = priority_queue.pop()

        # If this state has already been visited at a lower cost, skip it
        if state in visited and visited[state] <= cost:
            continue

        # Mark state as visited with its cost
        visited[state] = cost

        # Goal check
        if problem.isGoal(state):
            return path  # Return the solution path

        # Expand the current state and push successors with updated cost
        for successor, action, step_cost in problem.successorStates(state):
            new_cost = cost + step_cost
            f_value = new_cost + heuristic(successor, problem)
            priority_queue.push((successor, path + [action], new_cost), f_value)

    return []  # No solution found
