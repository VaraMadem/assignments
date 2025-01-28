"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent

from pacai.core.directions import Directions
from pacai.core import distance

from pacai.util.priorityQueue import PriorityQueue
from pacai.student.search import breadthFirstSearch


class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.


    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()

        # Define the four corners of the maze.
        top, right = self.walls.getHeight() - 2, self.walls.getWidth() - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))

        # Initial state: Pac-Man's position and an empty set of visited corners
        self.startState = (self.startingPosition, frozenset())

        # ✅ Track the number of expanded nodes
        self._numExpanded = 0

    def startingState(self):
        """
        Returns the initial state: Pac-Man's position and an empty set of visited corners.
        """
        return self.startState  # ✅ This was missing!

    def isGoal(self, state):
        """
        Returns True if all four corners have been visited.
        """
        _, visitedCorners = state
        return len(visitedCorners) == 4  # ✅ This was missing!

    def successorStates(self, state):
        """Expands the state and increments the expansion counter."""
        successors = []
        currentPosition, visitedCorners = state

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:  # Ensure the move is valid
                nextPosition = (nextx, nexty)
                newVisitedCorners = visitedCorners

                if nextPosition in self.corners and nextPosition not in visitedCorners:
                    newVisitedCorners = frozenset(visitedCorners | {nextPosition})

                successors.append(((nextPosition, newVisitedCorners), action, 1))

        self._numExpanded += 1  # ✅ Increment expanded node count
        return successors

        
    

    def actionsCost(self, actions):
        """Returns the cost of a sequence of actions."""
        if actions is None:
            return 999999  # High penalty if no valid actions

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:  # Invalid move into a wall
                return 999999

        return len(actions)  # Each move costs 1

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    position, visitedCorners = state
    unvisitedCorners = [corner for corner in problem.corners if corner not in visitedCorners]

    if not unvisitedCorners:
        return 0  # Goal state reached

    # 1️⃣ Find the Manhattan distance to the closest unvisited corner
    min_dist_to_corner = min(distance.manhattan(position, corner) for corner in unvisitedCorners)

    # 2️⃣ Compute a Minimum Spanning Tree (MST) approximation
    mst_cost = mst(unvisitedCorners)

    return min_dist_to_corner + mst_cost  # Lower bound estimate
    
    
def mst(points):
    """
    Compute an approximate MST cost using Prim's algorithm.
    """

    if not points:
        return 0

    # Start from an arbitrary node
    total_cost = 0
    remaining = set(points)
    current = remaining.pop()

    while remaining:
        # Find the closest point in the remaining set
        closest = min(remaining, key=lambda point: distance.manhattan(current, point))
        total_cost += distance.manhattan(current, closest)
        remaining.remove(closest)
        current = closest

    return total_cost
    
    
    

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state
    foodList = foodGrid.asList()

    if not foodList:
        return 0  # No food left, return 0

    # Step 1: Compute the distance from Pacman to the closest food pellet.
    closest_food_dist = min(distance.manhattan(position, food) for food in foodList)

    # Step 2: Compute the distance between the two farthest food pellets.
    if len(foodList) > 1:
        max_pairwise_food_dist = max(distance.manhattan(a, b) for a in foodList for b in foodList)
    else:
        max_pairwise_food_dist = 0

    # Step 3: Compute a Minimum Spanning Tree (MST) estimate for all food positions.
    mst_cost = computeMST(foodList) if len(foodList) > 1 else 0

    # Step 4: Combine the heuristic components
    return closest_food_dist + max_pairwise_food_dist + mst_cost



def computeMST(foodList):
    """
    Compute an approximation of the Minimum Spanning Tree (MST) using Prim’s algorithm.

    Returns:
        The estimated cost of connecting all food points using an MST.
    """
    if len(foodList) <= 1:
        return 0  # No MST needed for 1 or fewer points

    total_cost = 0
    visited = set()
    priorityQueue = PriorityQueue()

    # Start from an arbitrary food location
    start = foodList[0]  # Ensure this is an (x, y) tuple
    visited.add(start)

    # Add all edges from the start node to the priority queue
    for food in foodList[1:]:
        if isinstance(food, tuple) and len(food) == 2:  # Ensure valid coordinate
            priorityQueue.push((start, food), distance.manhattan(start, food))

    while len(visited) < len(foodList):
        # Get the closest unvisited node
        popped_item = priorityQueue.pop()

        if not isinstance(popped_item, tuple) or len(popped_item) != 2:
            continue  # Ignore invalid edges

        fromNode, toNode = popped_item  # Ensure it's a tuple (x, y)

        if not (isinstance(fromNode, tuple) and len(fromNode) == 2):
            continue  # Skip invalid nodes

        if not (isinstance(toNode, tuple) and len(toNode) == 2):
            continue  # Skip invalid nodes

        if toNode in visited:
            continue  # Skip already visited nodes

        # Add edge to MST
        visited.add(toNode)
        total_cost += distance.manhattan(fromNode, toNode)

        # Add new edges from this node
        for food in foodList:
            if food not in visited and isinstance(food, tuple) and len(food) == 2:
                priorityQueue.push((toNode, food), distance.manhattan(toNode, food))

    return total_cost
    
    
class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, searchFunction=breadthFirstSearch, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        print("🔍 DEBUG: ClosestDotSearchAgent is using", self.searchFunction.__name__)  # Print search function

        currentState = state
        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception(f"Illegal move: {action} in {currentState}")

                currentState = currentState.generateSuccessor(0, action)

        logging.info(f"Path found with cost {len(self._actions)}.")

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        problem = AnyFoodSearchProblem(gameState)  # ✅ Create a search problem
        return breadthFirstSearch(problem)  # Ensure BFS is used  # Solve using BFS

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState):
        # Call the parent constructor but do not set a specific goal
        super().__init__(gameState)
        self.food = gameState.getFood()  # Grid of food locations

    def isGoal(self, state):
        """Goal test: Return True if this position contains food."""
        x, y = state
        return self.food[x][y]  # If the position contains food, it's a goal.
        
        
class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
