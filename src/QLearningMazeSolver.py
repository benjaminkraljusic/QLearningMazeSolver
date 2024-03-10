import numpy as np
import matplotlib.pyplot as plt
import random


class QLearningMazeSolver:
    """A Q learning based maze solver
    """

    def __init__(self, maze, nTrainingEpisodes, maxSteps, discountFactor, epsMin, epsMax, epsDecayRate, finalState=None):
        """Initializes the instance with the maze and the desired start and end position

        Args:
            maze (np.ndarray): The maze represented as a 2x2 matrix with 0 for walls and 1 for paths
            nTrainingEpisodes (int): Number of training episodes
            maxSteps (int): Maximum number of moves/steps that the agent can perform in a single episode
            discountFactor (float): Discount factor for Q learning algorithm
            epsMin (float): Minimum value of the epsilon parameter in the decayed-epsilon-greedy policy
            epsMax (float): Maximum value of the epsilon parameter in the decayed-epsilon-greedy policy
            epsDecayRate (float): The decay rate in the decayed-epsilon-greedy policy
            finalState ((int, int)): Matrix indices that specified the desired end position
        """

        self.maze = maze
        self.mazeHeight, self.mazeWidth = np.shape(maze)

        if finalState is None:
            finalState = (self.mazeHeight - 2, self.mazeWidth - 2)
        elif finalState[0] < 0 or self.mazeHeight <= finalState[0] or \
                finalState[1] < 0 or self.mazeWidth <= finalState[1] \
                or maze[finalState[0], finalState[1]] != 1:
            raise ValueError("End position is not part of the maze")

        self.nTrainingEpisodes = nTrainingEpisodes
        self.maxSteps = maxSteps
        self.gamma = discountFactor
        self.epsMin = epsMin
        self.epsMax = epsMax
        self.eps = 1
        self.epsDecayRate = epsDecayRate

        self.initialState = None
        self.finalState = finalState
        self.previousPreviousState = None
        self.previousState = None
        self.currentState = None

        self.QTable = {}
        self.rewards = {}
        self.increments = {'up': [-1, 0],
                           'down': [1, 0],
                           'left': [0, -1],
                           'right': [0, 1]}
        self.oppositeAction = {'left': 'right',
                               'right': 'left',
                               'up': 'down',
                               'down': 'up'}

        self.visitedStates = []
        self.QTableInit()


    def QTableInit(self):
        """Initializes the Q table and the state-action transition table
        """

        LUT = np.argwhere(self.maze)
        LUT = [tuple(i) for i in LUT]

        for state in LUT:
            actions = []

            if self.maze[state[0] - 1][state[1]] == 1:
                actions.append('up')
            if self.maze[state[0] + 1][state[1]] == 1:
                actions.append('down')
            if self.maze[state[0]][state[1] - 1] == 1:
                actions.append('left')
            if self.maze[state[0]][state[1] + 1] == 1:
                actions.append('right')

            qualities = [0 for _ in range(len(actions))]

            self.QTable[state] = (actions, qualities)


    def changeState(self, action):
        """Transition the agent from the current state to the next state using the specified action

        Args:
            action (string): The action that will transition the agent from the current state
        """

        self.previousPreviousState = self.previousState
        self.previousState = self.currentState
        self.currentState = (self.currentState[0] + self.increments[action][0],
                             self.currentState[1] + self.increments[action][1])


    def getActionEps(self):
        """Selects an action using the epsilon-greedy policy

        Returns:
            An action that the agent can take
        """

        rand = np.random.uniform()

        if rand > self.eps:
            return self.QTable[self.currentState][0][np.argmax(self.QTable[self.currentState][1])]
        else:
            return self.QTable[self.currentState][0][random.randint(0, len(self.QTable[self.currentState][0]) - 1)]


    def learn(self, initialState = None, deadEndCheckerEnabled = False, greedy=False):
        """Trains/Learns the Q table on the given maze

        Args:
            initialState ((int, int)): Matrix indices that specified the desired start position. If None a random initial position/state will be chosen in each episode
            deadEndCheckerEnabled (bool): Specifies whether dead-end checking and optimization is performed
            greedy (bool): Specifies whether the agent will be greedy, i.e., take the first solution it finds, instead of using all of the episodes to try and find the optimal one
        """

        if greedy is True and initialState is None:
            raise ValueError("Greedy solution finding requires a fixed initial state")

        for key in self.QTable:
            self.rewards[key] = 0

        self.rewards[self.finalState] = 100

        for ep in range(self.nTrainingEpisodes):
            self.eps = self.epsMin + (self.epsMax - self.epsMin) * np.exp(-self.epsDecayRate * ep)

            if initialState is None:
                self.initialState = None
                while self.initialState is None:
                    self.initialState = (np.random.randint(1, self.mazeHeight), np.random.randint(1, self.mazeWidth))
                    if self.maze[self.initialState[0]][self.initialState[1]] == 0:
                        self.initialState = None
            else:
                self.initialState = initialState

            self.previousPreviousState = self.initialState
            self.previousState = self.initialState
            self.currentState = self.initialState
            visitedStates = [self.currentState]

            for i in range(self.maxSteps):
                action = self.getActionEps()

                # Transition to a new state
                self.changeState(action)

                visitedStates.append(self.currentState)

                # Check if the previous state is a dead-end
                if deadEndCheckerEnabled and self.currentState != self.finalState \
                        and self.currentState == self.previousPreviousState \
                        and len(self.QTable[self.previousState][0]) == 1:
                    # Remove the action that transitions current state into previous state (dead-end)
                    indexOfAction = self.QTable[self.currentState][0].index(self.oppositeAction[action])
                    del self.QTable[self.currentState][0][indexOfAction]
                    del self.QTable[self.currentState][1][indexOfAction]
                else:
                    # Update Q table normally
                    self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)] = \
                        self.rewards[self.currentState] + self.gamma * max(self.QTable[self.currentState][1])

                # Episode is done if the final state is visited
                if self.currentState == self.finalState:
                    self.visitedStates.append(visitedStates)
                    break

            # If the greedy parameter is set to true the learning is done as soon as the solution is found. This
            # assumes that the initial state is fixed and provides a greedy solution which is not necessarily optimal.
            if greedy and initialState is not None and any(self.QTable[initialState][1]) != 0:
                return


    def getPath(self, initialState):
        """Traverses the maze from the specified start position using the trained Q table

        Args:
            initialState ((int, int)): Matrix indices that specified the desired start position

        Returns:
            List of positions that the agent has visited while traversing the maze
        """

        if not (any(self.QTable[initialState][1]) != 0):
            raise Exception("Maze is not solved. Path could not be found.")

        path = [initialState]
        self.currentState = initialState
        solutionFound = False

        # Do the walk
        for _ in self.QTable:
            self.changeState(self.QTable[self.currentState][0][np.argmax(self.QTable[self.currentState][1])])
            path.append(self.currentState)
            if self.currentState == self.finalState:
                solutionFound = True
                break

        return path, solutionFound


    def showPath(self, initialState):
        """Plots the maze with a traversed path from the specified start position

        Args:
           initialState ((int, int)): Matrix indices that specified the desired start position
        """

        if not (any(self.QTable[initialState][1]) != 0):
            raise Exception("Maze is not solved. Path could not be found.")

        M = 3 * self.maze

        path, solutionFound = self.getPath(initialState)

        if not solutionFound:
            raise Exception("Maze is not solved. Path could not be found.")

        for s in path:
            M[s[0]][s[1]] = 1

        plt.figure()
        plt.imshow(M, cmap='hot')
        plt.show()


    def animate(self):
        """Plots an animation of the agent learning the maze
        """

        if len(self.maze) * len(self.maze[0]) > 49:
            raise ValueError("Animation not supported for mazes with more than 49 blocks.")

        for i in range(len(self.visitedStates)):
            title = 'Episode ' + str(i + 1)

            for j in range(len(self.visitedStates[i])):
                M = np.zeros((len(self.maze), len(self.maze[0]), 3), dtype=np.uint8)
                M[self.maze == 1] = [255, 255, 255]
                M[self.visitedStates[i][0][0]][self.visitedStates[i][0][1]] = [0, 255, 0]
                M[self.finalState[0]][self.finalState[1]] = [255, 0, 0]
                M[self.visitedStates[i][j][0]][self.visitedStates[i][j][1]] = [0, 0, 255]

                plt.title(title)
                plt.imshow(M)
                plt.pause(0.01)
