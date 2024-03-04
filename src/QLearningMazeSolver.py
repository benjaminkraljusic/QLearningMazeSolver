import numpy as np
from bidict import bidict
import matplotlib.pyplot as plotter


def loadMazeFromTxt(mazeFilePath):
    """Loads a maze from a text file as a 2x2 matrix

    Args:
        mazeFilePath (string): Path to the text file

    Returns:
        2x2 matrix that represents the maze
    """
    return np.loadtxt(mazeFilePath, dtype='i', delimiter='\t')


class QLearningMazeSolver:
    """A Q learning based maze solver
    """

    def __init__(self, maze, startOfMaze=(1, 1), endOfMaze=None):
        """Initializes the instance with the maze and the desired start and end position

        Args:
            maze (np.ndarray): The maze represented as a 2x2 matrix with 0 for walls and 1 for paths
            startOfMaze ((int, int)): Matrix indices that specified the desired start position
            endOfMaze ((int, int)): Matrix indices that specified the desired end position
        """
        self.maze = maze
        self.mazeHeight, self.mazeWidth = np.shape(maze)

        if startOfMaze[0] < 0 or self.mazeHeight <= startOfMaze[0] or \
                startOfMaze[1] < 0 or self.mazeWidth <= startOfMaze[1] \
                or maze[startOfMaze[0], startOfMaze[1]] != 1:
            raise ValueError("Start position is not part of the maze")

        self.startOfMaze = startOfMaze

        if endOfMaze is None:
            endOfMaze = (self.mazeHeight - 2, self.mazeWidth - 2)
        elif endOfMaze[0] < 0 or self.mazeHeight <= endOfMaze[0] or \
                endOfMaze[1] < 0 or self.mazeWidth <= endOfMaze[1] \
                or maze[endOfMaze[0], endOfMaze[1]] != 1:
            raise ValueError("End position is not part of the maze")

        self.endOfMaze = endOfMaze

        self.__enumerateStatesInMaze()
        self.__initializeQTable()


    def __enumerateStatesInMaze(self):
        """Enumerates every position in the maze with a unique state (number)
        """
        self.states = bidict()
        for i in range(self.mazeHeight):
            for j in range(self.mazeWidth):
                if self.maze[i][j] == 1:
                    self.states.put(len(self.states), (i, j))

        self.numOfStates = len(self.states)


    def __initializeQTable(self):
        """Initializes the Q table and creates a state-action transition lookup table
        """
        self.QTable = -np.inf * np.ones((self.numOfStates, 4))
        self.possibleMoves = {}

        for currentState in range(self.numOfStates):
            self.possibleMoves[currentState] = []
            i, j = self.states[currentState]

            if self.states.values().__contains__((i, j - 1)):
                adjacentState = self.states.inv[(i, j - 1)]
                self.possibleMoves[currentState].append((adjacentState, 0))
                self.QTable[currentState][0] = 0

            if self.states.values().__contains__((i, j + 1)):
                adjacentState = self.states.inv[(i, j + 1)]
                self.possibleMoves[currentState].append((adjacentState, 1))
                self.QTable[currentState][1] = 0

            if self.states.values().__contains__((i - 1, j)):
                adjacentState = self.states.inv[(i - 1, j)]
                self.possibleMoves[currentState].append((adjacentState, 2))
                self.QTable[currentState][2] = 0

            if self.states.values().__contains__((i + 1, j)):
                adjacentState = self.states.inv[(i + 1, j)]
                self.possibleMoves[currentState].append((adjacentState, 3))
                self.QTable[currentState][3] = 0


    def train(self, discountFactor, numOfEpisodes, maxNumberOfMoves, epsilonMin, epsilonMax, decayRate,
              deadEndCheckerEnabled=False):
        """Trains the Q table on the given maze

        Args:
            discountFactor (float): Discount factor for Q learning algorithm
            numOfEpisodes (int): Number of episodes
            maxNumberOfMoves (int): Maximum number of moves that the agent can perform in a single episode
            epsilonMin (float): Minimum value of the epsilon parameter in the decayed-epsilon-greedy policy
            epsilonMax (float): Maximum value of the epsilon parameter in the decayed-epsilon-greedy policy
            decayRate (float): The decay rate in the decayed-epsilon-greedy policy
            deadEndCheckerEnabled (bool): Specifies whether dead-end checking and optimization is performed
        """
        firstState = self.states.inv[self.startOfMaze]
        finalState = self.states.inv[self.endOfMaze]

        for episode in range(numOfEpisodes):
            epsilon = epsilonMin + (epsilonMax - epsilonMin) * np.exp(-decayRate * episode)
            state = firstState
            prevState = state

            for i in range(maxNumberOfMoves):
                # Select action using the epsilon-greedy policy
                if np.random.uniform(0, 1) > epsilon:
                    action = np.argmax(self.QTable[state])
                else:
                    action = np.random.choice([move[1] for move in self.possibleMoves[state]])

                newState = [move[0] for move in self.possibleMoves[state] if move[1] == action][0]

                if newState == finalState:
                    reward = 1
                else:
                    reward = 0

                # Check if the current state is a dead-end
                if deadEndCheckerEnabled and state != firstState and state != finalState and \
                        newState == prevState and len(self.possibleMoves[state]) == 1:
                    # Remove current state (dead-end) from future consideration
                    moveFromPrevToCurrState = [move for move in self.possibleMoves[prevState] if move[0] == state][0]
                    self.possibleMoves[prevState] \
                        .remove(moveFromPrevToCurrState)
                    self.QTable[state] = np.array(4 * [-np.inf])
                    self.QTable[prevState][moveFromPrevToCurrState[1]] = -np.inf
                else:
                    # Update Q table normally
                    self.QTable[state][action] = reward + discountFactor * np.max(self.QTable[newState])

                # End episode if the final state is reached
                if newState == finalState:
                    break

                prevState = state
                state = newState


    def traverseMaze(self, maxNumberOfMoves):
        """Traverses the maze using the trained Q table

        Args:
            maxNumberOfMoves (int): Maximum number of moves that the agent can perform

        Returns:
            List of positions that the agent has visited while traversing the maze
        """
        state = self.states.inv[self.startOfMaze]
        finalState = self.states.inv[self.endOfMaze]
        path = []

        for i in range(maxNumberOfMoves):
            path.append(self.states[state])
            action = np.argmax(self.QTable[state])

            newState = [move[0] for move in self.possibleMoves[state] if move[1] == action][0]

            if newState == finalState:
                path.append(self.states[newState])
                break

            state = newState

        return path


    def plotMaze(self, path=None):
        """Plots the maze with a traversed path (if one is specified)

        Args:
            path (list): List of positions that the agent has visited while traversing the maze
        """
        if path is None:
            plotter.figure()
            plotter.imshow(self.maze, cmap='binary_r')
            plotter.show()
        else:
            mazeWithPath = self.maze * 2

            for position in path:
                mazeWithPath[position[0]][position[1]] = 1

            plotter.figure()
            plotter.imshow(mazeWithPath, cmap='bone')
            plotter.show()


    def getQTable(self):
        """Returns the Q table

        Returns:
            Q table
        """
        return self.QTable
