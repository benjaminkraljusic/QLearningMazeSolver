import numpy as np
import matplotlib.pyplot as plt
import random 
from timeit import default_timer as timer

class QLearningMazeSolver:

    def __init__(self, maze, nTrainingEpisodes, maxSteps, initialState, finalState, gamma, epsMin, epsMax, epsDecayRate, greedy = True):
        self.maze = maze
        self.nTrainingEpisodes = nTrainingEpisodes
        self.maxSteps = maxSteps
        self.initialState = initialState
        self.finalState = finalState
        self.gamma = gamma
        self.epsMin = epsMin
        self.epsMax = epsMax
        self.eps = 1
        self.epsDecayRate = epsDecayRate
        self.QTable = {}
        self.currentState = initialState
        self.previousState = ()
        self.Rewards = {}
        self.increments = {'up' : [-1, 0], 
                           'down' : [1, 0], 
                           'left' : [0, -1], 
                           'right' : [0, 1]}
        self.status = False
        self.path = None
        self.greedy = greedy
        self.visitedStates = [] # Matrix of the visited states to be used for the animation.
        self.QTableInit()
        
    def QTableInit(self):
        LUT = np.argwhere(self.maze)

        LUT = [tuple(i) for i in LUT]

        for state in LUT:
            actions = []
            qualities = []
            
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
        self.currentState = (self.currentState[0] + self.increments[action][0], self.currentState[1] + self.increments[action][1])
        
    # returns the action with greediness proportional to eps 
    def getActionEps(self):
        rand = np.random.uniform()

        if rand > self.eps:
            return self.QTable[self.currentState][0][np.argmax(self.QTable[self.currentState][1])]
        else:
            return self.QTable[self.currentState][0][random.randint(0, len(self.QTable[self.currentState][0]) - 1)]
            
    def learn(self):
        for key in self.QTable:
            self.Rewards[key] = 0

        self.Rewards[self.finalState] = 100

        for ep in range(self.nTrainingEpisodes):
            self.eps = self.epsMin + (self.epsMax - self.epsMin)*np.exp(-self.epsDecayRate*ep)

            # if ep%100 == 0 and ep != 0:
            #         print(str(ep) + " episodes finished!")

            self.currentState = self.initialState
            visitedStates = [self.currentState]

            for i in range(self.maxSteps):
                action = self.getActionEps()

                self.previousState = self.currentState

                # transition to a new state
                self.changeState(action)
                visitedStates.append(self.currentState)

                currentQ = self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)]

                # update the Q-table for the previous state
                self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)] =  self.Rewards[self.currentState] + self.gamma*max(self.QTable[self.currentState][1]) 
                # episode is done if the final state is visited
                if self.currentState == self.finalState:
                    self.visitedStates.append(visitedStates)
                    break
                
                # episode is done if a dead end is visited
                # if new_state == previous_state and len(self.QTable[current_state][0]) == 1:
                #     self.QTable[current_state][1][0] == -np.inf
                #     break

            if any(self.QTable[self.initialState][1]) != 0:
                self.status = True
                if self.greedy:
                    return # Learning is done as soon as the solution is found. This provides a greedy solution which is not necessarily optimal.

    def getPath(self):
        if not self.status:
            raise Exception("Maze is not solved. Path could not be found.") 
        
        if self.path == None and self.status:
            self.path = [self.initialState]
            self.currentState = self.initialState

            # do the walk
            for _ in self.QTable:
                self.changeState(self.QTable[self.currentState][0][np.argmax(self.QTable[self.currentState][1])])
                self.path.append(self.currentState)
                if self.currentState == self.finalState:
                    break
            
        return self.path


    def showPath(self):
        M = 3*self.maze

        path = self.getPath()

        for s in path:
            M[s[0]][s[1]] = 1 
        
        plt.figure()
        plt.imshow(M, cmap='hot')
        plt.title('SOLVED!')
        plt.show()

    def animate(self):
        if len(self.maze)*len(self.maze[0]) > 49:
            print("Animation not supported for big mazes.")
            return
        
        initM = np.zeros((len(self.maze), len(self.maze[0]), 3), dtype=np.uint8)
        initM[self.maze == 1] = [255, 255, 255]
        initM[self.initialState[0]][self.initialState[1]] = [255, 0, 0]
        initM[self.finalState[0]][self.finalState[1]] = [255, 0, 0]

        for i in range(len(self.visitedStates)):
            title = 'Episode ' + str(i + 1)

            for j in range(len(self.visitedStates[i])):
                M = initM.copy()
                M[self.visitedStates[i][j][0]][self.visitedStates[i][j][1]] = [0, 0, 255]
                plt.title(title)
                plt.imshow(M) 
                plt.pause(0.01)

        self.showPath()
        plt.show()
        


        

