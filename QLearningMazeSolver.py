import numpy as np
import matplotlib.pyplot as plt
import random 
from timeit import default_timer as timer

class QLearningMazeSolver:

    def __init__(self, maze, nTrainingEpisodes, maxSteps, initialState, finalState, alpha, gamma, epsMin, epsMax, epsDecayRate):
        self.maze = maze
        self.nTrainingEpisodes = nTrainingEpisodes
        self.maxSteps = maxSteps
        self.initialState = initialState
        self.finalState = finalState
        self.alpha = alpha
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

            if ep%100 == 0 and ep != 0:
                    print(str(ep) + " episodes finished!")

            self.currentState = self.initialState
            
            for i in range(self.maxSteps):
                action = self.getActionEps()

                self.previousState = self.currentState

                # transition to a new state
                self.changeState(action)

                currentQ = self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)]

                # update the Q-table for the previous state
                # self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)] =  currentQ + self.alpha*(self.Rewards[self.currentState] + self.gamma*max(self.QTable[self.currentState][1]) - currentQ)
                self.QTable[self.previousState][1][self.QTable[self.previousState][0].index(action)] =  self.Rewards[self.currentState] + self.gamma*max(self.QTable[self.currentState][1]) 
                # episode is done if the final state is visited
                if self.currentState == self.finalState:
                    break
                
                # episode is done if a dead end is visited
                # if new_state == previous_state and len(self.QTable[current_state][0]) == 1:
                #     self.QTable[current_state][1][0] == -np.inf
                #     break


            if any(self.QTable[self.initialState][1]) != 0:
                self.status = True

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
        plt.figure()
        plt.imshow(self.maze, cmap='binary_r')

        M = 3*self.maze

        path = self.getPath()

        for s in path:
            M[s[0]][s[1]] = 1 
        
        plt.figure()
        plt.imshow(M, cmap='hot')
        plt.show()
