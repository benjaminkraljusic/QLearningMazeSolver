import numpy as np
import matplotlib.pyplot as plt
import random 
from timeit import default_timer as timer
from QLearningMazeSolver import QLearningMazeSolver

def main():
    # learning hyperparameters

    n_training_episodes = 700 # number of the episodes agent will use for the training
    gamma = 0.9 # discount factor

    max_steps = 4500 # maximum number of steps

    # exploration/exploatation parameters
    eps_max = 1.00
    eps_min = 0.05
    # Smaller eps_decay_rate allows the agent to choose random actions more often, which leads to better exploration.
    eps_decay_rate = 0.0002 

    # L = np.loadtxt('mazes/maze15x15MultiSol.txt', usecols=range(15), dtype=int)
    L = np.loadtxt('mazes/maze31x31.txt', usecols=range(31), dtype=int)
    # L = np.loadtxt('mazes/maze15x15.txt', usecols=range(15), dtype=int)
    # L = np.loadtxt('mazes/maze11x11.txt', usecols=range(11), dtype=int)
    # L = np.loadtxt('mazes/maze7x7.txt', usecols=range(7), dtype=int)

    # Here, initial and final states are chosen.
    initial_state = (1, 1)
    final_state = (L.shape[0] - 2, L.shape[1] - 2)

    solver = QLearningMazeSolver(L, n_training_episodes, max_steps, initial_state, final_state, gamma, eps_min, eps_max, eps_decay_rate)

    start = timer()
    solver.learn() # solving the maze
    end = timer()

    print("Execution time in seconds: " + str(end - start))

    if solver.status:
        status = "SUCCESS!"
    else:
        status = "FAILED!"

    print("Status: " + status)

    solver.showPath()
    
if __name__ == '__main__':
    main()