# QLearningMazeSolver

## Overview
The `QLearningMazeSolver` is a Python-based project developed as part of the Intelligent Systems course at Mälardalen University, Västerås during the spring semester 2024. This project demonstrates the application of the Q-Learning algorithm for solving mazes. The core of this project is encapsulated in the `QLearningMazeSolver.py` class, where the Q-Learning algorithm and its associated methods are implemented.

## Project done by
- Benjamin Kraljušić
- Dženan Kreho
- Enesa Hrustić

## Project Structure
- `QLearningMazeSolver.py`: Contains the main implementation of the Q-Learning algorithm for solving mazes.
- `main.py`: Used for testing solutions on different mazes, demonstrating the functionality of the Q-Learning algorithm.
- `mazes/`: A directory where maze files are stored. Each maze file represents a different environment for testing the algorithm.

## Running the Solver
To solve a maze, simply run the `main.py` script:
```
python main.py
```
By default, `main.py` will attempt to solve a predefined maze. You can modify `main.py` to load different mazes from the `mazes/` directory. Note that for the mazes of different sizes, different learning parameters, such as number of episodes and maximum number of steps per episode, can lead to more efficient solutions.


## How It Works
The `QLearningMazeSolver` utilizes the Q-Learning algorithm, a model-free reinforcement learning technique, to find the optimal path through the maze. The algorithm iteratively updates the Q-values for each state-action pair until it converges to the optimal policy. This process involves exploring the environment (the maze), learning the rewards associated with different actions in various states, and exploiting this knowledge to navigate through the maze efficiently.

