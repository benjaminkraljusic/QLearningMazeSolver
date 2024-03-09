from src.QLearningMazeSolver import *
from src.MazeGenerator import *
from timeit import default_timer as timer


discountFactor = 0.9
maxNumberOfMoves = 100
numOfEpisodes = 50
epsilonMax = 1.0
epsilonMin = 0.05
decayRate = 0.05


maze = loadMazeFromTxt("mazes/maze7x7.txt")
mazeSolver = QLearningMazeSolver(maze, numOfEpisodes, maxNumberOfMoves, discountFactor,
                                 epsilonMin, epsilonMax, decayRate)


start = timer()
mazeSolver.learn(initialState=None, deadEndCheckerEnabled=True, greedy=False)
print("Learning time in seconds: " + str(timer() - start))

# mazeSolver.animate()

mazeSolver.showPath((1, 1))
# mazeSolver.showPath((49, 20))
# mazeSolver.showPath((17, 30))
# mazeSolver.showPath((1, 49))

