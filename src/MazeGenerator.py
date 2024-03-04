import numpy as np


class MazeGenerator:
    """Random maze matrix generator
    """

    def __init__(self, mazeHeight, mazeWidth):
        """Initializes the instance with the maze width and height

        Args:
            mazeHeight (int): Height of the maze
            mazeWidth (int): Width of the maze
        """
        self.mazeWidth = mazeWidth
        self.mazeHeight = mazeHeight
        self.mazeWidthWithoutBorder = mazeWidth - 2
        self.mazeHeightWithoutBorder = mazeHeight - 2
        self.moves = {
            "left": np.array([0, -2]),
            "right": np.array([0, 2]),
            "up": np.array([-2, 0]),
            "down": np.array([2, 0])
        }


    def generate(self):
        """Generates the maze in the form of a 2x2 matrix, where 0s represent walls and 1s represent free space

        Returns:
            2x2 matrix that represents the maze
        """
        maze = np.zeros((self.mazeHeightWithoutBorder, self.mazeWidthWithoutBorder))
        visited = np.full((self.mazeHeightWithoutBorder, self.mazeWidthWithoutBorder), False, dtype=bool)
        stack = []

        point = np.array([0, 0])

        addToStack = True
        while True:
            if addToStack:
                stack.append(point)
                visited[point[0]][point[1]] = True
                maze[point[0]][point[1]] = 1
            else:
                addToStack = True

            possibleMoves = list(self.moves.keys())
            if point[0] <= 1:
                possibleMoves.remove("up")
            elif point[0] >= self.mazeHeightWithoutBorder - 2:
                possibleMoves.remove("down")
            if point[1] <= 1:
                possibleMoves.remove("left")
            elif point[1] >= self.mazeWidthWithoutBorder - 2:
                possibleMoves.remove("right")

            if np.count_nonzero([not visited[point[0] + self.moves[move][0]][point[1] + self.moves[move][1]]
                                 for move in possibleMoves]) == 0:
                stack.pop()
                if len(stack) == 0:
                    break
                point = stack[-1]
                addToStack = False
                continue

            move = np.random.choice(possibleMoves)
            wall = point + np.array(self.moves[move] / 2, dtype=int)
            maze[wall[0]][wall[1]] = 1
            point = point + self.moves[move]

        mazeWithFrame = np.zeros((self.mazeHeight, self.mazeWidth), dtype=int)
        mazeWithFrame[1:self.mazeHeight-1, 1:self.mazeWidth-1] = maze
        return mazeWithFrame
