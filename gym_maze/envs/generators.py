import numpy as np
from numpy.random import randint
from itertools import product as cartesian_product
from skimage.draw import circle, circle_perimeter
import gym
import my_maze
import numpy as np
import random

maze_size = 8
print('maze_size:',maze_size)
action_num = 4
maze_name = "maze-sample-"+str(maze_size)+"x"+str(maze_size)+"-v0"
#
class MazeGenerator(object):
    def __init__(self):
        self.maze = None
    
    def sample_state(self):
        init_state = [1, 1]
        goal_states = [[2*maze_size-1, 2*maze_size-1]]
        return init_state, goal_states
        
    def get_maze(self):
        return self.maze

class RandomMazeGenerator(MazeGenerator):
    def __init__(self):
        super().__init__()
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        # 初始化环境
        env = gym.make(maze_name)
        maze = np.zeros([maze_size * 2 + 1, maze_size * 2 + 1]) + 1


        for i in range(maze_size):
            x = 2 * i + 1
            for j in range(maze_size):
                y = 2 * j + 1
                maze[x][y] = 0
                state = np.array([i, j])
                # 每个点只考虑右和下
                if j < maze_size - 1:
                    maze[x][y + 1] = 1 - env.unwrapped.check(state, 1)
                    maze[x + 1][y] = 1 - env.unwrapped.check(state, 2)

                # 最后一行每个点只考虑右
                else:
                    maze[x + 1][y] = 1 - env.unwrapped.check(state, 2)
        # print()
        # print(maze.astype(int))
        # print()
        return maze.astype(int)



