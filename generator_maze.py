"""
this generator from https://github.com/zuoxingdong/gym-maze
"""
from gym_maze.envs.maze import MazeEnv
from gym_maze.envs.generators import *

def random_maze(render_trace:bool):
    maze = RandomMazeGenerator()
    env = MazeEnv(maze, live_display=True, render_trace=render_trace)
    return env
