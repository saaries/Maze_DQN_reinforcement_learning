from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='my_maze.envs:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='my_maze.envs:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-sample-7x7-v0',
    entry_point='my_maze.envs:MazeEnvSample7x7',
    timestep_limit=2000,
)

register(
    id='maze-sample-8x8-v0',
    entry_point='my_maze.envs:MazeEnvSample8x8',
    timestep_limit=2000,
)

register(
    id='maze-random-8x8-v0',
    entry_point='my_maze.envs:MazeEnvRandom8x8',
    timestep_limit=2000,
)

register(
    id='maze-sample-9x9-v0',
    entry_point='my_maze.envs:MazeEnvSample9x9',
    timestep_limit=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='my_maze.envs:MazeEnvRandom5x5',
    timestep_limit=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='my_maze.envs:MazeEnvSample10x10',
    timestep_limit=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='my_maze.envs:MazeEnvRandom10x10',
    timestep_limit=10000,
    nondeterministic=True,
)

register(
    id='maze-random-50x50-v0',
    entry_point='my_maze.envs:MazeEnvRandom50x50',
    timestep_limit=10000,
    nondeterministic=True,
)


register(
    id='maze-random-30x30-v0',
    entry_point='my_maze.envs:MazeEnvRandom30x30',
    timestep_limit=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='my_maze.envs:MazeEnvSample3x3',
    timestep_limit=1000,
)

register(
    id='maze-sample-30x30-v0',
    entry_point='my_maze.envs:MazeEnvSample30x30',
    timestep_limit=1000,
)

register(
    id='maze-sample-50x50-v0',
    entry_point='my_maze.envs:MazeEnvSample50x50',
    timestep_limit=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='my_maze.envs:MazeEnvRandom3x3',
    timestep_limit=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='my_maze.envs:MazeEnvSample100x100',
    timestep_limit=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='my_maze.envs:MazeEnvRandom100x100',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='my_maze.envs:MazeEnvRandom10x10Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='my_maze.envs:MazeEnvRandom20x20Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='my_maze.envs:MazeEnvRandom30x30Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)
