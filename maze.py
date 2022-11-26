import numpy as np
import tensorflow as tf
import gym
import time
import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value
import random
from collections import deque
import dqn
import os
from generator_maze import *
env = random_maze(render_trace=True)

# define parameters
INPUT_SIZE = env.observation_space.shape
OUTPUT_SIZE = env.action_space.n
DISCOUNT = 0.95
REPLAY_MEMORY = 10000
BATCH_SIZE = 32
UPDATE_FEQ = 1
MAX_EP = 100
MAX_TRY_STEP = 10000
maze_size = 8

RENDERING = True
RENDERING_STEP = 500

# INPUT_SIZE = env.observation_space.shape
# OUTPUT_SIZE = env.action_space.n
# DISCOUNT = 0.9
# REPLAY_MEMORY = 50000
# BATCH_SIZE = 32
# UPDATE_FEQ = 50
# MAX_EP = 100
# MAX_TRY_STEP = 10000



def get_copy_var_ops(src_scope:str, dest_scope:str):
    copied = []
    src_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope)
    dest_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope)
    for src_var, dest_var in zip(src_vars, dest_vars):
        copied.append(dest_var.assign(src_var.value()))
    return copied

def training(mainDQN:dqn.DQN, targetDQN:dqn.DQN, train_batch:list):
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    # print('rewards:',rewards)
    # print('noutput:',DISCOUNT*np.max(targetDQN.predict(next_states), axis=1)* ~done)

    Q_target = rewards + DISCOUNT*np.max(targetDQN.predict(next_states), axis=1)* ~done
    x = states
    y = mainDQN.predict(states)
    y[np.arange(BATCH_SIZE), actions] = Q_target

    return mainDQN.update(x, y)


if __name__ == "__main__":

    save_summary_path = 'summary/'
    configure(save_summary_path)

    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    step_list = []
    loss_list = []
    best_step = MAX_TRY_STEP
    best_action = []
    t1 = time.time()
    with tf.compat.v1.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.compat.v1.global_variables_initializer())

        copy_ops = get_copy_var_ops("main", "target")
        sess.run(copy_ops)

        for ep in range(MAX_EP):
            e = 1./((ep/10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            loss = 0
            loss_sum = 0
            action_list = []

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                next_state, reward, done, info = env.step(action)
                action_list.append(action)

                # if RENDERING:
                #     if step_count % RENDERING_STEP == 0:
                #         env.render()
                env.render()

                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = training(mainDQN, targetDQN, minibatch)
                    loss_sum += loss


                if step_count % UPDATE_FEQ == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1
                print(" step count : {} | loss : {}".format(step_count, loss), end= "\r")

                if step_count > MAX_TRY_STEP:
                    break

                if done:
                    reward = 1
                    print("\n ! maze cleared in ep : {} | step : {}".format(ep+1, step_count))
                    if best_step > step_count:
                        best_step = step_count
                        best_action = action_list
                    break

            print("\n EP : {} | step : {} ".format(ep+1, step_count))
            step_list.append(step_count)
            loss_list.append(loss)

        print("\n final best step : {}".format(best_step))
        t2 = time.time()
        print('total time used:',t2-t1)

        for act in best_action:
            env.step(act)
            env.render()

        env.close()
        # plt.plot(step_array)
        # plt.show()

        # plt.plot(step_array)
        # plt.plot(step_array)
        #
        # plt.show()

        path = 'result'
        if not os.path.exists(path):
            os.mkdir(path)

        plt.xlabel('Episode')
        plt.ylabel('Step used')
        plt.plot(step_list, linewidth='0.8', color='red')
        plt.savefig(path + '/step' + str(maze_size) +'.png')
        plt.close()

         # print(loss_list)


        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(loss_list, linewidth='0.8', color='red')
        plt.savefig(path+'/loss' + str(maze_size) + '.png')
        # plt.show()

