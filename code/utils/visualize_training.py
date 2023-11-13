# For env register
import gym_duckietown
import imgui.core as imgui
from imgui.integrations.pyglet import create_renderer

from typing import List
import gymnasium

from gym_duckietown.wrappers import DiscreteWrapper

from . import gui
from multiprocessing.connection import Client, Listener

client_addr = ("localhost", 6001)
server_addr = ("localhost", 6000)
listener = Listener(server_addr, authkey=b"secret password")


def train(id):
    print(f"Running train on process {id}")
    env = gymnasium.make("Duckietown-udem1-v0")
    # env = make_vec_env(
    #     "Duckietown-udem1-v0", n_envs=1, wrapper_class=wrap, seed=SEED + 100 * id
    # )
    # env = VecFrameStack(env, 5)
    env.reset(seed=123)
    while True:
        action = [0.3, 0]
        obs, rew, done, trunc, *_ = env.step(action)
        env.render()
        if done:
            print("Done!")
            env.reset(seed=123)
            continue

    print(f"Thread {id} done.")


train(1)
