# For env register
import gym_duckietown
import imgui.core as imgui
from imgui.integrations.pyglet import create_renderer

from typing import List
import gymnasium

from gym_duckietown.wrappers import DiscreteDifferentialWrapper


def train(id):
    print(f"Running train on process {id}")
    env = gymnasium.make("Duckietown-udem1-v0_d")
    env = DiscreteDifferentialWrapper(env)
    env.reset(seed=123)
    while True:
        # Action is 0, 1, or 2
        action = 2
        obs, rew, done, trunc, *_ = env.step(action)
        env.render()
        if done:
            print("Done!")
            env.reset(seed=123)
            continue

    print(f"Thread {id} done.")


train(1)
