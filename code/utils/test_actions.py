# For env register
import gym_duckietown
import imgui.core as imgui
from imgui.integrations.pyglet import create_renderer

from typing import List
import gymnasium

from gym_duckietown.wrappers import DiscreteDifferentialWrapper


def next_action(cur_action: int):
    if cur_action == 2:
        return 0
    return cur_action + 1


def train(id):
    print(f"Running train on process {id}")
    env = gymnasium.make("Duckietown-udem1-v0_d")
    env = DiscreteDifferentialWrapper(env)
    env.reset(seed=123)
    # Action is 0, 1, or 2
    action = 0
    counter = 0

    while True:
        counter += 1
        if counter % 60 == 0:
            action = next_action(action)

        obs, rew, done, trunc, *_ = env.step(action)
        env.render()
        # if done:
        #     print("Done!")
        #     env.reset(seed=123)
        #     continue


train(1)
