# For env register
import gym_duckietown
import pygame
from imgui.integrations.pyglet import create_renderer
from gym_duckietown.wrappers import (
    CropObservation,
    SegmentMiddleLaneWrapper,
    SegmentRemoveExtraInfo,
    SegmentLaneWrapper,
    TransposeToConv2d,
)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from typing import List
import gymnasium

from gym_duckietown.wrappers import DiscreteDifferentialWrapper
from gymnasium.wrappers.resize_observation import ResizeObservation


def next_action(cur_action: int):
    if cur_action == 2:
        return 0
    return cur_action + 1


def wrap(env):
    env = DiscreteDifferentialWrapper(env)
    env = segment(env)
    return env


def segment(env):
    env = CropObservation(env, 240)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, (384, 640))
    env = TransposeToConv2d(env)
    return env


SEED = 123


def train(id):
    print(f"Running train on process {id}")
    env = make_vec_env(
        "Duckietown-udem1-v0_pietroluongo_train",
        n_envs=1,
        wrapper_class=wrap,
        seed=SEED,
    )
    env = VecFrameStack(env, 5)
    env.reset()
    # Action is 0, 1, or 2
    action = 0
    counter = 0
    display = pygame.display.set_mode((1920, 1080))
    pause = False

    while True:
        if pause:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                elif event.key == pygame.K_q:
                    return
            continue
        counter += 1
        if counter % 60 == 0:
            action = next_action(action)

        event = pygame.event.wait(10)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pause = not pause
            elif event.key == pygame.K_q:
                return

        obs, rew, done, trunc, *_ = env.step([action] * 5)
        for i in range(0, 5):
            low = 3 * i
            high = 3 * i + 3
            single_obs = obs[0, low:high, :, :]
            surf = pygame.surfarray.make_surface(single_obs.transpose(2, 1, 0))
            display.blit(surf, (384 * i, 0))
        pygame.display.update()

        if done:
            print("Done!")
            env.reset()
            continue


pygame.init()
train(1)
