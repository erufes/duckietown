# For env register
import gym_duckietown

from typing import List
import pygame
from stable_baselines3 import DQN
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium import make
from gym_duckietown.wrappers import (
    DiscreteDifferentialWrapper,
    CropObservation,
    SegmentMiddleLaneWrapper,
    SegmentRemoveExtraInfo,
    SegmentLaneWrapper,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import cv2


def wrap(env):
    env = DiscreteDifferentialWrapper(env)
    env = segment(env)
    return env


def segment(env):
    env = CropObservation(env, 240)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, (40, 80))
    return env


SEED = 9981239780123


def enjoy():
    seed = SEED
    view_env = make("Duckietown-udem1-v0_d")
    view_env = DiscreteDifferentialWrapper(view_env)
    view_env.reset(seed=seed)

    env = make_vec_env("Duckietown-small_loop-v0_d", n_envs=1, wrapper_class=wrap)
    env = VecFrameStack(env, 5)
    env.seed(seed)
    obs, *_ = env.reset()
    display = pygame.display.set_mode((640 + 400, 480))
    font = pygame.font.SysFont("Arial", 24)

    model = DQN.load("dqn_customnet_v2_stack_sm_tested_4_duck", env=env)

    ext_obs = view_env.reset(seed=seed)

    while True:
        pygame.event.pump()
        action = model.predict(obs)
        obs, rew, done, trunc, *_ = env.step(action)

        ext_obs, rew, vdone, *_ = view_env.step(action[0])

        # surf = pygame.surfarray.make_surface(ext_obs)
        # display.blit(surf, (0, 0))
        # pygame.display.update()
        surf = pygame.surfarray.make_surface(ext_obs)
        display.blit(surf, (0, 0))
        for i in range(0, 5):
            text = font.render(f"{5-i}", True, (255, 255, 255))
            low = 3 * i
            high = 3 * i + 3
            single_obs = obs[0, :, :, low:high]
            single_obs = cv2.resize(single_obs, (96, 400))
            surf = pygame.surfarray.make_surface(single_obs)
            display.blit(surf, (640, 96 * i))
            display.blit(text, (640, 96 * i + 96 / 2))
        pygame.display.update()

        if done or vdone:
            print("Done!")
            seed += 1
            env.seed(seed)
            env.reset()
            view_env.reset(seed=seed)
            continue


pygame.init()
enjoy()
