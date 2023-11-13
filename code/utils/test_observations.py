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


from typing import List
import gymnasium

from gym_duckietown.wrappers import DiscreteDifferentialWrapper
from gymnasium.wrappers.resize_observation import ResizeObservation


def next_action(cur_action: int):
    if cur_action == 2:
        return 0
    return cur_action + 1


def wrap(env):
    env = segment(env)
    return env


def segment(env):
    env = CropObservation(env, 240)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, (640, 480))
    return env


def train(id):
    print(f"Running train on process {id}")
    env = gymnasium.make("Duckietown-udem1-v0_d")
    env = DiscreteDifferentialWrapper(env)
    env = wrap(env)
    env.reset(seed=123)
    # Action is 0, 1, or 2
    action = 0
    counter = 0
    display = pygame.display.set_mode((640, 480))
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

        obs, rew, done, trunc, *_ = env.step(action)
        surf = pygame.surfarray.make_surface(obs)
        display.blit(surf, (0, 0))
        pygame.display.update()

        env.render()
        if done:
            print("Done!")
            env.reset()
            continue


pygame.init()
train(1)
