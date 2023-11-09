# For env register
import threading
import gym_duckietown
import gymnasium

from typing import List

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.resize_observation import ResizeObservation

from gymnasium.wrappers.frame_stack import FrameStack
from gym_duckietown.wrappers import DiscreteWrapper, CropObservation, SegmentMiddleLaneWrapper, SegmentRemoveExtraInfo, SegmentLaneWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import numpy as np

import multiprocessing

MODEL_PREFIX = "dqn"
SEED = 123123123

ckpt_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix=f"{MODEL_PREFIX}_ckpt",
    save_replay_buffer=True,
    save_vecnormalize=True,

)

def wrap(env):
    env = DiscreteWrapper(env)
    env = segment(env)
    return env

def segment(env):
    env = CropObservation(env, 140)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, 120)
    return env

def train(id):
    print(f"Running train on {id}")
    env = make_vec_env("Duckietown-udem1-v0", n_envs=4, wrapper_class=wrap, seed=SEED)
    model = DQN(
        policy="CnnPolicy",
        batch_size=32,
        env=env,
        gamma=0.99,
        learning_rate=0.00005,
        buffer_size=50000,
        tensorboard_log="./runs",
        learning_starts=10000,
    )

    model.learn(
        total_timesteps=500_000,
        callback=ckpt_callback,
        progress_bar=True,
        tb_log_name=f"{MODEL_PREFIX}_{id}",
    )
    model.save(f"{MODEL_PREFIX}_{id}_duck")
    print(f"Thread {id} done.")

procs: List[multiprocessing.Process] = []

procs.append(multiprocessing.Process(target=train, args=[1]))
procs.append(multiprocessing.Process(target=train, args=[2]))
procs.append(multiprocessing.Process(target=train, args=[3]))
procs.append(multiprocessing.Process(target=train, args=[4]))

for p in procs:
    p.start()

for p in procs:
    p.join()
