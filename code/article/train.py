# For env register
import gym_duckietown

from typing import List

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.resize_observation import ResizeObservation

from gym_duckietown.wrappers import (
    DiscreteDifferentialWrapper,
    CropObservation,
    SegmentMiddleLaneWrapper,
    SegmentRemoveExtraInfo,
    SegmentLaneWrapper,
    TransposeToConv2d,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.monitor import Monitor

import multiprocessing

from gym_duckietown.custom_net import CustomCNN

MODEL_PREFIX = "dqn_customnet_v2_stack_sm_tested"
SEED = 2**30 + 8394
THREAD_COUNT = 4


def wrap(env):
    env = DiscreteDifferentialWrapper(env)
    env = segment(env)
    return env


def segment(env):
    env = CropObservation(env, 300)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, (40, 80))
    env = TransposeToConv2d(env)
    env = Monitor(env)
    return env


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=3),
)


def train(id):
    print(f"Running train on process {id}")
    env = make_vec_env(
        "Duckietown-udem1-v0_pietroluongo_train",
        n_envs=1,
        wrapper_class=wrap,
        seed=SEED + id,
    )
    env = VecFrameStack(env, 5)

    model = DQN(
        policy="CnnPolicy",
        batch_size=32,
        env=env,
        gamma=0.99,
        learning_rate=0.00005,
        buffer_size=50000,
        tensorboard_log="./runs",
        learning_starts=10000,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        verbose=2,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
    )

    ckpt_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=f"{MODEL_PREFIX}_{id}_ckpt",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=500_000,
        callback=ckpt_callback,
        progress_bar=True,
        tb_log_name=f"{MODEL_PREFIX}_{id}",
    )
    model.save(f"{MODEL_PREFIX}_{id}_duck")
    print(f"Thread {id} done.")


if __name__ == "__main__":
    procs: List[multiprocessing.Process] = []
    for i in range(1, THREAD_COUNT + 1):
        procs.append(multiprocessing.Process(target=train, args=[i]))

    for p in procs:
        p.start()

    for p in procs:
        p.join()
