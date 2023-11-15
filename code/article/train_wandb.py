import gym_duckietown
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import DQN
from gym_duckietown.wrappers import DiscreteDifferentialWrapper
from gym_duckietown.wrappers import (
    CropObservation,
    SegmentMiddleLaneWrapper,
    SegmentRemoveExtraInfo,
    SegmentLaneWrapper,
    TransposeToConv2d,
)
from gymnasium.wrappers.resize_observation import ResizeObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.monitor import Monitor

MODEL_PREFIX = "dqn_wandb"
SEED = 2**30 + 8394

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 500_000,
    "env_id": "Duckietown-udem1-v0_pietroluongo_train",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)


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
    verbose=2,
    optimize_memory_usage=True,
    replay_buffer_kwargs={"handle_timeout_termination": False},
)


model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()
