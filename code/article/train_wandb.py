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
from gym_duckietown.custom_net import CustomCNN

MODEL_PREFIX = "dqn_cnet_wrapped_stack"
SEED = 2**30 + 8394


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=3),
)


config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 500_000,
    "env_id": "Duckietown-udem1-v0_pietroluongo_train",
    "batch_size": 32,
    "gamma": 0.99,
    "learning_rate": 0.00005,
    "buffer_size": 50000,
    "learning_starts": 10000,
    "seed": SEED,
    "verbose": 2,
    "optimize_memory_usage": True,
    "replay_buffer_kwargs": {"handle_timeout_termination": False},
    "policy_kwargs": policy_kwargs,
}

run = wandb.init(
    project="pgzitos",
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
    seed=SEED,
)
env = VecFrameStack(env, 5)

model = DQN(
    policy=config["policy_type"],
    batch_size=config["batch_size"],
    env=env,
    gamma=config["gamma"],
    learning_rate=config["learning_rate"],
    buffer_size=config["buffer_size"],
    tensorboard_log="./wandb_runs",
    learning_starts=config["learning_starts"],
    seed=config["seed"],
    verbose=2,
    optimize_memory_usage=config["optimize_memory_usage"],
    replay_buffer_kwargs=config["replay_buffer_kwargs"],
    policy_kwargs=config["policy_kwargs"],
)


model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"wandb_models/{run.id}",
        verbose=2,
    ),
)

run.finish()