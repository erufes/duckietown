# For env register
import gym_duckietown
import gymnasium

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.resize_observation import ResizeObservation

from gymnasium.wrappers.frame_stack import FrameStack
from gym_duckietown.wrappers import DiscreteWrapper, CropObservation, ResizeWrapper, SegmentMiddleLaneWrapper, SegmentRemoveExtraInfo, SegmentLaneWrapper

import numpy as np

MODEL_PREFIX = "dqn"
SEED = 123

ckpt_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix=f"{MODEL_PREFIX}_ckpt",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

env = gymnasium.make("Duckietown-udem1-v0")
env = DiscreteWrapper(env)

print(env.observation_space.shape)
env = ResizeObservation(env, 80)
print(env.observation_space.shape)

env = FrameStack(env, 4)
print(env.observation_space.shape)

model = DQN(
    policy="CnnPolicy",
    batch_size=32,
    env=env,
    gamma=0.99,
    learning_rate=0.00005,
    buffer_size=50000,
    verbose=2,
    tensorboard_log="./runs",
    learning_starts=10000
)

model.learn(
    total_timesteps=1_000_000,
    callback=ckpt_callback,
    progress_bar=True,
    tb_log_name=f"{MODEL_PREFIX}",
)
model.save(f"{MODEL_PREFIX}_duck")
