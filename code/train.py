from gym_duckietown.envs import DuckietownEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_duckietown.simulator import Simulator
from gym import Env


def generate_env() -> Env:
    return Simulator(
        seed=123,  # random seed
        map_name="loop_empty",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,  # start close to straight
        full_transparency=True,
        distortion=True,
    )


# env = make_vec_env(generate_env, 16)

model = PPO(
    policy="MlpPolicy",
    env=generate_env(),
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=2,
)

model.learn(total_timesteps=1000000)
model_name = "ppo-duck"
model.save(model_name)
