# duckietown
Duckietown experiments

## Docker Compose:
Instantiates a new container with duckietown gym.
- Mounts code folder to `/code` internal container folder
- Mounts X11 to allow for display piping through docker

## Reinforcement learning
Run `python learning/reinforcement/pytorch/train_reinforcement.py` to train a model.
