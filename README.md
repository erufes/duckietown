# duckietown
Duckietown experiments

## Docker Compose:
Instantiates a new container with duckietown gym.
- Mounts code folder to `/code` internal container folder
- Mounts X11 to allow for display piping through docker

## SSH into container
`docker exec -it duckietown-gym-sim bash`

## Reinforcement learning
Run `python /code/learning/reinforcement/pytorch/train_reiforcement.pny` to train a model. Or, from outside container, run `docker exec -it duckietown-gym-sim python /code/learning/reinforcement/pytorch/train_reinforcement.py`

## Simulator
Run `xhost +` before starting the simulator. Then, inside the container, run `export DISPLAY=:0`.
