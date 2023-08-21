#FROM pytorch/pytorch
# FROM nvidia/opengl:1.2-glvnd-devel
FROM duckietown/gym-duckietown
RUN apt-get update -y && apt-get install -y  \
    libglib2.0-0 fontconfig \
    && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /gym-duckietown

COPY . .

#RUN pip install -v -e .

# RUN export PYTHONPATH="${PYTHONPATH}:/gym-duckietown"

# RUN python3 -m pip install pyglet==1.5.15
ENV PYTHONPATH=/code
RUN python3 -c "from gym_duckietown import *"
