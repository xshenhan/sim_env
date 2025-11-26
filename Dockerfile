FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    xvfb \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    libx11-dev \
    libxext-dev \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    COPPELIASIM_ROOT=/root/.coppeliasim \
    QT_QPA_PLATFORM_PLUGIN_PATH=/root/.coppeliasim \
    QT_QPA_PLATFORM=xcb \
    QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/root/.coppeliasim

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
    torch \
    torchvision \
    einops \
    hydra-core \
    wandb \
    dill \
    "zarr<3" \
    numba \
    diffusers \
    accelerate \
    gsutil \
    transformers \
    av \
    gymnasium \
    gdown \
    vector-quantize-pytorch \
    robomimic==0.2.0 \
    robosuite==1.4.0 \
    easydict \
    bddl \
    cloudpickle

RUN wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    mkdir -p $COPPELIASIM_ROOT && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1 && \
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

RUN pip install --no-cache-dir git+https://github.com/Chaoqi-LIU/RLBench.git@factorized_diffusion_policy#egg=RLBench

# xvfb-run -a python scripts/gen_data.py rlbench -t close_box