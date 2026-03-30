#!/bin/bash

apt-get update
apt-get install -y tmux
apt-get install -y nvtop

# making sure correct python version is installed in the env
conda install -y python==3.11.8
python3 -m pip install --upgrade pip
pip3 install --no-cache-dir -r requirements_torch.txt
pip cache purge --no-input
conda clean -a -y

# opt-out of dvc data collection
dvc config core.analytics false
dvc pull

pre-commit install