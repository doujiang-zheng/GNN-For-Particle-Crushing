# GNN-For-Particle-Crushing

> Code for the paper `On The Generalization Of Graph Neural Networks For Predicting Particle Crushing Strength`.

We have generated 45,000 numerical simulations for particle crushing with 900 different particle types in total, the Cartesian product of 20 different particle diameters, 15 different scale shapes in the (X, Y, Z) axes, and 3 different compression axes under one-dimensional compression.
Download `data_files` from [Google Drive](https://drive.google.com/drive/folders/1umqn2aj68uTItp9H-nQGX0QQKRmic_1u?usp=sharing) and place the directory in the current folder.

## Setup the Environment

- `conda create -n particle python=3.9 -y`

- `conda activate particle`

- `pip install -r requirements.txt`. My torch version is torch-1.10.2+cu113.

- Install the dependency of `torch_geoemtric` according to [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). My PYG version is `torch-1.10.2+cu113`, downloaded from [PyG-torch.10.2+cu113](https://data.pyg.org/whl/torch-1.10.2+cu113.html).
- `pip install torch-geometric==2.0.3`

- `bash ./init.sh` to create useful directories.

## Run non-Deep methods

- `python main_baselines.py --model Linear`
- `python main_baselines.py --model Ridge`
- `python main_baselines.py --model RF`
- `python main_baselines.py --model XGB`
- `python main_baselines.py --model LGB`

## Run DNNs

- `python main_nn.py --model MLP`
- `python main_nn.py --model MeshNet`
- `python main_nn.py --model GIN`
- `python main_nn.py --model ExpC`

## Other parameters

- Set the task, `--test-choice diameter | scale | rotation`, refers to the diameter, shape, and axis task in our paper.