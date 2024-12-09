# SPHINX

Implementation of _What's the Move? Hybrid Imitation Learning via Salient Points_ and baselines on Robomimic and real-world tasks.

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://sphinx-manip.github.io/)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://sphinx-manip.github.io/)

## Clone and create Python environment

### Clone the repo.
```shell
git clone https://github.com/priyasundaresan/sphinx.git
```

### Create conda env

First create a conda env:
```shell
conda env create -f linux_env.yml  
```

Then, source `set_env.sh` to activate the `sphinx_env` conda env and set the `PYTHONPATH` appropriately.

```shell
# NOTE: run this once per shell before running any script from this repo
source set_env.sh
```

## Reproduce RoboMimic Results

Remember to run `source set_env.sh`  once per shell before running any script from this repo.

### Download data

Download dataset and models from [Google Drive](#todo) and put them under the `data` folder.

### Robomimic Square and Can

To train SPHINX, run the following (training logs and eval success rates will be logged to Weights & Biases).

#### SPHINX

```shell
# can
python scripts/train_waypoint.py --config_path cfgs/waypoint/can.yaml

# square
python scripts/train_waypoint.py --config_path cfgs/waypoint/square.yaml
```

Use `--save_dir PATH` to specify where to store the logs and models.
Use `--use_wb 0` to disable logging to W&B (useful when debugging).

#### Vanilla Waypoint
To train the Vanilla Waypoint baseline.
```shell
python scripts/train_waypoint.py --config_path cfgs/waypoint/<square/can>_vanilla.yaml
```

To train the Vanilla Waypoint baseline with Aux. Salient Point classification.
```shell
python scripts/train_waypoint.py --config_path cfgs/waypoint/<square/can>_vanilla_auxsp.yaml
```

#### Diffusion Policies
To train the image-based Diffusion Policy:
```shell
python scripts/train_dense.py --config_path cfgs/dense/<square/can>_dp.yaml
```
To train the point-cloud based Diffusion Policy:
```shell
python scripts/train_dp3.py --config_path cfgs/dense/<square/can>_dp3.yaml
```

## Try the SPHINX Data Collection UI in Sim
(NOTE: We provide a mac_env.yml if you'd like to create a Mac-compatible environment to run our data collection interface)

In sim, we provide a script to allow you to try waypoint-mode data collection for the RoboMimic can/square environments (square by default). This is what was used to collect the provided `square` and `can` datasets. See this [Google Doc]() for some tips and tricks on how to use the UI.
```shell
source set_env.sh
python interactive_scripts/record_sim.py
```

## Real Deployment
As always:
```shell
source set_env.sh
```
### Setup Overview
Our real setup has a Franka Panda FR3 mounted on a table with calibrated RealSense cameras. We run a 1-time calibration procedure to obtain camera extrinsics, stored in `calibration_files/.` See `envs/fr3.yaml` for an example of our configured Panda workspace and RealSense cameras.

We use a workstation to run all scripts which talks to an Intel NUC running a Polymetis controller for the robot and gripper.

### Starting up the Robot & Gripper
Install the [monometis](https://github.com/hengyuan-hu/monometis.git) repo.

#### Launch the robot/gripper server
After going to `172.16.0.2` on the workstation, unlocking the robot, and activating FCI, do the following on NUC:
```shell
# robot
cd launcher; conda activate robo; ./launch_robot.sh

# gripper
cd launcher; conda activate robo; ./launch_gripper.sh
```
#### Start the controller (for receving / executing waypoint/dense actions)
Clone this repo to the NUC
```shell
conda activate robo
python envs/minrobot/server.py
```

### Data Collection in Real
NOTE: Assumes you have started the robot/gripper as above on the NUC.
The below commands are all run on the workstation, assuming you have cloned the repo and created the conda env.

In real, we use a very similar script to the simulated example above, with SpaceMouse integration for seamless switching between waypoint and dense mode:
```shell
source set_env.sh
python interactive_scripts/record_demo.py
```

### Training/Evaling SPHINX
#### SPHINX
To train SPHINX on coffee (as an example, we use the same procedure for other real tasks):
```shell
python scripts/train_waypoint.py --config cfgs/waypoint/coffee.yaml
python scripts/train_dense.py --config cfgs/dense/coffee.yaml
```

Assuming the resulting checkpoints are saved to `exps/dense/coffee` and `exps/waypoint/coffee`, to eval SPHINX:
```shell
python scripts/eval_sphinx.py --dense_weight exps/dense/coffee --waypoint_weight exps/waypoint/coffee --env_cfg_path envs/fr3.yaml --freq 10
```

You can also evaluate waypoint-only or dense-only policies using the provided `scripts/eval_dense.py` and `scripts/eval_waypoint.py.`
