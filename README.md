# SPHINX

Implementation of _What's the Move? Hybrid Imitation Learning via Salient Points_ (SPHINX) and baselines on Robomimic and real-world tasks.

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://sphinx-manip.github.io/assets/sphinx.pdf)
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

Download the `can` and `square` datasets from [Google Drive](https://drive.google.com/drive/folders/1283M3vPEYml87Y-N8Ievvv3XAVt7iHu6?usp=sharing) and put them under the `data` folder. (i.e. you should have `data/square` and `data/can`).

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
python scripts/train_dense.py --config_path cfgs/dense/dp_<square/can>.yaml
```
To train the point-cloud based Diffusion Policy:
```shell
python scripts/train_dp3.py --config_path cfgs/dense/dp3_<square/can>.yaml
```

## Try the SPHINX Data Collection UI in Sim
(NOTE: We provide a mac_env.yml if you'd like to create a Mac-compatible environment to run our data collection interface)

In sim, we provide a script to allow you to try waypoint-mode data collection for the RoboMimic can/square environments (square by default). This is what was used to collect the provided `square` and `can` datasets. See this [Google Doc](https://docs.google.com/document/d/1mpHAVoCbp7k2y1qc_WS0c4HW7EAOpovFvp6tYPP46hI/edit?usp=sharing) for a walkthrough on how to use the UI.
```shell
source set_env.sh
python interactive_scripts/record_sim.py
```

Open your web browser and navigate to `http://localhost:8080` to access the interface

![Interface Demo](assets/interface.gif)

#### Trouble Shooting

* Make sure no processes are using the required ports:
```shell
sudo lsof -ti:8080,8765,8766 | xargs kill -9
```

* Set the DISPLAY environment variable if not already set:
```shell
export DISPLAY=:0
```

## Real Deployment
As always:
```shell
source set_env.sh
```
### Setup Overview
Our real setup has a Franka Panda FR3 mounted on a table with calibrated RealSense cameras. We run a 1-time calibration procedure to obtain camera extrinsics, stored in `calibration_files/.` See `envs/fr3.yaml` for an example of our configured Panda workspace and RealSense cameras.

We use a workstation to run all scripts which talks to an Intel NUC running a Polymetis controller for the robot and gripper.

### Starting up the Robot & Gripper [On NUC]
Install the [monometis](https://github.com/hengyuan-hu/monometis.git) repo.

#### Launch the robot/gripper server [On NUC]
After going to `172.16.0.2` on the workstation, unlocking the robot, and activating FCI, do the following on NUC:
```shell
# robot
cd launcher; conda activate robo; ./launch_robot.sh

# gripper
cd launcher; conda activate robo; ./launch_gripper.sh
```
#### Start the controller (for receving / executing waypoint/dense actions) [On NUC]
Clone & install  this repo on the NUC
```shell
conda activate robo
python envs/minrobot/server.py
```

### Data Collection in Real [On Workstation]
NOTE: Assumes you have started the robot/gripper as above on the NUC.
The below commands are all run on the workstation, assuming you have cloned the repo and created the conda env.

In real, we use a very similar script to the simulated example above, with SpaceMouse integration for seamless switching between waypoint and dense mode:
```shell
source set_env.sh
python interactive_scripts/record_demo.py
```

### Training/Evaling SPHINX

#### Training
To train SPHINX on a given task (i.e. coffee)
Download the coffee dataset from [Google Drive](https://drive.google.com/drive/folders/1283M3vPEYml87Y-N8Ievvv3XAVt7iHu6?usp=sharing) and put it under the `data` folder (i.e. you should have `data/coffee`).
```shell
python scripts/train_waypoint.py --config cfgs/waypoint/coffee.yaml
python scripts/train_dense.py --config cfgs/dense/dp_coffee.yaml
```

#### Eval [On Workstation]
Assuming the resulting checkpoints are saved to `exps/dense/coffee` and `exps/waypoint/coffee`, to eval SPHINX:
```shell
python scripts/eval_sphinx.py --dense_weight exps/dense/coffee --waypoint_weight exps/waypoint/coffee --env_cfg_path envs/fr3.yaml --freq 10
```

You can also evaluate waypoint-only or dense-only policies using the provided `scripts/eval_dense.py` and `scripts/eval_waypoint.py.`
