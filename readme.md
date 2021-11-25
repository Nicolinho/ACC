# Adaptively Calibrated Critic Estimates for Deep Reinforcement Learning
<img src="https://user-images.githubusercontent.com/21196568/143023503-70cdf47a-d039-4c20-9e3c-efa23d3f0383.png">
Official implementation of ACC, described in the paper ["Adaptively Calibrated Critic Estimates for Deep Reinforcement Learning"](https://arxiv.org/abs/2111.12673).
The source code is based on the pytorch implementation of [TQC](https://github.com/SamsungLabs/tqc_pytorch),
which again is based on [TD3](https://github.com/sfujim/TD3). 
We thank the authors for making their source code publicly available.


## Requirements
### Install MuJoCo

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50 from the MuJoCo website.
We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150`).

2. Copy your MuJoCo license key (mjkey.txt) to ```~/.mujoco/mjkey.txt```:

### Install 
We recommend to use an anaconda environment.
In our experiment we used ```python 3.7``` and the following dependencies
```
pip install gym==0.17.2 mujoco-py==1.50.1.68 numpy==1.19.1 torch==1.6.0 torchvision==0.7.0

```

## Running ACC
You can run ACC for TQC on one of the gym continuous control environments by calling
```
python main.py --env "HalfCheetah-v3" --max_timesteps 5000000 --seed 0
```
To run the data efficient variant with 4 critic update steps per environment step you can call

```
python main.py --env "HalfCheetah-v3" --max_timesteps 1000000 --num_critic_updates 4 --seed 0
```
An example script that runs the experiments for 10 seeds and all environments is in
```run_experiment.sh``` and ```run_experiment_data_efficient.sh```.

You can speed up the experiments by using fewer networks to in the ensemble of TQC.
This trades off a little bit of performance for a faster runtime (see the Appendix of the paper).
The number of networks can be controlled with ```--n_nets```. For example
```
python main.py --env "HalfCheetah-v3" --max_timesteps 5000000 --n_nets 2--seed 0
```