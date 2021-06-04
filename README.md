# Software and Results for the *State Planning Policy Reinforcement Learning* Paper

This repository is the official implementation of the State Planning Policy Reinforcement Learning.  
Demo [video](https://youtu.be/dWnhNnX6f0g).

<img src="plots/spprl.jpg" alt="SPPRL" width="500"/>

## Requirements

Code was run on Ubuntu 20.04 Ubuntu 20.4 install notes

1. download mujoco200 linux https://www.roboti.us/index.html and put into .mujoco directory with licence
add following line to .bashrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cyranka/.mujoco/mujoco200/bin
```
2. install mujoco-py requirements
```
sudo apt install cmake
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
3. install patchelf
```
sudo add-apt-repository ppa:jamesh/snap-support
sudo apt-get update
sudo apt install patchelf
```

4. 
```
pip install -r requirements.txt
```

Requirements will install mujoco-py which will work only on installed mujoco with licence (see **Install MuJoCo** section in [mujoco-py documentation](https://github.com/openai/mujoco-py))

Then install `rltoolkit` with:
```rltoolkit install
pip install -e rltoolkit/
```

## Training

To train the models in the paper, you can use scripts from `train` folder.
For example, to train SPP-TD3 on Ant, simply run:

```train
?
```

## Evaluation

Model evaluation code is available in the jupyter notebook: `notebooks/load_and_test.ipynb`.
There you can load pre-trained models, evaluate their reward, and render in the environment.


## Pre-trained Models

You can find pre-trained models in `models` directory and check how to load them in `load_and_test.ipynb` notebook.


## Results

Our model achieves the following performance on [OpenAI gym MuJoCo environments](https://gym.openai.com/envs/#mujoco):

Ant results:

<p float="left">
<img src="plots/Ant-v3_DDPG.jpg" alt="ant (spp)ddpg" width="330"/>
<img src="plots/Ant-v3_SAC.jpg" alt="ant (spp)sac" width="330"/>
<img src="plots/Ant-v3_TD3.jpg" alt="ant (spp)td3" width="330"/>
</p>

Humanoid results:

<p float="left">
<img src="plots/Humanoid-v3_DDPG.jpg" alt="humanoid (spp)ddpg" width="330"/>
<img src="plots/Humanoid-v3_SAC.jpg" alt="humanoid (spp)sac" width="330"/>
<img src="plots/Humanoid-v3_TD3.jpg" alt="humanoid (spp)td3" width="330"/>
</p>

Our model achieves the following performance on [OpenAI safety-gym environments](https://github.com/openai/safety-gym):

Doggo-Goal results:

<img src="plots/DoggoGoal.jpg" alt="doggo goal td3" width="500"/>

Doggo-Button results: 

<img src="plots/DoggoButton.jpg" alt="doggo button td3" width="500"/>

Car-Push results:

<img src="plots/CarPush.jpg" alt= "car push td3" width="500"/>





