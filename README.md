# Software and Results for the *State Planning Policy Reinforcement Learning* Paper

This repository is the official implementation of the State Planning Policy Reinforcement Learning.  
Demo [video](https://youtu.be/dWnhNnX6f0g).

<img src="plots/spprl.jpg" alt="SPPRL" width="500"/>

## Requirements

Code was run on Ubuntu 18.03 in anaconda environment, in case of another set-up, extra dependencies could be required.
To install requirements run:

```setup
pip install -r rltoolkit/requirements.txt
```

Requirements will install mujoco-py which will work only on installed mujoco with licence (see **Install MuJoCo** section in [mujoco-py documentation](https://github.com/openai/mujoco-py))

Then install `rltoolkit` with:
```rltoolkit install
pip install -e rltoolkit/
```

## Training

To train the models in the paper, you can use scripts from `train` folder.
For example, to train SPP-SAC on the hopper, simply run:

```train
python train/spp_sac_hopper.py
```

After running the script the folder with logs will appear. It will contain tensorboard logs of your runs and `basic_logs` folder. In `basic_logs` you can find 2 pickle files per experiment one with model and one with pickled returns history.

You can find hyperparameters used in our experiments either in paper appendix or `train` folder scripts.

take note of the `N_CORES` parameter within the training scripts, which 
should be set accordingly to the available CPU unit(s).

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





