#!/usr/bin/env python
import gym
import safety_gym
import argparse

ROBOTS = ['Point', 'Car', 'Doggo']
TASKS = ['Run', 'Circle', 'Goal', 'Button', 'Push']

parser = argparse.ArgumentParser(description='Prints names of all observations')
parser.add_argument('robot', choices=ROBOTS)
parser.add_argument('task', choices=TASKS)
parser.add_argument('level', nargs='?', type=int, choices=(0,1,2), default=0)

def get_obs_names(env_name):
    env = gym.make(env_name)
    env.toggle_observation_space()
    obs = env.reset()
    obs_map = {}
    obs_list = list(range(sum(map(len, obs.values()))))
    for i, (name, pos) in enumerate(
        [(k,i) for k in sorted(obs.keys()) for i in range(len(obs[k]))]):
       name = '{}[{}]'.format(name, pos)
       obs_map[name] = i
       obs_list[i] = name
    return (obs_list, obs_map)

def get_no_velocity_indexes(env_name):
    obs, _ = get_obs_names(env_name)
    return [ i for i, name in enumerate(obs) if ( name.find('accelerometer') == -1 and name.find('jointvel_ankle') == -1 and name.find('jointvel_hip') == -1 and name.find('touch_ankle') == -1 ) ]

def get_no_lidar_indexes(env_name):
    obs, _ = get_obs_names(env_name)
    return [ i for i, name in enumerate(obs) if name.find('lidar') == -1 ]

def get_no_lidar_no_velocity_indexes(env_name):
    obs, _ = get_obs_names(env_name)
    return [ i for i, name in enumerate(obs) if ( name.find('lidar') == -1 and name.find('accelerometer') == -1 and name.find('jointvel_ankle') == -1 and name.find('jointvel_hip') == -1 and name.find('touch_ankle') == -1 ) ]    

if __name__ == '__main__':
    args = parser.parse_args()
    env_name = 'Safexp-{}{}{}-v0'.format(
        args.robot, args.task, '' if args.task in ['Run', 'Circle'] else args.level)
    obs, _ = get_obs_names(env_name)
    for i, name in enumerate(obs):
        print(i, name)
