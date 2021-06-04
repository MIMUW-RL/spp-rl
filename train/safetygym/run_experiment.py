#!/usr/bin/env python
import multiprocessing as mp
import argparse
import copy
import yaml
import gym
import safety_gym
import time
from rltoolkit import EvalsWrapper, EvalsWrapperACM
from itertools import product
from print_obs_space import get_no_lidar_indexes, get_no_velocity_indexes, get_no_lidar_no_velocity_indexes

from safety_gym.envs.engine import Engine
from gym.envs.registration import register, registry


# Doggo Columns custom environment

config = {
    'robot_base': 'xmls/doggo.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    'observe_pillars': True,
    'pillars_num': 10,
    'goal_size': 0.3,
    'goal_keepout': 0.305, 
}

env = Engine(config)
register(id='Safexp-DoggoColumns0-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config})

#end of custom-envs definition



TASKS = [ 'Run', 'Circle', 'Goal', 'Button', 'Push', 'Columns']
ROBOTS = ['Point', 'Car', 'Doggo']
ALGORITHMS = ['ddpg', 'ppo', 'sac', 'td3']


parser = argparse.ArgumentParser(description='Safety experiment configuration')
parser.add_argument('algorithm', choices=ALGORITHMS)
parser.add_argument('task', nargs='?', choices=TASKS, default='Run')
parser.add_argument('robot', nargs='?', choices=ROBOTS, default='Point')
parser.add_argument('-l', '--level', type=int, choices=range(3), default=0)
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('--spp', nargs='?', const=True, default=False)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--n_cores', type=int, default=1)
parser.add_argument('--tune', nargs='*', help='Config parameters to tune, '
                    'which have list of possible values to check')
parser.add_argument('--neptune_proj', type = str)                    
parser.add_argument('--neptune_token', type=str)
parser.add_argument('--log_dir', type=str, default='.')


def train(kwargs, args):
    if not kwargs.get('use_lidars', True) and not kwargs.get('restrict_obs', True):
        kwargs['acm_ob_idx'] = get_no_lidar_indexes(kwargs['env_name'])
    
    if kwargs.get('restrict_obs', True) and kwargs.get('use_lidars', True):
        kwargs['acm_ob_idx'] = get_no_velocity_indexes(kwargs['env_name'])

    if kwargs.get('restrict_obs', True) and not kwargs.get('use_lidars', True):
        kwargs['acm_ob_idx'] = get_no_lidar_no_velocity_indexes(kwargs['env_name'])
    
    if args.neptune_proj:
        import neptune
        neptune.init('cyranka/spp-mlp', api_token=args.neptune_token)
        neptune.create_experiment(params=kwargs)

    if 'restrict_obs' in kwargs:
        del kwargs['restrict_obs']

    if 'use_lidars' in kwargs:
        del kwargs['use_lidars']
    try:
        if args.spp:
            kwargs['acm_fn'] = lambda in_dim, o_dim, lim, discr: BasicAcM(
                in_dim, o_dim, discr)
            EvalsWrapperACM(**kwargs).perform_evaluations()
        else:
            del kwargs['acm_ob_idx']            
            EvalsWrapper(**kwargs).perform_evaluations()
        if args.neptune_proj:            
            neptune.stop()
    except Exception as e:
        if args.neptune_proj:
            neptune.stop(str(e))
        raise e



def get_algorithm(name, spp):
    if spp:
        name = '{}-{}'.format('spp', name)
    if name == 'ddpg':
        from rltoolkit import DDPG
        return DDPG
    if name == 'spp-ddpg':
        from rltoolkit import DDPG_AcM
        return DDPG_AcM
    if name == 'sac':
        from rltoolkit import SAC
        return SAC
    if name == 'spp-sac':
        from rltoolkit import SAC_AcM
        return SAC_AcM
    if name == 'ppo':
        from rltoolkit import PPO
        return PPO
    if name == 'spp-ppo':
        from rltoolkit import PPO_AcM
        return PPO_AcM
    if name == 'td3':
        from rltoolkit import TD3
        return TD3
    if name == 'spp-td3':
        from rltoolkit.acm.off_policy.td3_acm import TD3_AcM
        return TD3_AcM
    if name == 'safe-td3':
        from rltoolkit import SafeTD3
        return SafeTD3
    raise AttributeError()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['env_name'] = 'Safexp-{}{}{}-v0'.format(
        args.robot, args.task, '' if args.task in ['Run', 'Circle'] else args.level)
    config['max_episode_steps'] = 1000
    algorithm = get_algorithm(args.algorithm, args.spp)
    kwargs_list = []
    acm_args = None
    if args.spp:
        from rltoolkit.acm.models.basic_acm import BasicAcM
        env = gym.make(config['env_name'])
        acm_args = (
            2 * env.observation_space.shape[0], env.action_space.shape[0], False)
    args_to_tune = [] if args.tune is None else args.tune
    configs = []
    for i, h_params in enumerate(product(*[product([param], config[param]) for param in args_to_tune])):
        c = copy.deepcopy(config)
        for param, value in h_params:
            c[param] = value
        for run in range(args.n_runs):
            c['Algo'] = algorithm
            c['evals'] = 1
            c['tensorboard_dir'] = args.log_dir + "/{}{}-{}-{}-{}".format(
                'spp-' if args.spp else '', args.algorithm, c['env_name'], time.time(), i)
            c['log_dir'] = c['tensorboard_dir'] + '/logdir/'
            configs.append(c)
    if len(configs) == 1:
        train(configs[0], args)
    else:
        with mp.Pool(args.n_cores) as p:
            p.starmap(train, product(configs, [args]))
