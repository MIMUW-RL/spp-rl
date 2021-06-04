import numpy as np
import gym

def get_goal_sample_fn(env_name):
  if env_name == 'AntMaze':
    # NOTE: When evaluating (i.e. the metrics shown in the paper,
    # we use the commented out goal sampling function.  The uncommented
    # one is only used for training.
    #return lambda: np.array([0., 16.])
    return lambda: np.random.uniform((-4, -4), (20, 20))
  elif env_name == 'AntPush':
    return lambda: np.array([0., 19.])
  elif env_name == 'AntFall':
    return lambda: np.array([0., 27., 4.5])
  else:
    assert False, 'Unknown env'


def get_reward_fn(env_name):
  if env_name == 'AntMaze':
    return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
  elif env_name == 'AntPush':
    return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
  elif env_name == 'AntFall':
    return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
  else:
    assert False, 'Unknown env'


def success_fn(last_reward):
  return last_reward > -5.0


class EnvWithGoal(object):

  def __init__(self, base_env, env_name):
    self.base_env = base_env
    self.env_name = env_name
    self.goal_sample_fn = get_goal_sample_fn(env_name)
    self.reward_fn = get_reward_fn(env_name)
    self.goal = None

  def reset(self, eval = False):
    obs = self.base_env.reset()
    if eval and self.env_name == 'AntMaze':
      self.goal = np.array([0., 16.])
    else:
      self.goal = self.goal_sample_fn()    
    return np.concatenate([obs, self.goal])

  def step(self, a):
    obs, _, done, info = self.base_env.step(a)
    reward = self.reward_fn(obs, self.goal)
    return np.concatenate([obs, self.goal]), reward, done, info

  def success(self, r):
    return success_fn(r)

  @property
  def observation_space(self):
      shape = self.base_env.observation_space.shape + np.array([2, 0])
      high = np.inf * np.ones(shape)
      low = -high
      return gym.spaces.Box(low, high)

  @property
  def action_space(self):
    return self.base_env.action_space
