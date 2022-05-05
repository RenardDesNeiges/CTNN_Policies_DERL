from platform import processor
from gym import Wrapper
import numpy as np

class ProcessEnv(Wrapper):
  """ Runs observations and/or actions through preprocessing. """ 
  def __init__(self, env, process_obs = [], process_act = []):
    super(ProcessEnv, self).__init__(env)
    self.process_obs = process_obs
    self.process_act = process_act

  def step(self, act):
    for process in self.process_act:
      act = process(act)
      
    obs, rew, done, info = self.env.step(act)
      
    for process in self.process_obs:
      obs = process(obs)

    return obs, rew, done, info

class Mask_vec():
  def __init__(self, ids):
    self.ids = ids
    self.mask = None
  def __call__(self, vec):
    if self.mask is None:
      self.mask = np.zeros_like(vec)
      self.mask[self.ids] = 1
    return vec*self.mask
