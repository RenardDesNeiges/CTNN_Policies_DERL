from platform import processor
from gym import Wrapper
import numpy as np
import numpy.linalg as la

class ProcessEnv(Wrapper):
  """ Runs observations and/or actions through preprocessing. """ 
  def __init__(self, env, process_obs = [], process_act = [], reward_function = None):
    super(ProcessEnv, self).__init__(env)
    self.process_obs = process_obs
    self.process_act = process_act
    self.reward_function = reward_function

  def step(self, act):
    for process in self.process_act:
      act = process(act)
      
    obs, rew, done, info = self.env.step(act)
    
    if self.reward_function is not None:
      rew = self.reward_function(obs, rew, done, act, info)
      
    for process in self.process_obs:
      obs = process(obs)

    return obs, rew, done, info

""" Replaces elements of a vector with zeros as specific indices """
class Mask_vec():
  def __init__(self, ids):
    self.ids = ids
    self.mask = None
  def __call__(self, vec):
    if self.mask is None:
      self.mask = np.zeros_like(vec)
      self.mask[self.ids] = 1
    return vec*self.mask

""" 
Computes a reward as follows 
  - t is a constant time reward given for not having failed yet
  - q is a q cost vector on the observation vector o
  - e is an energy penality vector on action vector a
  - p is a constant penality for failure
  at each s -> s' transition, reward is given by :
    - if not done : rew = t - |o'*q|^2 - |e'*a|^2 > 0
    - if done     : rew = p                       < 0
"""
class LQR_rew():
  def __init__(self, t = 1.0, q = None, e = None, p = -2.0):
    self.t = t
    self.q = q
    self.e = e
    self.p = p

  def __call__(self, obs, rew, done, act, info):
    if done: 
      return self.p
    
    rew = self.t
    if self.q is not None:
      rew -= la.norm(self.q * obs)**2
    if self.e is not None:
      rew -= la.norm(self.e * act)**2
    return rew