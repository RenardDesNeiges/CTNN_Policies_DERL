""" RL env runner """
from collections import defaultdict
from multiprocessing import Pool
from .env.summarize import AsyncRewardSummarizer
from .multiprocessing_runner import get_trajectory, stack_trajectories, summarize_traj
from copy import deepcopy
import numpy as np
import random
import os

from .base import BaseRunner
from .trajectory_transforms import (
    GAE, MergeTimeBatch, NormalizeAdvantages)

from collections import namedtuple
State = namedtuple("State", "policy value")

class EnvRunner(BaseRunner):
  """ Reinforcement learning runner in an environment with given policy """
  def __init__(self, env, policy, nsteps, cutoff=None,
               asarray=True, transforms=None, step_var=None):
    super().__init__(env, policy, step_var)
    self.env = env
    self.policy = policy
    self.nsteps = nsteps
    self.cutoff = cutoff
    self.asarray = asarray
    self.transforms = transforms or []
    self.state = {"latest_observation": self.env.reset()}

  def reset(self):
    """ Resets env and runner states. """
    self.state["latest_observation"] = self.env.reset()
    self.policy.reset()

  def get_next(self):
    """ Runs the agent in the environment.  """
    
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    states = []
    self.state["env_steps"] = self.nsteps
    self.state["policy_state"] = State(None, None)
    
    if self.policy.is_recurrent():
      self.policy.reset(self.env.nenvs)
      self.state["policy_state"] = self.policy.get_state()

    for i in range(self.nsteps):
      observations.append(self.state["latest_observation"])
      states.append(self.state["policy_state"])
      act, new_state = self.policy.act(self.state["latest_observation"],state=self.state["policy_state"])
      if "actions" not in act:
        raise ValueError("result of policy.act must contain 'actions' "
                         f"but has keys {list(act.keys())}")

      for key, val in act.items():
        trajectory[key].append(val)

      obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
      self.state["latest_observation"] = obs
      self.state["policy_state"] = new_state
      rewards.append(rew)
      resets.append(done)
      self.step_var.assign_add(self.nenvs or 1)

      # Only reset if the env is not batched. Batched envs should auto-reset.
      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.state["latest_observation"] = self.env.reset()
        if self.policy.is_recurrent():
          self.policy.reset(self.env.nenvs)
          self.state["policy_state"] = self.policy.get_state()
        if self.cutoff or (self.cutoff is None and self.policy.is_recurrent()):
          break
        
    def state_to_array(states):
      policies = np.array([np.array(s.policy)[:,:]for s in states]) if states[0].policy is not None else None
      values   = np.array([np.array(s.value)[:,:]for s in states]) if states[0].value is not None else None
      return State(policies,values)
    
    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    if self.asarray:
      for key, val in trajectory.items():
        try:
          trajectory[key] = np.asarray(val)
        except ValueError:
          raise ValueError(
              f"cannot convert value under key '{key}' to np.ndarray")
    trajectory.update(states=state_to_array(states))
    trajectory["state"] = self.state
    
    for transform in self.transforms:
      transform(trajectory)
      
    return trajectory


class TrajectorySampler(BaseRunner):
  """ Samples parts of trajectory for specified number of epochs. """
  def __init__(self, runner, num_epochs=4, num_minibatches=4,
               shuffle_before_epoch=True, transforms=None):
    super().__init__(runner.env, runner.policy, runner.step_var)
    self.runner = runner
    self.workers = self.runner.env.nenvs
    if self.workers > 1:
      self.summarizer = AsyncRewardSummarizer(1, 'rewards')
    self.num_epochs = num_epochs
    self.num_minibatches = num_minibatches
    self.shuffle_before_epoch = shuffle_before_epoch
    self.transforms = transforms or []
    self.minibatch_count = 0
    self.epoch_count = 0
    self.trajectory = None
    self.init = False
  def trajectory_is_stale(self):
    """ True iff new trajectory should be generated for sub-sampling. """
    return self.epoch_count >= self.num_epochs

  def shuffle_trajectory(self):
    """ Reshuffles trajectory along the first dimension. """
    sample_size = self.trajectory["observations"].shape[0]
    indices = np.random.permutation(sample_size)
    for key, val in filter(lambda kv: isinstance(kv[1], np.ndarray),
                           self.trajectory.items()):
      self.trajectory[key] = val[indices]

  def get_next(self):
    
    if self.trajectory is None or self.trajectory_is_stale():
      self.epoch_count = self.minibatch_count = 0

      if self.workers > 1:
        if not self.init:
          _ = self.runner.policy.act(self.env.reset(),state=self.policy.get_state())
          self.init = True
          
        _seeds = [random.randint(0,int(1e6)) for _ in range(self.workers)]
        _inmap = [(s,deepcopy(self.env), self.logdir, [],self.runner.nsteps//self.workers) for s in _seeds]
        
        self.runner.policy.model.save_weights(os.path.join(self.logdir, "model"))
        with Pool(self.workers) as p:
          trajectories = p.starmap(get_trajectory, _inmap) 
        self.step_var.assign_add(self.runner.nsteps)  
        
        self.trajectory = stack_trajectories(trajectories)
        self.summarizer.add_traj(self.trajectory)
        
      else:
        self.trajectory = self.runner.get_next()
      
      if self.shuffle_before_epoch:
        self.shuffle_trajectory()

    sample_size = self.trajectory["observations"].shape[0]
    mbsize = sample_size // self.num_minibatches
    start = self.minibatch_count * mbsize
    indices = np.arange(start, min(start + mbsize, sample_size))
    minibatch = {key: val[indices] for key, val in self.trajectory.items()
                 if isinstance(val, np.ndarray)}
    minibatch['states'] = State(self.trajectory['states'].policy[indices] \
        if self.trajectory['states'].policy is not None else None ,\
        self.trajectory['states'].value[indices] if self.trajectory['states'].value is not None else None)

    self.minibatch_count += 1
    if self.minibatch_count == self.num_minibatches:
      self.minibatch_count = 0
      self.epoch_count += 1
      if self.shuffle_before_epoch and not self.trajectory_is_stale():
        self.shuffle_trajectory()

    for transform in self.transforms:
      transform(minibatch)
    return minibatch


def make_ppo_runner(env, policy, num_runner_steps, gamma=0.99, lambda_=0.95,
                    num_epochs=3, num_minibatches=4):
  """ Returns env runner for PPO """
  transforms = [GAE(policy, gamma=gamma, lambda_=lambda_, normalize=False)]
  if not policy.is_recurrent() and getattr(env.unwrapped, "nenvs", None):
    transforms.append(MergeTimeBatch())
  runner = EnvRunner(env, policy, num_runner_steps, cutoff=False, transforms=transforms)
  runner = TrajectorySampler(runner, num_epochs=num_epochs,
                             num_minibatches=num_minibatches,
                             shuffle_before_epoch=(not policy.is_recurrent()),
                             transforms=[NormalizeAdvantages()],)
  return runner

class EvalRunner(EnvRunner):
  def __init__(self, env, policy, nsteps, render=False):
    super().__init__(env, policy, nsteps, cutoff=False,
               asarray=True, transforms=None, step_var=None)
    self.render = render
    
  def get_next(self, n_steps = None):
    """ Runs the agent in the environment.  """
    
    if self.render:
      self.env.render()
    
    self.env.seed(random.randint(0,20000))
    if n_steps is None:
      n_steps = self.nsteps
      
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    states = []
    self.state["env_steps"] = n_steps
    self.state["policy_state"] = State(None, None)
    
    if self.policy.is_recurrent():
      self.state["policy_state"] = self.policy.get_state()
      
    for i in range(n_steps):
      if self.render:
        self.env.render()
        
      observations.append(self.state["latest_observation"])
      states.append(self.state["policy_state"])
      act, new_state = self.policy.act(self.state["latest_observation"],state=self.state["policy_state"])
      if "actions" not in act:
        raise ValueError("result of policy.act must contain 'actions' "
                          f"but has keys {list(act.keys())}")
      for key, val in act.items():
        trajectory[key].append(val)

      obs, rew, done, _ = self.env.step(trajectory["actions"][-1])

      self.state["latest_observation"] = obs
      self.state["policy_state"] = new_state
      rewards.append(rew)
      resets.append(done)
      self.step_var.assign_add(self.nenvs or 1)

      # Only reset if the env is not batched. Batched envs should auto-reset.
      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.env.seed(random.randint(0,20000))
        self.state["latest_observation"] = self.env.reset()
        self.policy.reset(1)
        if self.policy.is_recurrent():
          self.state["policy_state"] = self.policy.get_state()
        if self.cutoff or (self.cutoff is None and self.policy.is_recurrent()):
          break

    self.env.reset()
    self.policy.reset(1)
    self.env.close()

    trajectory.update(observations=observations, 
                      rewards=rewards, 
                      resets=resets,
                      states=states)
    
    if self.asarray:
      for key, val in trajectory.items():
        try:
          trajectory[key] = np.asarray(val)
        except ValueError:
          raise ValueError(
              f"cannot convert value under key '{key}' to np.ndarray")
    trajectory["state"] = self.state

    return trajectory