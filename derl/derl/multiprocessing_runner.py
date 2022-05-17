""" Function for parallel mujoco env sampling """
import numpy as np
from derl.policies import ActorCriticPolicy
from utils import make_mlp_class, eval_parser, parse_arg_archive
from models import ContinuousActorCriticModel
from collections import defaultdict, namedtuple
from .trajectory_transforms import (GAE)

State = namedtuple("State", "policy value")

def get_trajectory(seed, env, logdir, transforms, _nsteps):
  
  parser = eval_parser()  
  run_args = parse_arg_archive(logdir + '/args.txt')
  env.seed(seed)
  
  
  policy = make_mlp_class(run_args.policy_net, run_args.recurrent_policy, run_args)(env.action_space.shape[0])
  value = make_mlp_class(run_args.value_net, run_args.recurrent_value, run_args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)
  if logdir[-1] == '/':
    model.load_weights(logdir+'model') # load the weights from the logged policy
  else:
    model.load_weights(logdir+'/model') # load the weights from the logged policy
  policy_object = ActorCriticPolicy(model)
  
  transforms.append(GAE(policy_object, gamma=run_args.gamma, lambda_=run_args.lambda_, normalize=False))
  trajectory = get_next(policy_object, env, _nsteps, transforms)
  
  return trajectory


def get_next(policy, env, nsteps, transforms):
  trajectory = defaultdict(list, {"actions": []})
  observations = []
  rewards = []
  resets = []
  states = []
  state = {}
  state["env_steps"] = nsteps
  state["policy_state"] = State(None, None)
  
  if policy.is_recurrent():
    state["policy_state"] = policy.get_state()

  state["latest_observation"] = env.reset()

  for i in range(nsteps):
    observations.append(state["latest_observation"])
    states.append(state["policy_state"])
    act, new_state = policy.act(state["latest_observation"],state=state["policy_state"])
    if "actions" not in act:
      raise ValueError("result of policy.act must contain 'actions' "
                        f"but has keys {list(act.keys())}")

    for key, val in act.items():
      trajectory[key].append(val)

    obs, rew, done, _ = env.step(trajectory["actions"][-1])
    state["latest_observation"] = obs
    state["policy_state"] = new_state
    rewards.append(rew)
    resets.append(done)

    # Only reset if the env is not batched. Batched envs should auto-reset. 
    # TODO : do we need that? My guess is we don't
    # if not self.nenvs and np.all(done):
    #   self.state["env_steps"] = i + 1
    #   self.state["latest_observation"] = self.env.reset()
    #   if self.policy.is_recurrent():
    #     self.state["policy_state"] = self.policy.get_state()
    #   if self.cutoff or (self.cutoff is None and self.policy.is_recurrent()):
    #     break
      
  def state_to_array(states):
    policies = np.array([np.array(s.policy)[0,:]for s in states]) if states[0].policy is not None else None
    values   = np.array([np.array(s.value)[0,:]for s in states]) if states[0].value is not None else None
    return State(policies,values)
  
  trajectory.update(observations=observations, rewards=rewards, resets=resets)

  # Multiprocess defaults to asarray
  for key, val in trajectory.items():
    try:
      trajectory[key] = np.asarray(val)
    except ValueError:
      raise ValueError(
          f"cannot convert value under key '{key}' to np.ndarray")
  trajectory.update(states=state_to_array(states))
  trajectory["state"] = state
  
  for transform in transforms:
    transform(trajectory)
    
  return trajectory


def stack_trajectories(_t_in):
  traj = defaultdict()
  for k in _t_in[0].keys():
    if type(_t_in[0][k]) == np.ndarray:
      if _t_in[0][k].ndim == 1:
        l = np.array([[]])
      else:
        l = np.empty((0,_t_in[0][k].shape[1]))
      for i in range(len(_t_in)):
        if _t_in[0][k].ndim == 1:
          l = np.append(l,_t_in[i][k])
        else:
          l = np.concatenate((l,_t_in[i][k]),axis=0)
      if _t_in[0][k].ndim == 1:
        traj[k] = l.astype(np.float32)
      else:
        traj[k] = l.astype(np.float32)
    if tuple in type(_t_in[0][k]).__bases__:
      d = {l:np.empty((0,getattr(_t_in[0][k],l).shape[1])) for l in _t_in[0][k]._fields}
      for l in _t_in[0][k]._fields:
        for i in range(len(_t_in)):
          array = getattr(_t_in[i][k],l)
          d[l] = np.concatenate((d[l],array),axis=0).astype(np.float32)
          
      NT = type(_t_in[0][k])
      traj[k] = NT(**d)
      
  return traj