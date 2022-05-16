""" Utilities to parse command line arguments and setup mujoco training/eval envs. """

from functools import partial
from models import ODEMLP, MLP, CTRNN, LTC
from preprocessing import Mask_vec, LQR_rew
from types import SimpleNamespace
import numpy as np
import argparse

ACCEPTED_MODELS = ['mlp','node','ctrnn','ltc']
INT_ARGS = ["seed", "hidden_units", "num_input_layers", "num_dynamics_layers", "num_output_layers","log_period","nenvs","num_epochs","num_minibatches","num_runner_steps"]
FLOAT_ARGS = ["tol","cliprange","entropy_coef","gamma","lambda_","lr","max_grad_norm","num_train_steps","optimizer_epsilon","value_loss_coef"]
BOOL_ARGS = ["save_weights",'recurrent_policy', 'recurrent_value']
OBS_PREPROCESSORS = {'invPendulumNoVelocity': Mask_vec([0,1])}
REWARD_FUNCTIONS = {'invPendulumEnergyPenalty': LQR_rew(e=np.array([0.2])),
                    'invPendulumEnergyPosition': LQR_rew(e=np.array([0.2]),q=np.array([0.001,0.,0.,0.])),}
ACT_PREPROCESSORS = {}


def make_mlp_class(model_arg, is_recurrent, args):
  """ Returns (partial) MLP class with args from args set. """
  if model_arg == 'node':
    return partial(ODEMLP, hidden_units=args.hidden_units,
                   num_input_layers=args.num_input_layers,
                   num_dynamics_layers=args.num_dynamics_layers,
                   num_output_layers=args.num_dynamics_layers,
                   rtol=args.tol, atol=args.tol, is_recurrent=is_recurrent)
  elif model_arg == 'ctrnn':
    return partial(CTRNN, hidden_units=args.hidden_units,
                    num_input_layers=args.num_input_layers,
                    num_dynamics_layers=args.num_dynamics_layers,
                    num_output_layers=args.num_dynamics_layers,
                    rtol=args.tol, atol=args.tol, is_recurrent=is_recurrent)
  elif model_arg == 'ltc':
    return partial(LTC, hidden_units=args.hidden_units,
                    num_input_layers=args.num_input_layers,
                    num_dynamics_layers=args.num_dynamics_layers,
                    num_output_layers=args.num_dynamics_layers,
                    rtol=args.tol, atol=args.tol, is_recurrent=is_recurrent)
  else:
    if is_recurrent:
      raise Exception('No recurrent implementation for MLP policies')
    return partial(MLP, hidden_units=args.hidden_units,
                 num_layers=(args.num_input_layers
                             + args.num_dynamics_layers
                             + args.num_output_layers))

def get_train_parser(base_parser):
  """ Adds neuralode-rl arguments to a give base parser. """
  base_parser.add_argument("--seed", type=int, default=0)
  base_parser.add_argument("--hidden-units", type=int, default=64)
  base_parser.add_argument("--num-input-layers", type=int, default=1)
  base_parser.add_argument("--num-dynamics-layers", type=int, default=1)
  base_parser.add_argument("--num-output-layers", type=int, default=1)
  base_parser.add_argument("--policy-net", type=str, default="mlp")
  base_parser.add_argument("--value-net", type=str, default="mlp")
  base_parser.add_argument("--recurrent-policy", type=bool, default=False)
  base_parser.add_argument("--recurrent-value", type=bool, default=False)
  base_parser.add_argument("--tol", type=float, default=1e-3)
  base_parser.add_argument("--save-weights", type=bool, default=True)
  base_parser.add_argument("--obs-preprocessors", type=str, default='')
  base_parser.add_argument("--act-preprocessors", type=str, default='')
  base_parser.add_argument("--reward-function", type=str, default='')
  return base_parser

def eval_parser():
  """ Parse the input arguments """
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", type=str, default=None, required=True)
  parser.add_argument("--eval-step", type=int, default=256, required=False)
  parser.add_argument("--render", type=bool, default=False, required=False)
  return parser

def parse_arg_archive(args_path):
  with open(args_path, "r") as f:
    args_array = f.readlines()
    
  run_args = dict()
  # setting defaults
  run_args["seed"] = 0
  run_args["hidden_units"] = 64
  run_args["num_input_layers"] = 1
  run_args["num_dynamics_layers"] = 1
  run_args["num_output_layers"] = 1
  run_args["policy_net"] = 'mlp'
  run_args["value_net"] = 'mlp'
  run_args["tol"] = 1e-3
  run_args["save_weights"] = True
  run_args["log_period"] = 1
  run_args["recurrent_policy"] = False
  run_args["recurrent_value"] = False
  
  for el in [a.replace('\n','').split(': ') for a in args_array]:
    run_args[el[0]] = el[1]
    if el[1] == "None":
      run_args[el[0]] = None
    else:
      if el[0] in INT_ARGS:
        run_args[el[0]] = int(el[1])
      if el[0] in FLOAT_ARGS:
        run_args[el[0]] = float(el[1])
      if el[0] in BOOL_ARGS:
        run_args[el[0]] = bool(el[1])
  run_args = SimpleNamespace(**run_args)
  return run_args

def parse_process_obs(arg):
  processors = []
  if arg == '':
    return processors
  for a in arg.split(','):
    if a not in OBS_PREPROCESSORS.keys():
      raise Exception("Invalid argumenst for obs preprocessor: {}".format(a))
    processors.append(OBS_PREPROCESSORS[a])
  return processors

def parse_process_abs(arg):
  processors = []
  if arg == '':
    return processors
  for a in arg.split(','):
    if a not in ACT_PREPROCESSORS.keys():
      raise Exception("Invalid argumenst for act preprocessor: {}".format(a))
    processors.append(ACT_PREPROCESSORS[a])
  return processors

def parse_reward_fun(arg):
  if arg == '':
    return None
  if arg not in REWARD_FUNCTIONS.keys():
    raise Exception("Invalid argumenst for reward function: {}".format(a))
  return REWARD_FUNCTIONS[arg]