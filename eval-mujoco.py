""" Script to eval muojco experiment on a single env (from pre-trained weights). """
import argparse
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from functools import partial
import tensorflow as tf
import derl
from derl import EvalRunner, ActorCriticPolicy
from models import ContinuousActorCriticModel, ODEMLP, MLP, CTRNN, LTC

tf.enable_eager_execution()

ACCEPTED_MODELS = ['mlp','node','ctrnn','ltc']


INT_ARGS = ["seed", "hidden_units", "num_input_layers", "num_dynamics_layers", "num_output_layers","log_period","nenvs","num_epochs","num_minibatches","num_runner_steps"]
FLOAT_ARGS = ["tol","cliprange","entropy_coef","gamma","lambda_","lr","max_grad_norm","num_train_steps","optimizer_epsilon","value_loss_coef"]
BOOL_ARGS = ["save_weights",'recurrent_policy', 'recurrent_value']

def _parser():
  """ Parse the input arguments """
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", type=str, default=None, required=True)
  parser.add_argument("--eval-step", type=int, default=256, required=False)
  parser.add_argument("--render", type=bool, default=False, required=False)
  return parser

  


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
  return partial(MLP, hidden_units=args.hidden_units,
                 num_layers=(args.num_input_layers
                             + args.num_dynamics_layers
                             + args.num_output_layers))

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

def main():
  """ Entrance point. """
  parser = _parser()
  args = parser.parse_args()

  run_args = parse_arg_archive(args.logdir + '/args.txt')
  
  env = derl.env.make(run_args.env_id)
  if hasattr(run_args, 'seed'):
      env.seed(run_args.seed)
  policy = make_mlp_class(run_args.policy_net, run_args.recurrent_policy, run_args)(env.action_space.shape[0])
  value = make_mlp_class(run_args.value_net, run_args.recurrent_value, run_args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)
  model.load_weights(args.logdir+'/model') # load the weights from the logged policy
  polcy_object = ActorCriticPolicy(model)
  runner = EvalRunner(env, polcy_object, args.eval_step, args.render)
  trajectory = runner.get_next()
  
  # rough RNN state display
  policy_states = trajectory['states'][:,0,0,:]
  fig, axs = plt.subplots(2)
  axs[0].plot(trajectory['observations'])
  axs[1].imshow(np.transpose(policy_states), extent=[0, 1, 0, 1])
  plt.show()

if __name__ == "__main__":
  main()
