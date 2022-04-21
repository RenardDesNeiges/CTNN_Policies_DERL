""" Script to run muojco experiment on a single env. """
# pylint: disable=invalid-name
from functools import partial
import tensorflow as tf
import derl
from models import ContinuousActorCriticModel, ODEMLP, MLP, CTRNN
tf.enable_eager_execution()

ACCEPTED_MODELS = ['mlp','node','ctrnn','ltc']

def get_parser(base_parser):
  """ Adds neuralode-rl arguments to a give base parser. """
  base_parser.add_argument("--seed", type=int, default=0)
  base_parser.add_argument("--hidden-units", type=int, default=64)
  base_parser.add_argument("--num-state-layers", type=int, default=1)
  base_parser.add_argument("--num-dynamics-layers", type=int, default=1)
  base_parser.add_argument("--num-output-layers", type=int, default=1)
  base_parser.add_argument("--policy-net", type=str, default="mlp")
  base_parser.add_argument("--value-net", type=str, default="mlp")
  base_parser.add_argument("--tol", type=float, default=1e-3)
  return base_parser


def make_mlp_class(model_arg, args):
  """ Returns (partial) MLP class with args from args set. """
  if model_arg == 'node':
    return partial(ODEMLP, hidden_units=args.hidden_units,
                   num_state_layers=args.num_state_layers,
                   num_dynamics_layers=args.num_dynamics_layers,
                   num_output_layers=args.num_dynamics_layers,
                   rtol=args.tol, atol=args.tol)
  elif model_arg == 'ctrnn':
    return partial(CTRNN, hidden_units=args.hidden_units,
                    num_state_layers=args.num_state_layers,
                    num_dynamics_layers=args.num_dynamics_layers,
                    num_output_layers=args.num_dynamics_layers,
                    rtol=args.tol, atol=args.tol)
  elif model_arg == 'ltc':
    raise Exception("LTC network not implemented")
  return partial(MLP, hidden_units=args.hidden_units,
                 num_layers=(args.num_state_layers
                             + args.num_dynamics_layers
                             + args.num_output_layers))


def main():
  """ Entrance point. """
  parser = get_parser(derl.get_parser(derl.PPOLearner.get_defaults("mujoco")))
  args = derl.log_args(parser.parse_args())

  #checking that the str arguments are acceptable
  if args.policy_net not in ACCEPTED_MODELS:
    raise Exception('Policy net argument "{}" not in accepted models ({})'.format(args.policy_net, ACCEPTED_MODELS))
  if args.value_net not in ACCEPTED_MODELS:
    raise Exception('Value net argument "{}" not in accepted models ({})'.format(args.value_net, ACCEPTED_MODELS))
  
  env = derl.env.make(args.env_id)
  env.seed(args.seed)
  policy = make_mlp_class(args.policy_net, args)(env.action_space.shape[0])
  value = make_mlp_class(args.value_net, args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)

  learner = derl.PPOLearner.from_env_args(env, args, model=model)
  learner.learn(args.num_train_steps, args.logdir, args.log_period)


if __name__ == "__main__":
  main()
