""" Script to eval muojco experiment on a single env (from pre-trained weights). """

import tensorflow as tf
import derl
from derl import EvalRunner, ActorCriticPolicy
from models import ContinuousActorCriticModel
from utils import make_mlp_class, eval_parser, parse_arg_archive
from visualization import state_action_obs_plot

tf.enable_eager_execution()

def main():
  """ Entrance point. """
  parser = eval_parser()
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
  policy_object = ActorCriticPolicy(model)
  runner = EvalRunner(env, policy_object, args.eval_step, args.render)
  trajectory = runner.get_next()
  state_action_obs_plot(trajectory)


if __name__ == "__main__":
  main()
