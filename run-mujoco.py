""" Script to run muojco experiment on a single env. """

import tensorflow as tf
import derl
from models import ContinuousActorCriticModel
from utils import make_mlp_class, get_train_parser, ACCEPTED_MODELS, parse_process_obs, parse_reward_fun
tf.enable_eager_execution()
from preprocessing import ProcessEnv




def main():
  """ Entrance point. """
  parser = get_train_parser(derl.get_parser(derl.PPOLearner.get_defaults("mujoco")))
  
  args = derl.log_args(parser.parse_args())

  if args.policy_net not in ACCEPTED_MODELS:
    raise Exception('Policy net argument "{}" not in accepted models ({})'.format(args.policy_net, ACCEPTED_MODELS))
  if args.value_net not in ACCEPTED_MODELS:
    raise Exception('Value net argument "{}" not in accepted models ({})'.format(args.value_net, ACCEPTED_MODELS))
  
  p_obs = parse_process_obs(args.obs_preprocessors)
  env = ProcessEnv(derl.env.make(args.env_id),process_obs=p_obs,reward_function=parse_reward_fun(args.reward_function))
  env.seed(args.seed)
  policy = make_mlp_class(args.policy_net, args.recurrent_policy, args)(env.action_space.shape[0])
  value = make_mlp_class(args.value_net, args.recurrent_value, args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)

  learner = derl.PPOLearner.from_env_args(env, args, model=model, logdir=args.logdir)
  learner.learn(args.num_train_steps, args.logdir, args.log_period)


if __name__ == "__main__":
  main()
