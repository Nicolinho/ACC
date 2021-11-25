import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy


EPISODE_LENGTH = 1000


def rl(args, results_dir, models_dir, prefix):
    print(' ' * 10 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 10 + k + ': ' + str(v))

    use_acc = args.use_acc

    # remove TimeLimit
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    # policy outputs values in [-1,1], this is rescaled to actual action space
    # which is [-1,1] for the gym cont. contr. envs except humanoid: [-0.4, 0.4]
    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env.seed(args.seed)
    eval_env.seed(10 + args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
    if len(prefix) == 0:
        file_name = f"{args.env}_{args.seed}"
    else:
        file_name = f"{prefix}_{args.env}_{args.seed}"

    if TB_AVAILABLE:
        writer = SummaryWriter(results_dir / file_name)
    else:
        class DummyWriter():
            def add_scalar(self, *args, **kwargs):
                pass
        writer = DummyWriter()

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item(),
                      use_acc=args.use_acc,
                      lr_dropped_quantiles=args.lr_dropped_quantiles,
                      adjusted_dropped_quantiles_init=args.adjusted_dropped_quantiles_init,
                      adjusted_dropped_quantiles_max=args.adjusted_dropped_quantiles_max,
                      diff_ma_coef=args.diff_ma_coef,
                      num_critic_updates=args.num_critic_updates,
                      writer=writer)

    evaluations = []
    state, done = env.reset(), False
    episode_return, last_episode_return = 0, 0
    episode_timesteps = 0
    episode_num = 0
    start_time = time.time()

    if use_acc:
        reward_list = []
        start_ptr = replay_buffer.ptr
        ptr_list = []
        disc_return = []
        time_since_beta_update = 0
        do_beta_update = False

    actor.train()
    for t in range(int(args.max_timesteps)):
        action = actor.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1

        if use_acc:
            reward_list.append(reward)
            time_since_beta_update += 1

        replay_buffer.add(state.astype('float32'), action, next_state, reward, done)
        state = next_state
        episode_return += reward
        if done or episode_timesteps >= EPISODE_LENGTH:
            if use_acc:
                ptr_list.append([start_ptr, replay_buffer.ptr])
                start_ptr = replay_buffer.ptr
                if t > 1:
                    for i in range(episode_timesteps):
                        disc_return.append(
                            np.sum(np.array(reward_list)[i:] * (args.discount ** np.arange(0, episode_timesteps - i))))
                    if time_since_beta_update >= args.beta_udate_rate and t >= args.init_num_steps_before_beta_updates:
                        do_beta_update = True
            reward_list = []

        # Train agent after collecting sufficient data
        if t >= args.init_expl_steps:
            if use_acc and do_beta_update:
                trainer.train(replay_buffer, args.batch_size, ptr_list, disc_return, do_beta_update)
                do_beta_update= False
                for ii, ptr_pair in enumerate(copy.deepcopy(ptr_list)):
                    if (ptr_pair[0] < replay_buffer.ptr - args.size_limit_beta_update_batch):
                        disc_return = disc_return[ptr_pair[1] - ptr_pair[0]:]
                        ptr_list.pop(0)
                    elif (ptr_pair[0] > replay_buffer.ptr and
                             replay_buffer.max_size - ptr_pair[0] + replay_buffer.ptr > args.size_limit_beta_update_batch):
                        if ptr_pair[1] > ptr_pair[0]:
                            disc_return = disc_return[ptr_pair[1] - ptr_pair[0]:]
                        else:
                            disc_return = disc_return[replay_buffer.max_size - ptr_pair[0] + ptr_pair[1]:]
                        ptr_list.pop(0)
                    else:
                        break
                time_since_beta_update = 0
            else:
                trainer.train(replay_buffer, args.batch_size)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Seed: {args.seed} Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.1f}")
            # Reset environment
            state, done = env.reset(), False

            last_episode_return = episode_return
            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            file_name = f"{prefix}_{args.env}_{args.seed}"
            avg_reward = eval_policy(actor, eval_env, EPISODE_LENGTH)
            evaluations.append(avg_reward)
            np.save(results_dir / file_name, evaluations)
            writer.add_scalar('evaluator_return', evaluations[-1], t)
            print( f"EVALUATION: {results_dir.parts[-1] + '/' + file_name} | Seed: {args.seed} Total T: {t + 1}  Reward: {evaluations[-1]:.1f}")
            if args.save_model: trainer.save(models_dir / file_name)

        if t % 1000 == 0 and t > 0:
            writer.add_scalar('exploration_return', last_episode_return, t)

        if t % 5000 == 0 and t > 0:
            if t >=10000:
                writer.add_scalar('fps', 5000 / round(time.time() - start_time), t)
            start_time = time.time()


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v3")              # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--init_expl_steps", default=5000, type=int)    # num of exploration steps before training starts
    parser.add_argument("--seed", default=0, type=int)                  # random seed
    parser.add_argument("--n_quantiles", default=25, type=int)          # number of quantiles for TQC
    parser.add_argument("--use_acc", default=True, type=str2bool)       # if acc for automatic tuning of beta shall be used, o/w top_quantiles_to_drop_per_net will be used
    parser.add_argument("--top_quantiles_to_drop_per_net",
                        default=2, type=int)        # how many quantiles to drop per net. Parameter has no effect if: use_acc = True
    parser.add_argument("--beta_udate_rate", default=1000, type=int)# num of steps between beta/dropped_quantiles updates
    parser.add_argument("--init_num_steps_before_beta_updates",
                        default=25000, type=int)    # num steps before updates to dropped_quantiles are started
    parser.add_argument("--size_limit_beta_update_batch",
                        default=5000, type=int)     # size of most recent state-action pairs stored for dropped_quantiles updates
    parser.add_argument("--lr_dropped_quantiles",
                        default=0.1, type=float)    # learning rate for dropped_quantiles
    parser.add_argument("--adjusted_dropped_quantiles_init",
                        default=2.5, type=float)     # initial value of dropped_quantiles
    parser.add_argument("--adjusted_dropped_quantiles_max",
                        default=5.0, type=float)    # maximal value for dropped_quantiles
    parser.add_argument("--diff_ma_coef", default=0.05, type=float)     # moving average param. for normalization of dropped_quantiles loss
    parser.add_argument("--num_critic_updates", default=1, type=int)    # number of critic updates per environment step
    parser.add_argument("--n_nets", default=5, type=int)                # number of critic networks
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--log_dir", default='results')                 # results directory
    parser.add_argument("--exp_name", default='eval_run')               # name of experiment
    parser.add_argument("--prefix", default='')                         # optional prefix to the name of the experiments
    parser.add_argument("--save_model", default=True, type=str2bool)    # if the model weights shall be saved
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    results_dir = log_dir / args.exp_name
    models_dir = results_dir / 'models'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    rl(args, results_dir, models_dir, args.prefix)
