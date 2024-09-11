# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import time
import pathlib
import argparse

import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, HierarchicalTrainer
from pmbrl.models import EnsembleModel, RewardModel, ActionModel
from pmbrl.control import HighLevelPlanner, LowLevelPlanner, HierarchicalAgent
from pmbrl.utils import Logger
from pmbrl import get_config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    logger = Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)

    rate_buffer = None
    if args.coverage:
        from pmbrl.envs.envs.ant import rate_buffer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f'Max episode length: {args.max_episode_len}')
    env = GymEnv(
        args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed
    )
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    goal_size = state_size

    normalizer = Normalizer()
    print(f'context_length: {args.context_length}')
    buffer = Buffer(state_size, action_size, goal_size, args.ensemble_size, normalizer, args.context_length,  device=DEVICE)

    ensemble = EnsembleModel(
        state_size + goal_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    )
    reward_model = RewardModel(state_size + goal_size, args.hidden_size, device=DEVICE)
    action_model = ActionModel(state_size, goal_size, action_size, args.hidden_size, device=DEVICE)

    high_level_planner = HighLevelPlanner(
        env=env,
        ensemble=ensemble,
        reward_model=reward_model,
        goal_size=goal_size,
        ensemble_size=args.ensemble_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        use_reward=args.use_reward,
        reward_scale=args.reward_scale,
        use_exploration=args.use_exploration,
        expl_scale=args.expl_scale,
        strategy=args.strategy,
        device=DEVICE,
    )

    low_level_planner = LowLevelPlanner(
        env=env,
        ensemble=ensemble,
        action_model=action_model,
        ensemble_size=args.ensemble_size,
        plan_horizon=args.context_length,
        action_noise_scale=args.action_noise_scale,
        device=DEVICE,
    )
    agent = HierarchicalAgent(env, high_level_planner, low_level_planner, context_length=args.context_length, logger=logger)

    trainer = HierarchicalTrainer(
        high_level_ensemble_model=ensemble,
        high_level_reward_model=reward_model,
        low_level_action_model=action_model,
        buffer=buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
        logger=logger,
        device=DEVICE,
    )

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    print(f'Recording every {args.record_every} episodes')

    for episode in range(1, args.n_episodes):
        logger.log("\n=== Episode {} ===".format(episode))
        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(msg.format(buffer.total_steps, buffer.total_steps * args.action_repeat))
        trainer.reset_models()
        h_ensemble_loss, h_reward_loss, l_action_loss = trainer.train()
        logger.log_losses(h_ensemble_loss, h_reward_loss, l_action_loss)

        recorder = None
        if args.record_every is not None and args.record_every % episode == 0:
            filename = logger.get_video_path(episode)
            print(f"Instantiating recorder: {filename}")
            recorder = VideoRecorder(env.unwrapped, path=filename)
            print(recorder)
            logger.log("Setup recoder @ {}".format(filename))

        logger.log("\n=== Collecting data [{}] ===".format(episode))
        reward, steps, stats = agent.run_episode(
            buffer, action_noise=args.action_noise, recorder=recorder
        )
        logger.log_episode(reward, steps)
        logger.log_stats(stats)

        if args.coverage:
            coverage = rate_buffer(buffer=buffer)
            logger.log_coverage(coverage)

        logger.log_time(time.time() - start_time)
        logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="mountain_car")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=7)
    args = parser.parse_args()
    config = get_config(args)
    main(config)