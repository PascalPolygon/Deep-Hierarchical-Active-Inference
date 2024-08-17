from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

class HierarchicalAgent(object):
    """
    Hierarchical agent that manages the interaction between the high-level and low-level planners.
    The agent controls the flow of information and decisions during an episode.
    """

    def __init__(self, env, high_level_planner, low_level_planner, logger=None):
        """
        Initialize the HierarchicalAgent.

        Args:
            env (gym.Env): The environment in which the agent operates.
            high_level_planner (HighLevelPlanner): The high-level planner.
            low_level_planner (LowLevelPlanner): The low-level planner.
            logger (Logger, optional): Logger for recording training information.
        """
        self.env = env
        self.high_level_planner = high_level_planner
        self.low_level_planner = low_level_planner
        self.logger = logger
        self.current_goal = None
        self.high_level_timer = 0

    def get_seed_episodes(self, buffer, n_episodes):
        """
        Collect seed episodes by interacting with the environment using random actions.

        Args:
            buffer (Buffer): The buffer to store experiences.
            n_episodes (int): Number of episodes to collect.

        Returns:
            Buffer: The buffer filled with seed episodes.
        """
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer
    
    def run_episode(self, buffer=None, action_noise=None, recorder=None):
        """
        Run a single episode in the environment, using the hierarchical planners.

        Args:
            buffer (Buffer, optional): The buffer to store experiences.
            action_noise (float, optional): Noise added to the actions for exploration.
            recorder (VideoRecorder, optional): Recorder for capturing video of the episode.

        Returns:
            Tuple: Total reward, total steps, and statistics from the episode.
        """
        total_reward = 0
        total_steps = 0
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                # Generate a new high-level goal if needed
                if self.high_level_timer == 0:
                    self.current_goal = self.high_level_planner(state)
                    self.high_level_timer = self.env.get_high_level_time()

                # Generate actions using the low-level planner
                action, low_level_reward = self.low_level_planner(state, self.current_goal)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                # Take action in the environment
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                total_steps += 1

                # Log progress every 25 steps
                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log("> Step {} [reward {:.2f}]".format(total_steps, total_reward))

                # Add experience to buffer and recorder
                if buffer is not None:
                    buffer.add(state, action, reward + low_level_reward, next_state, self.current_goal)
                if recorder is not None:
                    recorder.capture_frame()

                state = deepcopy(next_state)
                self.high_level_timer -= 1

                if done:
                    break

            # Update the high-level planner after the context ends
            if self.high_level_timer == 0:
                self.high_level_planner.update(state, self.current_goal, reward, next_state)

        # Close the recorder if it was used
        if recorder is not None:
            recorder.close()
            del recorder

        self.env.close()
        stats = self.low_level_planner.return_stats()
        return total_reward, total_steps, stats

    def _add_action_noise(self, action, noise):
        """
        Add noise to the actions for exploration.

        Args:
            action (torch.Tensor): The original action tensor.
            noise (float): The noise scale.

        Returns:
            torch.Tensor: The action tensor with added noise.
        """
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
