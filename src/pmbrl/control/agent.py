from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

class HierarchicalAgent(object):
    """
    Hierarchical agent that manages the interaction between the high-level and low-level planners.
    The agent controls the flow of information and decisions during an episode.
    """

    def __init__(self, env, high_level_planner, low_level_planner, context_length=1, logger=None, exploration_measure=None):
        """
        Initialize the HierarchicalAgent.

        Args:
            env (gym.Env): The environment in which the agent operates.
            high_level_planner (HighLevelPlanner): The high-level planner.
            low_level_planner (LowLevelPlanner): The low-level planner.
            context_length (int, optional): The number of steps a goal is active before resampling.
            logger (Logger, optional): Logger for recording training information.
            exploration_measure (object, optional): The exploration measure to use (Variance, Disagreement, InformationGain).
        """
        self.env = env
        self.high_level_planner = high_level_planner
        self.low_level_planner = low_level_planner
        self.context_length = context_length
        self.logger = logger
        self.current_goal = None
        self.next_goal = None

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
            step = 0
            transitions = []

            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)

                # Store transitions temporarily
                transitions.append((deepcopy(state), action, reward, deepcopy(next_state)))

                # After every context_length steps, retroactively compute the goals and store in buffer
                if step % self.context_length == 0 or done:
                    if step > 0:
                        buffer.update()

                    if len(transitions) == self.context_length:
                        final_goal = deepcopy(next_state)
                        
                        goals = [final_goal]
                        for i in range(self.context_length - 1, 0, -1):
                            prev_goal = goals[-1] + transitions[i][0] - transitions[i-1][0]
                            goals.append(prev_goal)
                        goals.reverse()

                        # Add the transitions and goals to the buffer
                        for i in range(self.context_length):
                            state, action, reward, next_state = transitions[i]
                            next_goal = goals[i]
                            buffer.add(state, goals[i], action, reward, next_state, next_goal)
                    transitions = []  # Clear transitions for the next context

                state = deepcopy(next_state)
                step += 1

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
        step = 0

        with torch.no_grad():
            state = self.env.reset()

            while not done:
                if step % self.context_length == 0:
                    self.current_goal = self.high_level_planner(state)
                    if step > 0:
                        corrected_goal = self.off_policy_goal_correction(buffer, state, exploration_scale=self.high_level_planner.expl_scale)
                        buffer.update(corrected_goal)

                action, low_level_reward = self.low_level_planner(state, self.current_goal)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)
                self.next_goal = state + self.current_goal - next_state
                total_reward += reward
                total_steps += 1

                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log("> Step {} [reward {:.2f}]".format(total_steps, total_reward))

                if buffer is not None:
                    buffer.add(deepcopy(state), deepcopy(self.current_goal), action, reward + low_level_reward, deepcopy(next_state), deepcopy(self.next_goal))
                if recorder is not None:
                    recorder.capture_frame()

                state = deepcopy(next_state)
                self.current_goal = deepcopy(self.next_goal)
                step += 1

                if done:
                    break

            if recorder is not None:
                recorder.close()
                del recorder

            self.env.close()
            stats = self.high_level_planner.return_stats()
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
    
    def off_policy_goal_correction(self, buffer, state, use_exploration=True, exploration_scale=1.0):
        """
        Implements off-policy goal correction with an optional exploration objective.

        Args:
            buffer (Buffer): The buffer containing the high-level transitions.
            state (torch.Tensor): The current state.
            use_exploration (bool): Whether to include an exploration objective (information gain).
            exploration_scale (float): Weight for the exploration term in the combined objective.

        Returns:
            torch.Tensor: The corrected goal that maximizes the combined objective.
        """
        state = torch.from_numpy(state).float().to(self.device)
        candidate_goals = []

        candidate_goals.append(self.current_goal)
        candidate_goals.append(buffer.high_level_next_states[-1] - buffer.high_level_states[-1])

        mean = buffer.high_level_next_states[-1] - buffer.high_level_states[-1]
        std_dev = 0.5 * torch.std(self.current_goal).item() * torch.ones_like(mean)
        for _ in range(8):
            sampled_goal = torch.normal(mean, std_dev).to(self.device)
            candidate_goals.append(sampled_goal)

        candidate_goals = [torch.clamp(goal, min=self.env.observation_space.low, max=self.env.observation_space.high) for goal in candidate_goals]

        best_goal = None
        best_objective = float('-inf')
        for candidate_goal in candidate_goals:
            log_prob = 0
            for i in range(len(buffer.low_level_actions)):
                action = buffer.low_level_actions[i]
                predicted_action = self.low_level_planner(buffer.low_level_states[i], candidate_goal)
                log_prob += -0.5 * torch.norm(action - predicted_action[0], p=2).item() ** 2

            if use_exploration and self.exploration_measure is not None:
                delta_means, delta_vars = self.high_level_planner.perform_rollout(state, candidate_goal.unsqueeze(0))
                exploration_bonus = self.high_level_planner.measure(delta_means, delta_vars).sum().item()
            else:
                exploration_bonus = 0

            objective = log_prob + exploration_scale * exploration_bonus

            if objective > best_objective:
                best_objective = objective
                best_goal = candidate_goal

        return best_goal

