import torch
import torch.nn as nn

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random

class HighLevelPlanner(nn.Module):
    #TODO: Verify measures for high-level planner
    def __init__(
        self,
        ensemble,
        goal_size,
        ensemble_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        use_exploration=True,
        expl_scale=1.0,
        strategy="information",
        device="cpu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.goal_size = goal_size
        self.ensemble_size = ensemble_size
        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates
        self.use_exploration = use_exploration
        self.expl_scale = expl_scale
        self.device = device

        # Set the exploration strategy for the high-level goals
        if strategy == "information":
            self.measure = InformationGain(self.ensemble, scale=expl_scale)
        elif strategy == "variance":
            self.measure = Variance(self.ensemble, scale=expl_scale)
        elif strategy == "random":
            self.measure = Random(self.ensemble, scale=expl_scale)
        elif strategy == "none":
            self.use_exploration = False

        # Initialize storage for trial rewards and bonuses
        self.trial_rewards = []
        self.trial_bonuses = []
        self.to(device)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        # Generate high-level goal as context
        goal_mean = torch.zeros(self.plan_horizon, 1, self.goal_size).to(self.device)
        goal_std_dev = torch.ones(self.plan_horizon, 1, self.goal_size).to(self.device)

        for _ in range(self.optimisation_iters):
            goals = goal_mean + goal_std_dev * torch.randn(
                self.plan_horizon, self.n_candidates, self.goal_size, device=self.device)

            # Ensure the sampled goals are within the environment's state bounds
            goals = torch.clamp(goals, min=self.env.observation_space.low, max=self.env.observation_space.high)

            states, delta_vars, delta_means = self.perform_rollout(state, goals)

            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                returns += expl_bonus
                self.trial_bonuses.append(expl_bonus)

            # Implement goal-based reward model
            if self.use_reward:
                _states = states.view(-1, state_size)
                _goals = goals.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _goals = _goals.view(-1, self.goal_size)
                rewards = self.reward_model(_goals)
                rewards = rewards * self.reward_scale
                rewards = rewards.view(self.plan_horizon, self.ensemble_size, self.n_candidates)
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards
                self.trial_rewards.append(rewards)

            goal_mean, goal_std_dev = self._fit_gaussian(goals, returns)

        # Return the high-level goal (context)
        return goal_mean[0].squeeze(dim=0)


    def perform_rollout(self, current_state, goals):
        T = self.plan_horizon + 1

        # Initialize lists to store states, delta means, and delta variances
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T

        # Prepare the initial state tensor
        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        # Prepare the goals tensor for the ensemble
        goals = goals.unsqueeze(0)
        goals = goals.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        # Simulate each time step in the planning horizon
        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], goals[t])
            # Update states towards achieving the goals
            states[t + 1] = states[t] + delta_mean
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        # Stack and return the final states, delta means, and delta variances
        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means

    def _fit_gaussian(self, goals, returns):
        # Replace NaN values in returns with zeros
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)

        # Select the top-performing goals based on returns
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_goals = goals[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.goal_size
        )

        # Calculate the mean and standard deviation of the best goals
        goal_mean, goal_std_dev = (
            best_goals.mean(dim=1, keepdim=True),
            best_goals.std(dim=1, unbiased=False, keepdim=True),
        )

        # Return the refined mean and standard deviation for the goal distribution
        return goal_mean, goal_std_dev

    def return_stats(self):
        if self.use_reward:
            reward_stats = self._create_stats(self.trial_rewards)
        else:
            reward_stats = {}
        if self.use_exploration:
            info_stats = self._create_stats(self.trial_bonuses)
        else:
            info_stats = {}
        self.trial_rewards = []
        self.trial_bonuses = []
        return reward_stats, info_stats

    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)
        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }

    # def update(self, state, current_goal, reward, next_state):
    #     """
    #     Update the high-level planner's goals based on agent performance.

    #     Args:
    #         state (np.ndarray): The current state of the agent.
    #         current_goal (torch.Tensor): The current high-level goal.
    #         reward (float): The reward received.
    #         next_state (np.ndarray): The state after taking the action.
    #     """
    #     state = torch.from_numpy(state).float().to(self.device)
    #     next_state = torch.from_numpy(next_state).float().to(self.device)
    #     current_goal = current_goal.to(self.device)

    #     # Compute the distance between the next state and the current goal
    #     goal_error = torch.norm(next_state - current_goal, p=2)

    #     # Update the goal based on the performance
    #     if goal_error > reward:
    #         # If the goal was not achieved, adjust the goal to be closer to the next state
    #         adjustment = 0.1 * (next_state - current_goal)
    #         new_goal = current_goal + adjustment
    #     else:
    #         # If the goal was achieved, maintain the goal or set a new one
    #         new_goal = current_goal

    #     # Log the updated goal
    #     if self.logger is not None:
    #         self.logger.log(f"Updated high-level goal: {new_goal.cpu().numpy()}")

    #     return new_goal
