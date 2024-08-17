import torch
import torch.nn as nn

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random

class HighLevelPlanner(nn.Module):
    """
    High-level planner responsible for generating high-level goals (contexts) 
    that guide the low-level planner. The planner uses a model ensemble to 
    evaluate different goal candidates and selects the most promising ones.
    """

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
        """
        Initialize the HighLevelPlanner.

        Args:
            ensemble (nn.Module): The ensemble model used for state transitions.
            goal_size (int): The dimensionality of the goal.
            ensemble_size (int): Number of models in the ensemble.
            plan_horizon (int): Number of time steps to plan ahead.
            optimisation_iters (int): Number of optimization iterations for goal selection.
            n_candidates (int): Number of goal candidates to evaluate.
            top_candidates (int): Number of top candidates to select.
            use_exploration (bool): Whether to include exploration bonuses in goal selection.
            expl_scale (float): Scaling factor for exploration bonuses.
            strategy (str): Strategy for exploration ("information", "variance", "random", or "none").
            device (str): Device to run computations on ("cpu" or "cuda").
        """
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
        """
        Generate a high-level goal based on the current state.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            torch.Tensor: The selected high-level goal.
        """
        state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        # Initialize mean and standard deviation for goal candidates
        goal_mean = torch.zeros(self.plan_horizon, 1, self.goal_size).to(self.device)
        goal_std_dev = torch.ones(self.plan_horizon, 1, self.goal_size).to(self.device)

        for _ in range(self.optimisation_iters):
            # Sample goal candidates from the current Gaussian distribution
            goals = goal_mean + goal_std_dev * torch.randn(
                self.plan_horizon, self.n_candidates, self.goal_size, device=self.device)

            # Perform rollout simulations with the goal candidates
            states, delta_vars, delta_means = self.perform_rollout(state, goals)

            # Initialize returns (rewards + exploration bonuses)
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            # Calculate exploration bonuses if enabled
            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                returns += expl_bonus
                self.trial_bonuses.append(expl_bonus)
            
            # Calculate rewards based on the goals (if reward model is used)
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

            # Refine the goal distribution based on the returns
            goal_mean, goal_std_dev = self._fit_gaussian(goals, returns)

        # Return the best goal as the high-level context
        return goal_mean[0].squeeze(dim=0)

    def perform_rollout(self, current_state, goals):
        """
        Perform a rollout simulation for the given goals.

        Args:
            current_state (torch.Tensor): The current state tensor.
            goals (torch.Tensor): The goals to evaluate.

        Returns:
            Tuple of torch.Tensor: The states, delta means, and delta variances.
        """
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
        """
        Refine the Gaussian distribution for goals based on their returns.

        Args:
            goals (torch.Tensor): The goal candidates.
            returns (torch.Tensor): The returns associated with the goal candidates.

        Returns:
            Tuple of torch.Tensor: The mean and standard deviation for the refined goal distribution.
        """
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
