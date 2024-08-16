import torch
import torch.nn as nn

from pmbrl.control.planner import Planner

class LowLevelPlanner(Planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, goal):
        # Generate actions that bring the state closer to the high-level goal
        actions = super().forward(state)

        # Adjust the reward calculation or exploration to consider the goal
        # For example, add a goal-based reward or exploration bonus
        return actions

    def perform_rollout(self, current_state, actions, goal):
        T = self.plan_horizon + 1

        # Initialize lists to store states, delta means, and delta variances
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T

        # Prepare the initial state tensor
        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        # Prepare the actions tensor for the ensemble
        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        # Simulate each time step in the planning horizon
        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.use_mean:
                states[t + 1] = states[t] + delta_mean
            else:
                states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        # Stack and return the final states, delta means, and delta variances
        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means
