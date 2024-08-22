import torch
import torch.nn as nn

from pmbrl.control.planner import Planner

class LowLevelPlanner(Planner):
    #TODO: Verify measures for low-level planner
    """
    Low-level planner responsible for generating actions that bring the agent
    closer to achieving the high-level goal provided by the HighLevelPlanner.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the LowLevelPlanner.

        Inherits parameters from the Planner base class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, state, goal):
        """
        Generate actions and calculate the reward based on the proximity to the goal.

        Args:
            state (torch.Tensor): The current state of the environment.
            goal (torch.Tensor): The high-level goal to achieve.

        Returns:
            Tuple of torch.Tensor: The generated actions and the reward.
        """
        # Generate actions using the base Planner's forward method
        
        # This action should be conditioned on the goal
        # It should also 
        actions = super().forward(state)

        # Calculate reward based on how close actions are to achieving the goal
        reward = -torch.norm(state - goal, p=2)  # Reward shaped by distance to goal
        return actions, reward

    def perform_rollout(self, current_state, actions, goal):
        """
        Perform a rollout simulation to evaluate the actions against the goal.

        Args:
            current_state (torch.Tensor): The current state tensor.
            actions (torch.Tensor): The actions to evaluate.
            goal (torch.Tensor): The goal to achieve.

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

        # Prepare the actions tensor for the ensemble
        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        # Simulate each time step in the planning horizon
        for t in range(self.plan_horizon):
            # TODO: states must be gnerated based on goal
            # delta_mean, delta_var = self.ensemble(states[t], actions[t], goal)
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.use_mean:
                states[t + 1] = states[t] + delta_mean
            else:
                states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
                # TODO: states must be gnerated based on goal
                # states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var, goal)

            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        # Stack and return the final states, delta means, and delta variances
        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means
