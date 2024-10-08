import torch
import torch.nn as nn

from pmbrl.control.planner import Planner

class LowLevelPlanner(nn.Module):
    #TODO: Verify measures for low-level planner
    """
    Low-level planner responsible for generating actions that bring the agent
    closer to achieving the high-level goal provided by the HighLevelPlanner.
    """

    def __init__(self, env, ensemble, action_model, ensemble_size, plan_horizon, action_noise_scale, device="cpu"):
        super().__init__()
        self.env = env
        self.ensemble = ensemble
        self.action_model = action_model
        self.ensemble_size = ensemble_size
        self.plan_horizon = plan_horizon
        self.action_noise_scale = action_noise_scale
        self.device = device

    def forward(self, state, goal):
        """
        Generate actions based on the current state and goal.

        Args:
            state (torch.Tensor): The current state of the environment.
            goal (torch.Tensor): The desired goal state.

        Returns:
            action (torch.Tensor): The generated action.
            reward (torch.Tensor): The negative distance between the predicted next state and the goal.
        """

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(goal, torch.Tensor):
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        # Reshape the state and goal tensors for the ensemble model
        state = state.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.ensemble_size, 1, 1)
        goal = goal.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        # Predict an action given the current state and goal
        action = self.action_model(state, goal)

        # Convert the numpy arrays to tensors
        min_action = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=action.device)
        max_action = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=action.device)

        # Clamp the action tensor
        action = torch.clamp(action, min=min_action, max=max_action)

        # Predict the next state using the ensemble model
        predicted_next_state, _ = self.ensemble(state, action)

        # Calculate the reward as the negative distance to the goal
        reward = -torch.norm(predicted_next_state - goal, p=2, dim=-1).mean()

        # To return a single action, take the mean of the actions across the ensemble
        action = action.mean(dim=0).squeeze(dim=0)

        return action, reward


    def perform_rollout(self, current_state, goal):
        """
        Perform a rollout simulation to evaluate the actions against the goal.

        Args:
            current_state (torch.Tensor): The current state tensor.
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
        current_state = current_state.repeat(self.ensemble_size, 1, 1)
        states[0] = current_state

        # Simulate each time step in the planning horizon
        for t in range(self.plan_horizon):
            action = self.action_model(states[t], goal)
            predicted_next_state, delta_var = self.ensemble(states[t], action)
            delta_means[t + 1] = predicted_next_state - states[t]
            delta_vars[t + 1] = delta_var
            states[t + 1] = predicted_next_state

        # Stack and return the final states, delta means, and delta variances
        states = torch.stack(states[1:], dim=0)
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        return states, delta_vars, delta_means
