import numpy as np
import torch
from copy import deepcopy

class Normalizer(object):
    """
    Normalizer for states, actions, state deltas, and goals.
    Maintains running statistics for normalization.
    """

    def __init__(self):
        """
        Initialize the Normalizer with empty statistics.
        """
        self.state_mean = None
        self.state_sk = None
        self.state_stdev = None
        self.action_mean = None
        self.action_sk = None
        self.action_stdev = None
        self.state_delta_mean = None
        self.state_delta_sk = None
        self.state_delta_stdev = None
        self.goal_mean = None
        self.goal_sk = None
        self.goal_stdev = None
        self.count = 0

    @staticmethod
    def update_mean(mu_old, addendum, n):
        """
        Update the mean with a new sample.

        Args:
            mu_old (np.ndarray): The old mean.
            addendum (np.ndarray): The new sample.
            n (int): The sample count.

        Returns:
            np.ndarray: The updated mean.
        """
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        """
        Update the sum of squares of differences from the mean.

        Args:
            sk_old (np.ndarray): The old sum of squares.
            mu_old (np.ndarray): The old mean.
            mu_new (np.ndarray): The new mean.
            addendum (np.ndarray): The new sample.

        Returns:
            np.ndarray: The updated sum of squares.
        """
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def update(self, state, action, state_delta, goal=None):
        """
        Update the normalizer with new state, action, state delta, and optionally goal.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            state_delta (np.ndarray): The change in state.
            goal (np.ndarray): The high-level goal, if provided.
        """
        self.count += 1

        if self.count == 1:
            self.state_mean = state.copy()
            self.state_sk = np.zeros_like(state)
            self.state_stdev = np.zeros_like(state)
            self.action_mean = action.copy()
            self.action_sk = np.zeros_like(action)
            self.action_stdev = np.zeros_like(action)
            self.state_delta_mean = state_delta.copy()
            self.state_delta_sk = np.zeros_like(state_delta)
            self.state_delta_stdev = np.zeros_like(state_delta)
            if goal is not None:
                self.goal_mean = goal.copy()
                self.goal_sk = np.zeros_like(goal)
                self.goal_stdev = np.zeros_like(goal)
            return

        # Store old means for update calculations
        state_mean_old = self.state_mean.copy()
        action_mean_old = self.action_mean.copy()
        state_delta_mean_old = self.state_delta_mean.copy()
        if goal is not None:
            goal_mean_old = self.goal_mean.copy()

        # Update means
        self.state_mean = self.update_mean(self.state_mean, state, self.count)
        self.action_mean = self.update_mean(self.action_mean, action, self.count)
        self.state_delta_mean = self.update_mean(self.state_delta_mean, state_delta, self.count)
        if goal is not None:
            self.goal_mean = self.update_mean(self.goal_mean, goal, self.count)

        # Update sums of squares
        self.state_sk = self.update_sk(self.state_sk, state_mean_old, self.state_mean, state)
        self.action_sk = self.update_sk(self.action_sk, action_mean_old, self.action_mean, action)
        self.state_delta_sk = self.update_sk(self.state_delta_sk, state_delta_mean_old, self.state_delta_mean, state_delta)
        if goal is not None:
            self.goal_sk = self.update_sk(self.goal_sk, goal_mean_old, self.goal_mean, goal)

        # Calculate standard deviations
        self.state_stdev = np.sqrt(self.state_sk / self.count)
        self.action_stdev = np.sqrt(self.action_sk / self.count)
        self.state_delta_stdev = np.sqrt(self.state_delta_sk / self.count)
        if goal is not None:
            self.goal_stdev = np.sqrt(self.goal_sk / self.count)

    @staticmethod
    def setup_vars(x, mean, stdev):
        """
        Prepare mean and standard deviation tensors.

        Args:
            x (torch.Tensor): The input tensor.
            mean (np.ndarray): The mean values.
            stdev (np.ndarray): The standard deviation values.

        Returns:
            Tuple of torch.Tensor: The mean and standard deviation tensors.
        """
        mean = torch.from_numpy(mean).float().to(x.device)
        stdev = torch.from_numpy(stdev).float().to(x.device)
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        """
        Normalize a tensor.

        Args:
            x (torch.Tensor): The input tensor.
            mean (torch.Tensor): The mean tensor.
            stdev (torch.Tensor): The standard deviation tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        mean, stdev = self.setup_vars(x, mean, stdev)
        n = x - mean
        n = n / torch.clamp(stdev, min=1e-8)
        return n

    def normalize_states(self, states):
        """
        Normalize the states.

        Args:
            states (torch.Tensor): The state tensor to normalize.

        Returns:
            torch.Tensor: The normalized states.
        """
        return self._normalize(states, self.state_mean, self.state_stdev)

    def normalize_actions(self, actions):
        """
        Normalize the actions.

        Args:
            actions (torch.Tensor): The action tensor to normalize.

        Returns:
            torch.Tensor: The normalized actions.
        """
        return self._normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        """
        Normalize the state deltas.

        Args:
            state_deltas (torch.Tensor): The state delta tensor to normalize.

        Returns:
            torch.Tensor: The normalized state deltas.
        """
        return self._normalize(state_deltas, self.state_delta_mean, self.state_delta_stdev)

    def denormalize_state_delta_means(self, state_deltas_means):
        """
        Denormalize state delta means.

        Args:
            state_deltas_means (torch.Tensor): The normalized state delta means.

        Returns:
            torch.Tensor: The denormalized state delta means.
        """
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return state_deltas_means * stdev + mean

    def denormalize_state_delta_vars(self, state_delta_vars):
        """
        Denormalize state delta variances.

        Args:
            state_delta_vars (torch.Tensor): The normalized state delta variances.

        Returns:
            torch.Tensor: The denormalized state delta variances.
        """
        _, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        """
        Renormalize state delta means.

        Args:
            state_deltas_means (torch.Tensor): The state delta means to renormalize.

        Returns:
            torch.Tensor: The renormalized state delta means.
        """
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return (state_deltas_means - mean) / torch.clamp(stdev, min=1e-8)

    def renormalize_state_delta_vars(self, state_delta_vars):
        """
        Renormalize state delta variances.

        Args:
            state_delta_vars (torch.Tensor): The state delta variances to renormalize.

        Returns:
            torch.Tensor: The renormalized state delta variances.
        """
        _, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars / (torch.clamp(stdev, min=1e-8) ** 2)
    
    def normalize_goals(self, goals):
        """
        Normalize the goals.

        Args:
            goals (torch.Tensor): The goal tensor to normalize.

        Returns:
            torch.Tensor: The normalized goals.
        """
        return self._normalize(goals, self.goal_mean, self.goal_stdev)

    def denormalize_goals(self, goals):
        """
        Denormalize the goals.

        Args:
            goals (torch.Tensor): The normalized goal tensor.

        Returns:
            torch.Tensor: The denormalized goals.
        """
        return goals * self.goal_stdev + self.goal_mean

    def normalize_goal_deltas(self, goal_deltas):
        """
        Normalize the goal deltas.

        Args:
            goal_deltas (torch.Tensor): The goal delta tensor to normalize.

        Returns:
            torch.Tensor: The normalized goal deltas.
        """
        return self._normalize(goal_deltas, self.goal_mean, self.goal_stdev)
    
    def renormalize_goal_deltas(self, goal_deltas):
        """
        Renormalize goal deltas.

        Args:
            goal_deltas (torch.Tensor): The goal deltas to renormalize.

        Returns:
            torch.Tensor: The renormalized goal deltas.
        """
        mean, stdev = self.setup_vars(goal_deltas, self.goal_mean, self.goal_stdev)
        return (goal_deltas - mean) / torch.clamp(stdev, min=1e-8)
    
    def denormalize_goal_deltas(self, goal_deltas):
        """
        Denormalize goal deltas.

        Args:
            goal_deltas (torch.Tensor): The normalized goal deltas.

        Returns:
            torch.Tensor: The denormalized goal deltas.
        """
        mean, stdev = self.setup_vars(goal_deltas, self.goal_mean, self.goal_stdev)
        return goal_deltas * stdev + mean
