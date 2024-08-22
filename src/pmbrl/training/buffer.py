import torch
import numpy as np

class Buffer(object):
    """
    Buffer for storing experiences (state, action, reward, next state, and goal) during training.
    This class now manages both a high-level and a low-level buffer.
    """

    def __init__(
        self,
        state_size,
        action_size,
        goal_size,
        ensemble_size,
        normalizer,
        context_length,
        signal_noise=None,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device
        self.context_length = context_length

        # Low-level buffer
        self.low_level_states = np.zeros((buffer_size, state_size))
        self.low_level_actions = np.zeros((buffer_size, action_size))
        self.low_level_rewards = np.zeros((buffer_size, 1))
        self.low_level_goals = np.zeros((buffer_size, goal_size))
        self.low_level_next_goals = np.zeros((buffer_size, goal_size))
        self.low_level_state_deltas = np.zeros((buffer_size, state_size))

        # High-level buffer
        self.high_level_states = []
        self.high_level_goals = []
        self.high_level_rewards = []
        self.high_level_next_states = []

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, goal, action, reward, next_state, next_goal):
        #TODO: Verify expected dimensions for state, goal, action, reward, next_state, next_goal
        """
        Add a new low-level experience to the buffer.

        Args:
            state (np.ndarray): The current state.
            goal (np.ndarray): The current goal.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state after the action.
            next_goal (np.ndarray): The next goal after the action.
        """
        idx = self._total_steps % self.buffer_size
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float().to(self.device)

        print(f"state: {state}")
        print(f"next_state: {next_state}")

        state_delta = next_state[0] - state

        self.low_level_states[idx] = state
        self.low_level_actions[idx] = action
        self.low_level_rewards[idx] = reward
        self.low_level_state_deltas[idx] = state_delta
        self.low_level_goals[idx] = goal
        self.low_level_next_goals[idx] = next_goal[0]
        self._total_steps += 1

        # Update the normalizer with the new experience
        self.normalizer.update(state, action, state_delta, goal)

    def update(self, corrected_goal=None):
        """
        Update the high-level buffer with the latest context information.
        This method assumes that the context has been fully populated in the low-level buffer.

        Args:
            corrected_goal (torch.Tensor, Optional): The corrected goal after off-policy correction.
        """
        if self._total_steps < self.context_length:
            return

        end_idx = self._total_steps - 1
        start_idx = end_idx - self.context_length + 1

        # Extract context transitions
        st = self.low_level_states[start_idx]
        if corrected_goal is not None:
            self.low_level_goals[start_idx] = corrected_goal.cpu().numpy()  # Use the corrected goal
        gt = self.low_level_goals[start_idx]
        Rt_t_c = np.sum(self.low_level_rewards[start_idx:end_idx + 1])
        st_c = self.low_level_states[end_idx]

        # Append the high-level transition tuple with corrected goal
        self.high_level_states.append(st)
        self.high_level_goals.append(gt)
        self.high_level_rewards.append(Rt_t_c)
        self.high_level_next_states.append(st_c)

    def get_train_batches(self, batch_size):
        """
        Get batches of low-level experiences for training.

        Args:
            batch_size (int): The size of the batches to return.

        Yields:
            Tuple of torch.Tensor: Batches of states, actions, rewards, and state deltas.
        """
        size = len(self)
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)

            if (j - i) < batch_size and i != 0:
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            # Extract batches from the low-level buffer
            states = self.low_level_states[batch_indices]
            actions = self.low_level_actions[batch_indices]
            rewards = self.low_level_rewards[batch_indices]
            state_deltas = self.low_level_state_deltas[batch_indices]
            goals = self.low_level_goals[batch_indices]

            # Convert to torch tensors
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)
            goals = torch.from_numpy(goals).float().to(self.device)

            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            # Reshape for ensemble processing
            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(
                self.ensemble_size, batch_size, self.state_size
            )
            goals = goals.reshape(self.ensemble_size, batch_size, self.goal_size)

            yield states, actions, rewards, state_deltas, goals
    
    def get_low_level_train_batches(self, batch_size):
        """
        Get batches of low-level experiences for training.

        Args:
            batch_size (int): The size of the batches to return.

        Yields:
            Tuple of torch.Tensor: Batches of states, goals, and actions.
        """
        size = min(self._total_steps, self.buffer_size)
        indices = np.random.permutation(size)

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)
            if (j - i) < batch_size and i != 0:
                return

            batch_indices = indices[i:j]

            # Extract batches from the low-level buffer
            states = torch.from_numpy(self.low_level_states[batch_indices]).float().to(self.device)
            goals = torch.from_numpy(self.low_level_goals[batch_indices]).float().to(self.device)
            actions = torch.from_numpy(self.low_level_actions[batch_indices]).float().to(self.device)

            if self.signal_noise is not None:
                states += self.signal_noise * torch.randn_like(states)

            # Reshape for ensemble processing
            states = states.view(self.ensemble_size, -1, self.state_size)
            goals = goals.view(self.ensemble_size, -1, self.goal_size)
            actions = actions.view(self.ensemble_size, -1, self.action_size)

            yield states, goals, actions

    def get_high_level_train_batches(self, batch_size):
        """
        Get batches of high-level experiences for training.

        Args:
            batch_size (int): The size of the batches to return.

        Yields:
            Tuple of torch.Tensor: Batches of states, goals, rewards, and state deltas.
        """
        size = len(self.high_level_states)
        indices = np.random.permutation(size)

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)
            if (j - i) < batch_size and i != 0:
                return

            batch_indices = indices[i:j]

            # Extract batches from the high-level buffer
            states = torch.from_numpy(np.array(self.high_level_states)[batch_indices]).float().to(self.device)
            goals = torch.from_numpy(np.array(self.high_level_goals)[batch_indices]).float().to(self.device)
            rewards = torch.from_numpy(np.array(self.high_level_rewards)[batch_indices]).float().to(self.device)
            next_states = torch.from_numpy(np.array(self.high_level_next_states)[batch_indices]).float().to(self.device)
            goal_deltas = next_states - states

            # Reshape for ensemble processing
            states = states.view(self.ensemble_size, -1, self.state_size)
            goals = goals.view(self.ensemble_size, -1, self.goal_size)
            rewards = rewards.view(self.ensemble_size, -1, 1)
            goal_deltas = goal_deltas.view(self.ensemble_size, -1, self.state_size)

            yield states, goals, rewards, goal_deltas

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps

