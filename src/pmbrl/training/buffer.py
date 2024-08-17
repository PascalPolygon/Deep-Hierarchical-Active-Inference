import torch
import numpy as np

class Buffer(object):
    """
    Buffer for storing experiences (state, action, reward, next state, and goal) during training.
    """

    def __init__(
        self,
        state_size,
        action_size,
        goal_size,
        ensemble_size,
        normalizer,
        signal_noise=None,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        """
        Initialize the Buffer.

        Args:
            state_size (int): Size of the state vector.
            action_size (int): Size of the action vector.
            goal_size (int): Size of the goal vector.
            ensemble_size (int): Number of models in the ensemble.
            normalizer (Normalizer): Normalizer for the states, actions, and goals.
            signal_noise (float, optional): Noise scale to add to the states.
            buffer_size (int, optional): Maximum size of the buffer.
            device (str, optional): Device to run computations on ("cpu" or "cuda").
        """
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device

        # Initialize arrays to store experiences
        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.goals = np.zeros((buffer_size, goal_size))
        self.state_deltas = np.zeros((buffer_size, state_size))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state, goal=None):
        """
        Add a new experience to the buffer.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state after the action.
            goal (np.ndarray, optional): The high-level goal at the time of the action.
        """
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        if goal is not None:
            self.goals[idx] = goal
        self._total_steps += 1

        # Update the normalizer with the new experience
        self.normalizer.update(state, action, state_delta, goal)


    def get_train_batches(self, batch_size):
        """
        Get batches of experiences for training.

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

            # Extract batches from the buffer
            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            # Convert to torch tensors
            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)

            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            # Reshape for ensemble processing
            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(
                self.ensemble_size, batch_size, self.state_size
            )

            yield states, actions, rewards, state_deltas

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of experiences currently stored.
        """
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        """
        Get the total number of steps taken so far.

        Returns:
            int: The total number of steps.
        """
        return self._total_steps
