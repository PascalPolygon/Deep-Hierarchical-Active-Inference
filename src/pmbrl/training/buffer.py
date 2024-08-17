class Buffer(object):
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
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.goals = np.zeros((buffer_size, goal_size))
        self.state_deltas = np.zeros((buffer_size, state_size))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state, goal=None):
        idx = self._total_steps % self.buffer_size
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        if goal is not None:
            self.goals[idx] = goal
        self._total_steps += 1

        self.normalizer.update(state, action, state_delta)

    def get_train_batches(self, batch_size):
        # Existing implementation...
        pass
