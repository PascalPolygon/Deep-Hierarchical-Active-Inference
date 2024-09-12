import gym
import numpy as np

SPARSE_MOUNTAIN_CAR = "SparseMountainCar"
HALF_CHEETAH_RUN = "HalfCheetahRun"
HALF_CHEETAH_FLIP = "HalfCheetahFlip"
ANT_MAZE = "AntMaze"
DM_REACHER = "DeepMindReacher"
DM_CATCH = "DeepMindCatch"


class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None):
        self._env = self._get_env_object(env_name)
        self.max_reward_state = self._get_max_reward_state(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        return state

    def step(self, action):
        reward = 0
        for i in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            # self.t += 1
            if i == self.action_repeat - 1:
                self.t += 1 # Only increment t once per action
            print(f"self t: {self.t} / max_episode_len: {self.max_episode_len}")

            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break
        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env


    def _get_env_object(self, env_name):
        if env_name == SPARSE_MOUNTAIN_CAR:
            from pmbrl.envs.envs.mountain_car import SparseMountainCarEnv

            return SparseMountainCarEnv()

        elif env_name == HALF_CHEETAH_RUN:
            from pmbrl.envs.envs.half_cheetah_run import HalfCheetahRunEnv

            return HalfCheetahRunEnv()

        elif env_name == HALF_CHEETAH_FLIP:
            from pmbrl.envs.envs.half_cheetah_flip import HalfCheetahFlipEnv

            return HalfCheetahFlipEnv()

        elif env_name == ANT_MAZE:
            from pmbrl.envs.envs.ant import SparseAntEnv

            return SparseAntEnv()

        elif env_name == DM_CATCH:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="ball_in_cup", task="catch")

        elif env_name == DM_REACHER:
            from pmbrl.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="reacher", task="easy")
        else:
            return gym.make(env_name)
        
    def _get_max_reward_state(self, env_name):
        if env_name == SPARSE_MOUNTAIN_CAR:
            return np.array([-0.5, 0])
        elif env_name == HALF_CHEETAH_RUN:
            return np.array([1, 0, 0])  # Replace with actual max reward state
        elif env_name == HALF_CHEETAH_FLIP:
            return np.array([0, 0, 0])  # Replace with actual max reward state
        elif env_name == ANT_MAZE:
            return np.array([0, 0])  # Replace with actual max reward state
        elif env_name == DM_REACHER:
            return np.array([1, 1])  # Replace with actual max reward state
        elif env_name == DM_CATCH:
            return np.array([0, 0])  # Replace with actual max reward state
        else:
            # Default max reward state for Pendulum or other unknown environments
            return np.array([1, 0, 0])
