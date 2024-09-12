from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import cv2
from PIL import Image, ImageDraw

# TODO: Ensure we're only using torch tensors and not numpy arrays
class HierarchicalAgent(object):
    """
    Hierarchical agent that manages the interaction between the high-level and low-level planners.
    The agent controls the flow of information and decisions during an episode.
    """

    def __init__(self, env, high_level_planner, low_level_planner, context_length=1, logger=None, exploration_measure=None, device="cpu"):
        """
        Initialize the HierarchicalAgent.

        Args:
            env (gym.Env): The environment in which the agent operates.
            high_level_planner (HighLevelPlanner): The high-level planner.
            low_level_planner (LowLevelPlanner): The low-level planner.
            context_length (int, optional): The number of steps a goal is active before resampling.
            logger (Logger, optional): Logger for recording training information.
            exploration_measure (object, optional): The exploration measure to use (Variance, Disagreement, InformationGain).
        """
        self.env = env
        self.high_level_planner = high_level_planner
        self.low_level_planner = low_level_planner
        self.context_length = context_length
        self.logger = logger
        self.current_goal = None
        self.next_goal = None
        self.device = device

    def get_seed_episodes(self, buffer, n_episodes):
        """
        Collect seed episodes by interacting with the environment using random actions.

        Args:
            buffer (Buffer): The buffer to store experiences.
            n_episodes (int): Number of episodes to collect.

        Returns:
            Buffer: The buffer filled with seed episodes.
        """
        print(f"Collecting {n_episodes} seed episodes.")
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            step = 0
            transitions = []

            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)

                # Store transitions temporarily
                transitions.append((deepcopy(state), action, reward, deepcopy(next_state)))

                # After every context_length steps, retroactively compute the goals and store in buffer
                if step % self.context_length == 0 or done:
                    if step > 0:
                        buffer.update()

                    if len(transitions) == self.context_length:
                        final_goal = deepcopy(next_state)
                        
                        goals = [final_goal]
                        for i in range(self.context_length - 1, 0, -1):
                            prev_goal = goals[-1] + transitions[i][0] - transitions[i-1][0]
                            goals.append(prev_goal)
                        goals.reverse()

                        # Add the transitions and goals to the buffer
                        for i in range(self.context_length):
                            state, action, reward, next_state = transitions[i]
                            next_goal = goals[i]
                            buffer.add(state, goals[i], action, reward, next_state, next_goal)
                            if self.logger:
                                self.logger.log(f"Low-level transition added: state={state}, goal={goals[i]}, action={action}, reward={reward}, next_state={next_state}, next_goal={next_goal}")

                    transitions = []  # Clear transitions for the next context

                state = deepcopy(next_state)
                step += 1

                if done:
                    break

            if self.logger:
                self.logger.log(f"Episode {episode+1}/{n_episodes} complete.")
        print (f"Buffer size: {len(buffer)}")
        print(f'Step count: {buffer._total_steps}')
        print(f"High-level goals: {len(buffer.high_level_goals)}")
        return buffer

    def run_episode(self, buffer=None, action_noise=None, recorder=None):
        """
        Run a single episode in the environment, using the hierarchical planners.
        """
        total_reward = 0
        total_steps = 0
        done = False
        step = 0

        # Check the environment for max episode length
        if hasattr(self.env.unwrapped, '_max_episode_steps'):
            max_episode_length = self.env.unwrapped._max_episode_steps
        else:
            max_episode_length = "Unknown"

        print(f"Max episode length: {max_episode_length}")

        # Prepare folder for saving frames
        if recorder is not None:
            folder_name = os.path.splitext(recorder.path)[0]  # Remove extension from video path
            os.makedirs(folder_name, exist_ok=True)  # Create folder with the same name as the video file

        with torch.no_grad():
            state = self.env.reset()

            while not done:
                if step % self.context_length == 0:
                    # Sample new high-level goal from the planner at the beginning of each context
                    self.current_goal = self.high_level_planner(state)

                    print(f"Context length: {self.context_length}")
                    print(f"New goal at step: {step} - {self.current_goal}")
                    print(f"Current state: {state}")

                    if step > 0:
                        # corrected_goal = self.off_policy_goal_correction(buffer, state, exploration_scale=self.high_level_planner.expl_scale)
                        corrected_goal = self.current_goal
                        print(f"Corrected goal at step: {step} - {corrected_goal}")
                        buffer.update(corrected_goal)

                action, low_level_reward = self.low_level_planner(state, self.current_goal)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)

                if not isinstance(self.current_goal, np.ndarray):
                    self.current_goal = np.array(self.current_goal)

                self.next_goal = state + self.current_goal - next_state
                total_reward += reward
                total_steps += 1

                if buffer is not None:
                    buffer.add(deepcopy(state), deepcopy(self.current_goal), action, reward, deepcopy(next_state), deepcopy(self.next_goal))
         
                if recorder is not None:
                    # Capture the frame for video
                    print(f"Recording frame {step}, done: {done}")

                    recorder.capture_frame()

                state = deepcopy(next_state)

                self.render_and_save_frame(state, action, folder_name, step)

                self.current_goal = deepcopy(self.next_goal)
                step += 1

                if done:
                    break

            if recorder is not None:
                recorder.close()
                del recorder

            self.env.close()
            stats = self.high_level_planner.return_stats()
            if self.logger:
                self.logger.log(f"Episode complete. Total reward: {total_reward}, Total steps: {total_steps}")
            return total_reward, total_steps, stats
        
    def _add_action_noise(self, action, noise):
        """
        Add noise to the actions for exploration.

        Args:
            action (torch.Tensor): The original action tensor.
            noise (float): The noise scale.

        Returns:
            torch.Tensor: The action tensor with added noise.
        """
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
    


    #TODO: BIG ONE!!!! Check this logic to ensure it is actually doing maximum likelihood relabling.
    # Lots of hacks here!
    def off_policy_goal_correction(self, buffer, state, use_exploration=True, exploration_scale=1.0):
        """
        Implements off-policy goal correction with an optional exploration objective.

        Args:
            buffer (Buffer): The buffer containing the high-level transitions.
            state (torch.Tensor): The current state.
            use_exploration (bool): Whether to include an exploration objective (information gain).
            exploration_scale (float): Weight for the exploration term in the combined objective.

        Returns:
            torch.Tensor: The corrected goal that maximizes the combined objective.
        """
        state = torch.from_numpy(state).float().to(self.device)
        candidate_goals = []

        # Ensure that current_goal is a torch.Tensor
        if not isinstance(self.current_goal, torch.Tensor):
            self.current_goal = torch.tensor(self.current_goal, dtype=torch.float32, device=self.device)

        candidate_goals.append(self.current_goal)
        candidate_goals.append(torch.tensor(buffer.high_level_next_states[-1] - buffer.high_level_states[-1], dtype=torch.float32, device=self.device))

        mean = candidate_goals[1]
        std_dev = 0.5 * torch.std(self.current_goal).item() * torch.ones_like(mean)
        for _ in range(8):
            sampled_goal = torch.normal(mean, std_dev).to(self.device)
            candidate_goals.append(sampled_goal)

        # Convert numpy arrays to torch tensors
        obs_low = torch.tensor(self.env.observation_space.low, dtype=torch.float32, device=self.device)
        obs_high = torch.tensor(self.env.observation_space.high, dtype=torch.float32, device=self.device)

        # Ensure all candidate goals are torch tensors and clamp them
        candidate_goals = [torch.clamp(goal, min=obs_low, max=obs_high) for goal in candidate_goals]

        # Stack the candidate goals into a single tensor and match dimensions
        candidate_goals = torch.stack(candidate_goals)
        candidate_goals = candidate_goals.unsqueeze(0).repeat(self.high_level_planner.plan_horizon, self.high_level_planner.plan_horizon, 1)

        # Expand the state dimensions to match the format expected by perform_rollout
        state = state.unsqueeze(0).unsqueeze(0).repeat(self.high_level_planner.ensemble_size, self.high_level_planner.plan_horizon, 1)

        best_goal = None
        best_objective = float('-inf')
        for candidate_goal in candidate_goals:
            log_prob = 0
            for i in range(len(buffer.low_level_actions)):
                action = torch.tensor(buffer.low_level_actions[i], dtype=torch.float32, device=self.device)  # Convert action to torch.Tensor
                state = buffer.low_level_states[i]
                # print(f"Low level state (correction): {state.shape}")
                # print(f"Low level goal (correction): {candidate_goal[0].shape}")
                predicted_action = self.low_level_planner(state, candidate_goal[0])
                log_prob += -0.5 * torch.norm(action - predicted_action[0], p=2).item() ** 2

            if use_exploration is not None:
             
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                # First, reshape the candidate_goal to separate the first dimension into two dimensions [5, 10, 3]

                candidate_goal = candidate_goal.view(self.high_level_planner.plan_horizon, 10, self.env.observation_space.shape[0])

                # Now, use repeat to expand the tensor to the desired shape [5, 500, 3]
                candidate_goal = candidate_goal.repeat(1, 50, 1)
                # candidate_goal = candidate_goal.repeat(1, n_episodes*max_episde_length, 1)
                # candidate_goals= candidate_goals.unsqueeze(0).repeat(self.high_level_planner.plan_horizon, self.high_level_planner.plan_horizon, 1)
                # print(f"Goals input shape in off_policy_goal_correction: {candidate_goal.shape}")
                # print(f'State input shape from off_policy_goal_correction: {state.shape}')
                _, delta_vars, delta_means= self.high_level_planner.perform_rollout(state, candidate_goal)
                exploration_bonus = self.high_level_planner.measure(delta_means, delta_vars).sum().item()
            else:
                exploration_bonus = 0

            objective = log_prob + exploration_scale * exploration_bonus

            if objective > best_objective:
                best_objective = objective
                best_goal = candidate_goal

        # if self.logger:
        #     self.logger.log(f"Off-policy goal correction: Selected goal={best_goal} with objective={best_objective}")

        # return best_goal
        return best_goal[0][0]
    

    def render_and_save_frame(self, state, action, folder_name, step):
        """
        Render the current frame, paint the goal and markers, and save the image.

        Args:
        state: Current state of the environment
        folder_name: Directory to save the frames
        step: Current step number
        """
        # Render the frame
        frame = self.env.unwrapped.render(mode="rgb_array")
        goal = self.current_goal

        if frame is not None:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Correct calculation for the goal state using cos(theta) and sin(theta)
            # We assume the goal is represented in the same way as the pendulum's state: [cos(goal_theta), sin(goal_theta)]
            goal_theta = np.arctan2(goal[1], goal[0])

            # Flip the sign of goal_theta to correct the orientation
            goal_theta = -goal_theta

            # Calculate goal position in the image
            goal_x = int(width / 2 + np.sin(goal_theta) * (height / 4))  # Similar logic to the pendulum's x position
            goal_y = int(height / 2 - np.cos(goal_theta) * (height / 4))  # Similar logic to the pendulum's y position

            # Draw the goal state on the image (green)
            draw.ellipse([goal_x-5, goal_y-5, goal_x+5, goal_y+5], fill=(0, 255, 0), outline=(0, 255, 0))

            # Draw markers at (0,0) and (1000,1000) (red) - Reference points
            draw.ellipse([0-5, 0-5, 0+5, 0+5], fill=(255, 0, 0), outline=(255, 0, 0))
            draw.ellipse([1000-5, 1000-5, 1000+5, 1000+5], fill=(255, 0, 0), outline=(255, 0, 0))

            # Corrected calculation for the pendulum's free end (orange)
            # For Pendulum-v0, state[0] is cos(theta), state[1] is sin(theta)
            theta = np.arctan2(state[1], state[0])

            # Flip the sign of theta to correct the orientation
            theta = -theta

            # Calculate pendulum's free end position, bringing the orange dot closer by reducing the height scaling
            pendulum_x = int(width / 2 + np.sin(theta) * (height / 4))  # Reduced length factor from height / 3 to height / 4
            pendulum_y = int(height / 2 - np.cos(theta) * (height / 4))  # Reduced length factor here as well

            # Debugging print statements to check positions
            print(f"Step: {step}, Pendulum position: ({pendulum_x}, {pendulum_y}), State: {state}, Action: {action}")
            print(f'Goal position: ({goal_x}, {goal_y}), Goal state : {self.current_goal}')

            # Draw the current state (pendulum's free end) (orange)
            draw.ellipse([pendulum_x-5, pendulum_y-5, pendulum_x+5, pendulum_y+5], fill=(255, 165, 0), outline=(255, 165, 0))

            # Save the modified frame as an image
            img.save(f"{folder_name}/frame_{step}.png")
        else:
            print(f"Warning: Frame {step} could not be rendered.")




    # def render_and_save_frame(self, state, folder_name, step):
    #     """
    #     Render the current frame, paint the goal and markers, and save the image.
        
    #     Args:
    #     state: Current state of the environment
    #     folder_name: Directory to save the frames
    #     step: Current step number
    #     """
    #     # Render the frame
    #     frame = self.env.unwrapped.render(mode="rgb_array")
    #     goal = self.current_goal
        
    #     if frame is not None:
    #         img = Image.fromarray(frame)
    #         draw = ImageDraw.Draw(img)
            
    #         # Convert the goal state to pixel coordinates (you may need to adjust this conversion)
    #         height, width = frame.shape[:2]
    #         goal_x = int((goal[0] + np.pi) / (2 * np.pi) * width)
    #         goal_y = int((goal[1] + 8) / 16 * height)
            
    #         # Draw the goal state on the image (green)
    #         draw.ellipse([goal_x-5, goal_y-5, goal_x+5, goal_y+5], fill=(0, 255, 0), outline=(0, 255, 0))
            
    #         # Draw markers at (0,0) and (1000,1000) (red)
    #         draw.ellipse([0-5, 0-5, 0+5, 0+5], fill=(255, 0, 0), outline=(255, 0, 0))
    #         draw.ellipse([1000-5, 1000-5, 1000+5, 1000+5], fill=(255, 0, 0), outline=(255, 0, 0))
            
    #         # Draw the current state (pendulum's free end) (orange)
    #         # For Pendulum-v0, state[0] is cos(theta), state[1] is sin(theta)
    #         theta = np.arctan2(state[1], state[0])
    #         pendulum_x = int(width / 2 + np.sin(theta) * height / 3)
    #         pendulum_y = int(height / 2 + np.cos(theta) * height / 3)
    #         draw.ellipse([pendulum_x-5, pendulum_y-5, pendulum_x+5, pendulum_y+5], fill=(255, 165, 0), outline=(255, 165, 0))
            
    #         # Save the modified frame as an image
    #         img.save(f"{folder_name}/frame_{step}.png")
    #     else:
    #         print(f"Warning: Frame {step} could not be rendered.")


    def state_to_pixel(self, state, frame_shape):
        # This is a placeholder implementation. You need to adjust this based on your environment.
        height, width = frame_shape[:2]
        x = int(state[0] * width)  # Assuming state[0] is normalized between 0 and 1
        y = int(state[1] * height)  # Assuming state[1] is normalized between 0 and 1
        return (x, y)









