import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def swish(x):
    """
    Swish activation function.
    """
    return x * torch.sigmoid(x)

class EnsembleDenseLayer(nn.Module):
    """
    A dense layer for ensemble models that supports a swish or linear activation function.
    """

    def __init__(self, in_size, out_size, ensemble_size, act_fn="swish"):
        """
        Initialize the EnsembleDenseLayer.

        Args:
            in_size (int): Input dimension size.
            out_size (int): Output dimension size.
            ensemble_size (int): Number of ensemble members.
            act_fn (str): Activation function name ("swish" or "linear").
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.act_fn_name = act_fn
        self.act_fn = self._get_act_fn(self.act_fn_name)
        self.reset_parameters()

    def forward(self, x):
        """
        Forward pass through the dense layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation and activation function.
        """
        x = x.to(self.weights.dtype)  # Ensure the input tensor has the same dtype as the weights
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def reset_parameters(self):
        """
        Initialize the weights and biases of the layer.
        """
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, act_fn_name):
        """
        Initialize the weights using Xavier initialization.

        Args:
            weight (torch.Tensor): Weight tensor.
            act_fn_name (str): Name of the activation function ("swish" or "linear").
        """
        if act_fn_name == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn_name == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn_name):
        """
        Retrieve the activation function based on its name.

        Args:
            act_fn_name (str): Name of the activation function ("swish" or "linear").

        Returns:
            Callable: Activation function.
        """
        if act_fn_name == "swish":
            return swish
        elif act_fn_name == "linear":
            return lambda x: x

class EnsembleModel(nn.Module):
    """
    High-level ensemble model that predicts state changes given the current state and goal.
    """

    def __init__(self, in_size, out_size, hidden_size, ensemble_size, normalizer, device="cpu"):
        """
        Initialize the HighLevelEnsembleModel.

        Args:
            in_size (int): Dimension of the state space + goal space.
            out_size (int): Dimension of the goal space.
            hidden_size (int): Dimension of the hidden layers.
            ensemble_size (int): Number of ensemble members.
            normalizer (Normalizer): Normalizer for the input data.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        super().__init__()
        self.fc_1 = EnsembleDenseLayer(in_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_2 = EnsembleDenseLayer(hidden_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_3 = EnsembleDenseLayer(hidden_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_4 = EnsembleDenseLayer(hidden_size, out_size * 2, ensemble_size, act_fn="linear")

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.to(device)

    def forward(self, states, goals):
        """
        Forward pass through the high-level ensemble model.

        Args:
            states (torch.Tensor): Input states.
            goals (torch.Tensor): Input goals.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the predicted state deltas.
        """
        norm_states, norm_goals = self._pre_process_model_inputs(states, goals)
        norm_delta_mean, norm_delta_var = self._propagate_network(norm_states, norm_goals)
        delta_mean, delta_var = self._post_process_model_outputs(norm_delta_mean, norm_delta_var)
        return delta_mean, delta_var

    def loss(self, states, goals, goal_deltas):
        """
        Compute the loss for the high-level ensemble model.

        Args:
            states (torch.Tensor): Input states.
            goals (torch.Tensor): Input goals.
            goal_deltas (torch.Tensor): Target state deltas.

        Returns:
            torch.Tensor: Loss value.
        """
        states, goals = self._pre_process_model_inputs(states, goals)
        delta_targets = self._pre_process_model_targets(goal_deltas)
        delta_mu, delta_var = self._propagate_network(states, goals)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss
    
    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, goals):
        """
        Forward pass through the network layers.

        Args:
            states (torch.Tensor): Normalized states.
            goals (torch.Tensor): Normalized goals.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the state deltas.
        """
        print(f'states.shape: {states.shape}')
        print(f'goals.shape: {goals.shape}')
        inp = torch.cat((states, goals), dim=2)  # Concatenate states and goals
        op = self.fc_1(inp)  # First hidden layer
        op = self.fc_2(op)   # Second hidden layer
        op = self.fc_3(op)   # Third hidden layer
        op = self.fc_4(op)   # Output layer

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _pre_process_model_inputs(self, states, goals):
        """
        Normalize the states and goals.

        Args:
            states (torch.Tensor): Input states.
            goals (torch.Tensor): Input goals.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized states and goals.
        """
        states = self.normalizer.normalize_states(states)
        goals = self.normalizer.normalize_goals(goals)
        return states, goals

    def _pre_process_model_targets(self, goal_deltas):
        """
        Normalize the target state deltas.

        Args:
            goal_deltas (torch.Tensor): Target state deltas.

        Returns:
            torch.Tensor: Normalized target state deltas.
        """
        return self.normalizer.normalize_goal_deltas(goal_deltas)

    def _post_process_model_outputs(self, delta_mean, delta_var):
        """
        Denormalize the model outputs (mean and variance of state deltas).

        Args:
            delta_mean (torch.Tensor): Predicted mean state deltas.
            delta_var (torch.Tensor): Predicted variance of state deltas.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Denormalized mean and variance.
        """
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var

class LowLevelEnsembleModel(nn.Module):
    """
    Low-level ensemble model that predicts state changes given the current state and action.
    """

    def __init__(self, state_size, action_size, hidden_size, ensemble_size, normalizer, device="cpu"):
        """
        Initialize the LowLevelEnsembleModel.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            hidden_size (int): Dimension of the hidden layers.
            ensemble_size (int): Number of ensemble members.
            normalizer (Normalizer): Normalizer for the input data.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        super().__init__()
        self.fc_1 = EnsembleDenseLayer(state_size + action_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_2 = EnsembleDenseLayer(hidden_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_3 = EnsembleDenseLayer(hidden_size, hidden_size, ensemble_size, act_fn="swish")
        self.fc_4 = EnsembleDenseLayer(hidden_size, state_size * 2, ensemble_size, act_fn="linear")

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.to(device)

    def forward(self, states, actions):
        """
        Forward pass through the low-level ensemble model.

        Args:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Input actions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the predicted state deltas.
        """
        norm_states, norm_actions = self._pre_process_model_inputs(states, actions)
        norm_delta_mean, norm_delta_var = self._propagate_network(norm_states, norm_actions)
        delta_mean, delta_var = self._post_process_model_outputs(norm_delta_mean, norm_delta_var)
        return delta_mean, delta_var

    def loss(self, states, actions, state_deltas):
        """
        Compute the loss for the low-level ensemble model.

        Args:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Input actions.
            state_deltas (torch.Tensor): Target state deltas.

        Returns:
            torch.Tensor: Loss value.
        """
        states, actions = self._pre_process_model_inputs(states, actions)
        delta_targets = self._pre_process_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        """
        Forward pass through the network layers.

        Args:
            states (torch.Tensor): Normalized states.
            actions (torch.Tensor): Normalized actions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the state deltas.
        """
        inp = torch.cat((states, actions), dim=2)  # Concatenate states and actions
        op = self.fc_1(inp)  # First hidden layer
        op = self.fc_2(op)   # Second hidden layer
        op = self.fc_3(op)   # Third hidden layer
        op = self.fc_4(op)   # Output layer

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _pre_process_model_inputs(self, states, actions):
        """
        Normalize the states and actions.

        Args:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Input actions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized states and actions.
        """
        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        """
        Normalize the target state deltas.

        Args:
            state_deltas (torch.Tensor): Target state deltas.

        Returns:
            torch.Tensor: Normalized target state deltas.
        """
        return self.normalizer.normalize_state_deltas(state_deltas)

    def _post_process_model_outputs(self, delta_mean, delta_var):
        """
        Denormalize the model outputs (mean and variance of state deltas).

        Args:
            delta_mean (torch.Tensor): Predicted mean state deltas.
            delta_var (torch.Tensor): Predicted variance of state deltas.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Denormalized mean and variance.
        """
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var

class RewardModel(nn.Module):
    def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)
        self.reset_parameters()
        self.to(device)

    def forward(self, states, goals):
        # print("in_size: ", self.in_size)
        # print("states.shape: ", states.shape)
        # print("actions.shape: ", goals.shape)
        inp = torch.cat((states, goals), dim=-1)
        reward = self.act_fn(self.fc_1(inp))
        reward = self.act_fn(self.fc_2(reward))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, goals, rewards):
        r_hat = self(states, goals)
        return F.mse_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.to(self.device)

class ActionModel(nn.Module):
    def __init__(self, state_size, goal_size, action_size, hidden_size, device="cpu"):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_size + goal_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.device = device
        self.to(device)

    def forward(self, state, goal):
          # Convert state and goal to tensors if they are numpy arrays
        # Debugging: Print the shapes of state and goal
        # print(f"State shape: {state.shape}, Goal shape: {goal.shape}")
        x = torch.cat([state, goal], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Use tanh to bound the action space
        return action

    def loss(self, states, goals, actions):
        predicted_actions = self.forward(states, goals)
        return F.mse_loss(predicted_actions, actions)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()