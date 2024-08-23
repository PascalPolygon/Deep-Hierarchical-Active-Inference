import torch

class HierarchicalTrainer(object):
    """
    Trainer class for hierarchical active inference agents.
    This class separately trains the high-level and low-level models using a shared buffer.
    """

    def __init__(
        self,
        high_level_ensemble_model,
        high_level_reward_model,
        low_level_action_model,
        buffer,
        n_train_epochs,
        batch_size,
        learning_rate,
        epsilon,
        grad_clip_norm,
        logger=None,
        device="cpu",
    ):
        """
        Initialize the HierarchicalTrainer.

        Args:
            high_level_ensemble_model (nn.Module): The high-level ensemble model.
            high_level_reward_model (nn.Module): The high-level reward model.
            low_level_action_model (nn.Module): The low-level action model.
            buffer (Buffer): Shared buffer for storing high-level and low-level experiences.
            n_train_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            epsilon (float): Epsilon value for the Adam optimizer.
            grad_clip_norm (float): Gradient clipping norm.
            logger (Logger, optional): Logger for recording training information.
        """
        self.high_level_ensemble = high_level_ensemble_model
        self.high_level_reward = high_level_reward_model
        self.low_level_action = low_level_action_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger
        self.device = device

        # Separate optimizers for high-level and low-level models
        self.high_level_params = list(high_level_ensemble_model.parameters()) + list(high_level_reward_model.parameters())
        self.low_level_params = list(low_level_action_model.parameters())
        self.high_level_optim = torch.optim.Adam(self.high_level_params, lr=learning_rate, eps=epsilon)
        self.low_level_optim = torch.optim.Adam(self.low_level_params, lr=learning_rate, eps=epsilon)

    def train(self):
        """
        Main training loop that iterates over the specified number of epochs.
        Separately trains the high-level and low-level models.
        """
        high_level_e_losses = []
        high_level_r_losses = []
        low_level_a_losses = []
        n_batches_high = []
        n_batches_low = []

        for epoch in range(1, self.n_train_epochs + 1):
            high_level_e_losses.append([])
            high_level_r_losses.append([])
            low_level_a_losses.append([])
            n_batches_high.append(0)
            n_batches_low.append(0)

            # Train high-level models
            self._train_high_level(epoch, high_level_e_losses, high_level_r_losses, n_batches_high)

            # Train low-level action model
            self._train_low_level(epoch, low_level_a_losses, n_batches_low)

        return (
            self._get_avg_loss(high_level_e_losses, n_batches_high, epoch), # high-level ensemble loss
            self._get_avg_loss(high_level_r_losses, n_batches_high, epoch), # high-level reward loss
            self._get_avg_loss(low_level_a_losses, n_batches_low, epoch), # low-level action model loss
        )

    def _train_high_level(self, epoch, e_losses, r_losses, n_batches):
        """
        Training loop for the high-level models.

        Args:
            epoch (int): Current training epoch.
            e_losses (list): List to store ensemble model losses.
            r_losses (list): List to store reward model losses.
            n_batches (list): List to store the number of batches per epoch.
        """
        for (states, goals, rewards, goal_deltas) in self.buffer.get_high_level_train_batches(self.batch_size):
            self.high_level_ensemble.train()
            self.high_level_reward.train()

            # Zero the gradients for the high-level models
            self.high_level_optim.zero_grad()

            # Compute losses for ensemble and reward models
            e_loss = self.high_level_ensemble.loss(states, goals, goal_deltas)
            r_loss = self.high_level_reward.loss(states, goals, rewards)
            
            # Backpropagate the losses
            (e_loss + r_loss).backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.high_level_params, self.grad_clip_norm, norm_type=2)
            
            # Perform a gradient update step
            self.high_level_optim.step()

            e_losses[epoch - 1].append(e_loss.item())
            r_losses[epoch - 1].append(r_loss.item())
            n_batches[epoch - 1] += 1

        # Log the average losses every 20 epochs
        if self.logger is not None and epoch % 20 == 0:
            avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
            avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
            message = "> High-Level Train epoch {} [ensemble {:.2f} | reward {:.2f}]"
            self.logger.log(message.format(epoch, avg_e_loss, avg_r_loss))

    def _train_low_level(self, epoch, a_losses, n_batches):
        """
        Training loop for the low-level action model.

        Args:
            epoch (int): Current training epoch.
            a_losses (list): List to store action model losses.
            n_batches (list): List to store the number of batches per epoch.
        """
        for (states, goals, next_states) in self.buffer.get_low_level_train_batches(self.batch_size):
            self.low_level_action.train()

            # Zero the gradients for the low-level action model
            self.low_level_optim.zero_grad()

            # Compute loss for the low-level action model
            a_loss = self.low_level_action.loss(states, goals, self.high_level_ensemble)

            # Backpropagate the loss
            a_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.low_level_params, self.grad_clip_norm, norm_type=2)
            
            # Perform a gradient update step
            self.low_level_optim.step()

            a_losses[epoch - 1].append(a_loss.item())
            n_batches[epoch - 1] += 1


        # Log the average loss every 20 epochs
        if self.logger is not None and epoch % 20 == 0:
            avg_a_loss = self._get_avg_loss(a_losses, n_batches, epoch)
            message = "> Low-Level Train epoch {} [action {:.2f}]"
            self.logger.log(message.format(epoch, avg_a_loss))

    def reset_models(self):
        """
        Resets the parameters of both high-level and low-level models,
        and reinitializes the optimizers.
        """
        self.high_level_ensemble.reset_parameters()
        self.high_level_reward.reset_parameters()
        self.low_level_action.reset_parameters()

        self.high_level_params = list(self.high_level_ensemble.parameters()) + list(self.high_level_reward.parameters())
        self.low_level_params = list(self.low_level_action.parameters())

        self.high_level_optim = torch.optim.Adam(self.high_level_params, lr=self.learning_rate, eps=self.epsilon)
        self.low_level_optim = torch.optim.Adam(self.low_level_params, lr=self.learning_rate, eps=self.epsilon)
    
    @staticmethod
    def _get_avg_loss(losses, n_batches, epoch):
        """
        Computes the average loss over all batches for the given epoch.

        Args:
            losses (list): List of loss values for each batch.
            n_batches (list): List of the number of batches per epoch.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        return sum(losses[epoch - 1]) / n_batches[epoch - 1] if n_batches[epoch - 1] > 0 else float("inf")
