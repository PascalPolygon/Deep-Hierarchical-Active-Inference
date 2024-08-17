import torch

class HierarchicalTrainer(object):
    def __init__(
        self,
        high_level_model,
        low_level_model,
        buffer,
        n_train_epochs,
        batch_size,
        learning_rate,
        epsilon,
        grad_clip_norm,
        logger=None,
    ):
        self.high_level_model = high_level_model
        self.low_level_model = low_level_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger

        self.low_level_params = list(low_level_model.parameters())
        self.high_level_params = list(high_level_model.parameters())
        self.low_level_optim = torch.optim.Adam(self.low_level_params, lr=learning_rate, eps=epsilon)
        self.high_level_optim = torch.optim.Adam(self.high_level_params, lr=learning_rate, eps=epsilon)

    def train(self):
        e_losses = []
        r_losses = []
        n_batches = []
        for epoch in range(1, self.n_train_epochs + 1):
            e_losses.append([])
            r_losses.append([])
            n_batches.append(0)

            for (states, actions, rewards, deltas, goals) in self.buffer.get_train_batches(self.batch_size):
                self.low_level_model.train()
                self.high_level_model.train()

                self.low_level_optim.zero_grad()
                self.high_level_optim.zero_grad()

                e_loss = self.low_level_model.loss(states, actions, deltas)
                r_loss = self.high_level_model.loss(states, goals, rewards)
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.low_level_params, self.grad_clip_norm, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.high_level_params, self.grad_clip_norm, norm_type=2)
                self.low_level_optim.step()
                self.high_level_optim.step()

                e_losses[epoch - 1].append(e_loss.item())
                r_losses[epoch - 1].append(r_loss.item())
                n_batches[epoch - 1] += 1

            if self.logger is not None and epoch % 20 == 0:
                avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
                avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
                message = "> Train epoch {} [ensemble {:.2f} | reward {:.2f}]"
                self.logger.log(message.format(epoch, avg_e_loss, avg_r_loss))

        return (
            self._get_avg_loss(e_losses, n_batches, epoch),
            self._get_avg_loss(r_losses, n_batches, epoch),
        )

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch
