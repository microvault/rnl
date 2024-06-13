import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class ICM(nn.Module):
    def __init__(
        self,
        inverse_model,
        forward_model,
        learning_rate=1e-3,
        lambda_=0.1,
        beta=0.2,
        device="cuda:0",
    ):
        super().__init__()
        self.inverse_model = inverse_model.to(device)
        self.forward_model = forward_model.to(device)

        self.forward_scale = 1.0
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = beta
        self.lambda_ = lambda_
        self.forward_loss = nn.MSELoss(reduction="none")
        self.inverse_loss = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.Adam(
            list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=1e-3,
        )

    def calc_errors(self, state1, state2, action):
        """
        Input: Torch Tensors state s, state s', action a with shapes
        s: (batch_size, state_size)
        s': (batch_size, state_size)
        a: (batch_size, 1)

        """
        assert (
            state1.device.type == "cuda"
            and state2.device.type == "cuda"
            and action.device.type == "cuda"
        )
        enc_state1 = self.inverse_model.encoder(state1).view(state1.shape[0], -1)
        enc_state2 = self.inverse_model.encoder(state2).view(state1.shape[0], -1)

        # assert enc_state1.shape == (32,1152), "Shape is {}".format(enc_state1.shape)
        # calc forward error
        forward_pred = self.forward_model(enc_state1.detach(), action)
        forward_pred_err = (
            1
            / 2
            * self.forward_loss(forward_pred, enc_state2.detach())
            .sum(dim=1)
            .unsqueeze(dim=1)
        )

        # calc prediction error
        pred_action = self.inverse_model(enc_state1, enc_state2)
        inverse_pred_err = self.inverse_loss(
            pred_action, action.flatten().long()
        ).unsqueeze(dim=1)

        return forward_pred_err, inverse_pred_err

    def update_ICM(self, forward_err, inverse_err):
        self.optimizer.zero_grad()
        loss = ((1.0 - self.beta) * inverse_err + self.beta * forward_err).mean()
        loss.backward(retain_graph=True)
        clip_grad_norm_(self.inverse_model.parameters(), 1)
        clip_grad_norm_(self.forward_model.parameters(), 1)
        self.optimizer.step()
        return loss.detach().cpu().numpy()
