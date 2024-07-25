import torch
import torch.nn as nn


class Inverse(nn.Module):
    """
    1. (primeiro submodelo) codifica o estado e o próximo estado no espaço de recursos.
    2. (segundo submodelo) o inverso aproxima a ação realizada pelo estado determinado e pelo próximo estado no tamanho do recurso

    retorna a ação prevista e o estado codificado para o modelo direto e o próximo estado codificado para treinar o modelo direto!

    otimizando o modelo Inverso pela perda entre a ação real tomada pela política atual e a ação prevista pelo modelo inverso
    """

    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.state_dim = len(state_size)
        self.state_size = state_size

        self.encoder = nn.Sequential(nn.Linear(state_size, 128), nn.ELU())
        self.layer1 = nn.Linear(2 * 128, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)

    def calc_input_layer(self):
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return x.flatten().shape[0]

    def forward(self, enc_state, enc_state1):
        """
        Input: state s and state s' as torch Tensors with shape: (batch_size, state_size)
        Output: action probs with shape (batch_size, action_size)
        """
        x = torch.cat((enc_state, enc_state1), dim=1)
        x = torch.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x


class Forward(nn.Module):
    def __init__(
        self, state_size, action_size, output_size, hidden_size=256, device="cuda:0"
    ):
        super().__init__()
        self.action_size = action_size
        self.device = device
        self.forwardM = nn.Sequential(
            nn.Linear(output_size + self.action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state, action):
        """
        Input: state s embeddings and action a as torch Tensors with shape
        s: (batch_size, embedding_size),
        a: (batch_size, action_size)

        Output:
        encoded state s' prediction by the forward model with shape: (batch_size, embedding_size)

        Gets as inputs the action taken from the policy and the encoded state by the encoder in the inverse model.
        The forward model tries to predict the encoded next state.
        Returns the predicted encoded next state.
        Gets optimized by the MSE between the actual encoded next state and the predicted version of the forward model!

        """
        # One-hot-encoding for the actions
        ohe_action = torch.zeros(action.shape[0], self.action_size).to(self.device)
        indices = torch.stack(
            (torch.arange(action.shape[0]).to(self.device), action.squeeze().long()),
            dim=0,
        )
        indices = indices.tolist()
        ohe_action[indices] = 1.0
        # concat state embedding and encoded action

        x = torch.cat((state, ohe_action), dim=1)
        assert x.device.type == "cuda"
        return self.forwardM(x)
