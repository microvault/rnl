import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Configurar device
# Check that MPS is available
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     else:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# else:
device = torch.device("mps")

print(f"Usando device: {device}")

# Definir seed para reproducibilidade
seed = 1
torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

# Hyperparâmetros ajustados
learning_rate = 0.0003
gamma = 0.99
GAE_LAMBDA = 0.95  # Lambda do GAE
CLIP_COEF = 0.2  # Coeficiente de clipping
ENT_COEF = 0.05  # Coeficiente de entropia
VF_COEF = 0.5  # Coeficiente do valor
MAX_GRAD_NORM = 0.5  # Clipping dos gradientes
UPDATE_EPOCHS = 4  # Número de épocas de atualização
T_horizon = 10
ACTION_STD_INIT = 0.6  # (não utilizado para ações discretas)


class PPO(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        input_dim: dimensão da entrada
        hidden_dims: lista com os tamanhos dos layers ocultos, ex: [256, 128]
        output_dim: dimensão da saída (número de ações)
        """
        super(PPO, self).__init__()
        self.data = []

        # Se input_dim for uma tupla, converte para int
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]

        # Cria uma "trunk" dinâmica com os layers ocultos
        self.trunk = self.build_mlp(input_dim, hidden_dims)
        trunk_out_dim = hidden_dims[-1]

        # Cabeça de política e valor
        self.fc_pi = nn.Linear(trunk_out_dim, output_dim)
        self.fc_v = nn.Linear(trunk_out_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def build_mlp(self, input_dim, hidden_dims, activation=nn.ReLU):
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        return nn.Sequential(*layers)

    def pi(self, x, softmax_dim=0):
        x = self.trunk(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.trunk(x)
        value = self.fc_v(x)
        return value

    def put_data(self, transition):
        # Cada transição: (estado, ação, recompensa, próximo estado, probabilidade, done)
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_lst.append(0 if done else 1)

        s = torch.from_numpy(np.array(s_lst)).float().to(device)
        a = torch.from_numpy(np.array(a_lst)).long().to(device).unsqueeze(1)
        r = torch.from_numpy(np.array(r_lst)).float().to(device).unsqueeze(1)
        s_prime = torch.from_numpy(np.array(s_prime_lst)).float().to(device)
        done_mask = torch.from_numpy(np.array(done_lst)).float().to(device).unsqueeze(1)
        prob_a = torch.from_numpy(np.array(prob_a_lst)).float().to(device).unsqueeze(1)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for _ in range(UPDATE_EPOCHS):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            # Cálculo da vantagem usando GAE
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * GAE_LAMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # relação pi/old_pi

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = VF_COEF * F.smooth_l1_loss(self.v(s), td_target.detach())

            dist = Categorical(pi)
            entropy = dist.entropy().mean()
            entropy_loss = -ENT_COEF * entropy

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
