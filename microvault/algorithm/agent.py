import copy
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from components.replaybuffer import PER
from network.model import Actor, Critic
from numpy import inf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = 2**20  # replay buffer size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
UPDATE_EVERY_STEP = 2  # how often to update the target and actor networks
BATCH_SIZE = 128  # minibatch size


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size: int = 24,
        action_size: int = 2,
        max_action: int = 1,
        min_action: int = 1,
        noise: float = 0.2,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
        pretraining: bool = False,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            max_action (ndarray): the maximum valid value for each action vector
            min_action (ndarray): the minimum valid value for each action vector
            noise (float): the range to generate random noise while learning
            noise_std (float): the range to generate random noise while performing action
            noise_clip (float): to clip random noise into this range
        """
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        # parameter noise
        self.distances = []
        self.desired_distance = 0.7
        self.scalar = 0.05
        self.scalar_decay = 0.99

        # normal noise
        self.normal_scalar = 0.25

        # step counter
        self.t = 0

        if pretraining:

            # Transfer Learning

            # Actor Network (w/ Target Network)
            self.actor = Actor(state_size, action_size, float(self.max_action[0])).to(
                device
            )
            self.actor.load_state_dict(torch.load("/content/checkpoint_actor.pth"))
            self.actor_target = (
                Actor(state_size, action_size, float(self.max_action[0]))
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
            self.actor_optimizer.load_state_dict(
                torch.load("/content/checkpoint_actor_optimizer.pth")
            )

            self.actor_noised = Actor(
                state_size, action_size, float(self.max_action[0])
            ).to(device)

            self.critic = Critic(state_size, action_size).to(device)
            self.critic.load_state_dict(torch.load("/content/checkpoint_critic.pth"))
            self.critic_target = (
                Critic(state_size, action_size).to(device).eval().requires_grad_(False)
            )
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
            self.critic_optimizer.load_state_dict(
                torch.load("/content/checkpoint_critic_optimizer.pth")
            )

        else:
            # Actor Network (w/ Target Network)
            self.actor = Actor(state_size, action_size, float(max_action[0])).to(device)
            self.actor_target = Actor(state_size, action_size, float(max_action[0])).to(
                device
            )
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

            self.actor_noised = Actor(state_size, action_size, float(max_action[0])).to(
                device
            )

            # Critic Network (w/ Target Network)
            self.critic = Critic(state_size, action_size).to(device)
            self.critic_target = Critic(state_size, action_size).to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # self.scaler = torch.cuda.amp.GradScaler()
        # self.clip_grad = torch.nn.utils.clip_grad_norm_

        # Inicializar schedulers
        # self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10, gamma=0.9)
        # self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10, gamma=0.9)

        # compile model
        # self.actor = torch.compile(self.actor)
        # self.critic = torch.compile(self.critic)

        # Replay memory
        self.memory = PER(BUFFER_SIZE, BATCH_SIZE)

        self.seed_everything()

        # Statistics
        # self.q1_record = collections.deque(maxlen=1000)
        # self.q2_record = collections.deque(maxlen=1000)
        # self.q_func1_loss_record = collections.deque(maxlen=100)
        # self.q_func2_loss_record = collections.deque(maxlen=100)
        # self.policy_loss_record = collections.deque(maxlen=100)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if isinstance(state, tuple):
            state = np.array(state[0])
        """Save experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done, reward)
        # self.memory.add(state, action, reward, next_state, done)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Returns actions for given state as per current policy."""

        assert isinstance(
            states, np.ndarray
        ), "States is not of data structure (np.ndarray) in PREDICT -> states: {}.".format(
            type(states)
        )
        assert isinstance(
            states[0], np.float32
        ), "States is not of type (np.float32) in PREDICT -> states type: {}.".format(
            type(states)
        )
        assert (
            states.shape[0] == 24
        ), "The size of the states is not (24) in PREDICT -> states size: {}.".format(
            states.shape[0]
        )
        assert (
            states.ndim == 1
        ), "The ndim of the states is not (1) in PREDICT -> states ndim: {}.".format(
            states.ndim
        )

        if isinstance(states, tuple):
            states = np.array(states[0])

        state = torch.from_numpy(states).float().to(device)

        self.actor.eval()
        self.actor_noised.eval()

        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()  # EXP: use detach ?

            # ---------------------------- adaptative noise weight ---------------------------- #
            self.actor_noised.load_state_dict(self.actor.state_dict().copy())
            # add noise to the copy
            self.actor_noised.add_parameter_noise(self.scalar)
            # get the next action values from the noised actor
            action_noised = self.actor_noised(state).cpu().data.numpy()
            # measure the distance between the action values from the regular and
            # the noised actor to adjust the amount of noise that will be added next round
            distance = np.sqrt(np.mean(np.square(action - action_noised)))
            # for stats and print only
            self.distances.append(distance)
            # adjust the amount of noise given to the actor_noised
            if distance > self.desired_distance:
                self.scalar *= self.scalar_decay
            if distance < self.desired_distance:
                self.scalar /= self.scalar_decay
            # set the noised action as action
            action = action_noised

        self.actor.train()

        action = action.clip(self.min_action[0], self.max_action[0])

        assert (
            action.shape[0] == self.action_size
        ), "The action size is different from the defined size in PREDICT."
        # assert isinstance(action[0], np.float32), "Action is not of type (np.float32) in PREDICT -> action type: {}.".format(type(action))

        return action

    def learn(
        self, n_iteraion: int, gamma=GAMMA
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            n_iteraion (int): the number of iterations to train network
            gamma (float): discount factor
        """
        if len(self.memory) > BATCH_SIZE:
            average_Q = 0
            max_Q = -inf
            average_critic_loss = 0
            average_actor_loss = 0

            for i in range(n_iteraion):
                idxs, (state, action, reward, next_state, done), is_weights = (
                    self.memory.sample()
                )
                is_weights = torch.from_numpy(is_weights).float().to(device)

                action_ = action.cpu().numpy()

                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models

                with torch.no_grad():

                    # Generate a random noise
                    noise = (
                        torch.FloatTensor(action_)
                        .data.normal_(0, self.noise)
                        .to(device)
                    )
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                    actions_next = (self.actor_target(next_state) + noise).clamp(
                        self.min_action[0].astype(float),
                        self.max_action[0].astype(float),
                    )

                    Q1_targets_next, Q2_targets_next = self.critic_target(
                        next_state, actions_next
                    )

                    Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)

                    average_Q += torch.mean(Q_targets_next)
                    max_Q = max(max_Q, torch.max(Q_targets_next))

                    # Compute Q targets for current states (y_i)
                    Q_targets = reward + (gamma * Q_targets_next * (1 - done)).detach()

                # Compute critic loss
                Q1_expected, Q2_expected = self.critic(state, action)
                Q_expected = torch.min(Q1_expected, Q2_expected)
                errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
                critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(
                    Q2_expected, Q_targets
                )

                self.memory.batch_update(idxs, errors)

                # Minimize the loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if i % UPDATE_EVERY_STEP == 0:
                    # ---------------------------- update actor ---------------------------- #
                    # Compute actor loss
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                    # Minimize the loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.critic, self.critic_target, TAU)
                    self.soft_update(self.actor, self.actor_target, TAU)

                average_critic_loss += critic_loss
                average_actor_loss += actor_loss

            loss_critic = average_critic_loss / n_iteraion
            loss_actor = average_actor_loss / n_iteraion
            average_policy = average_Q / n_iteraion
            max_policy = max_Q

            return (loss_critic, loss_actor, average_policy, max_policy)

        else:
            return (0, 0, 0, 0)

    @staticmethod
    def soft_update(local_model, target_model, tau) -> None:
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filename, version) -> None:
        """Save the model"""
        torch.save(self.critic.state_dict(), filename + "_critic_" + version + ".pth")
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "_critic_optimizer_" + version + ".pth",
        )

        torch.save(self.actor.state_dict(), filename + "_actor_" + version + ".pth")
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "_actor_optimizer_" + version + ".pth",
        )

    def load(self, filename) -> None:
        """Load the model"""
        self.critic.load_state_dict(
            torch.load(filename + "_critic.pth")
        )  # del torch.load
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pth")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pth")
        )
        self.actor_target = copy.deepcopy(self.actor)
