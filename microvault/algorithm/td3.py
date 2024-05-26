import copy
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from numpy import inf
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE = 0.2
NOISE_STD = 0.1
NOISE_CLIP = 0.5


class TD3Lightning(LightningModule):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        env,
        state_size: int,
        action_size: int,
        max_action: int,
        min_action: int,
        sync_rate: int = 10,
        lr: float = 1e-2,
        batch_size: int = 100,
        episode_length: int = 50,
        warm_start_steps: int = 200,
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
        super().__init__()
        self.warm_start_steps = warm_start_steps

        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.env = env
        self.nb_optim_iters = (4,)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.agent = Agent(self.env, self.memory, self.max_action, self.min_action)

        self.total_reward = 0
        self.episode_reward = 0
        self.lr = lr
        self.sync_rate = sync_rate
        self.batch_size = batch_size
        self.episode_length = episode_length

        # Set the device globally
        # torch.set_default_device(device)

        self.actor = Actor(state_size, action_size, float(max_action[0])).to(device)
        self.actor_target = Actor(state_size, action_size, float(max_action[0])).to(
            device
        )

        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.populate(self.warm_start_steps)

        self.total_reward = 0
        self.episode_reward = 0

        self.automatic_optimization = False

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        print("populate")
        for i in range(steps):
            self.agent.step(self.actor)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes in a state x through the network and returns the policy and a sampled action
        Args:
            x: environment state
        Returns:
            Tuple of policy and action
        """
        action = self.actor(state)
        action = action * self.max_action
        Q1_expected, Q2_expected = self.critic(state, action)

        return action, Q1_expected, Q2_expected

    def actor_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        state, action, reward, next_state, done = batch

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        print("loss actor")

        return actor_loss

    def critic_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        state, action, reward, next_state, done = batch

        action_ = action.cpu().numpy()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        with torch.no_grad():

            # Generate a random noise
            noise = torch.FloatTensor(action_).data.normal_(0, NOISE).to(device)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            actions_next = (self.actor_target(next_state) + noise).clamp(
                self.min_action[0].astype(float), self.max_action[0].astype(float)
            )

            Q1_targets_next, Q2_targets_next = self.critic_target(
                next_state, actions_next
            )

            Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)

            # Compute Q targets for current states (y_i)
            Q_targets = reward + (GAMMA * Q_targets_next * (1 - done)).detach()

        # Compute critic loss
        Q1_expected, Q2_expected = self.critic(state, action)

        critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(
            Q2_expected, Q_targets
        )

        print("loss critic")

        return critic_loss

    def train_batch(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            pi, action, log_prob, value = self.agent(self.state, self.device)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, bootstrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        _, _, _, value = self.agent(self.state, self.device)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.gamma
                )[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_qvals,
                    self.batch_adv,
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exclude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> OrderedDict:
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            n_iteraion (int): the number of iterations to train network
            gamma (float): discount factor
        """
        device = self.get_device(batch)

        actor_optimizer, critic_optimizer = self.configure_optimizers()

        reward, done = self.agent.step(self.actor, device)
        self.episode_reward += reward

        critic_loss = self.critic_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        ###################
        # Optimize Critic #
        ###################
        print("update critic")
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        if batch_idx % UPDATE_EVERY_STEP == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actor_loss = self.actor_loss(batch)

            ##################
            # Optimize Actor #
            ##################
            print("update actor")
            actor_optimizer.zero_grad()
            self.manual_backward(actor_loss)
            actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            print("update target networks")
            for target_param, local_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    TAU * local_param.data + (1.0 - TAU) * target_param.data
                )
            for target_param, local_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    TAU * local_param.data + (1.0 - TAU) * target_param.data
                )

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
        }
        # print("Reward: ", reward)

        return OrderedDict({"log": log, "progress_bar": log})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """
        Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
        for each data sample.
        """
        for i in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    # def save(self, filename, version) -> None:
    #       """ Save the model """
    #       torch.save(self.critic.state_dict(), filename + "_critic_" + version + ".pth")
    #       torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer_" + version + ".pth")

    #       torch.save(self.actor.state_dict(), filename + "_actor_" + version + ".pth")
    #       torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_" + version + ".pth")

    # def load(self, filename) -> None:
    #       """ Load the model """
    #       self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
    #       self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
    #       self.critic_target = copy.deepcopy(self.critic)

    #       self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
    #       self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
    #       self.actor_target = copy.deepcopy(self.actor)

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.memory, self.episode_length)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


from lightning.pytorch import Trainer, cli_lightning_logo, seed_everything


def main() -> None:
    env = CustomWrapper(gym.make("BipedalWalker-v3"), min_action=-1.0, max_action=1.0)
    model = TD3Lightning(
        env=env,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        max_action=env.action_space.high,
        min_action=env.action_space.low,
        sync_rate=10,
        lr=1e-2,
        episode_length=200,
        batch_size=100,
        warm_start_steps=1000,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        val_check_interval=100,
        max_epochs=1500,
        logger=WandbLogger(log_model="all"),
    )
    trainer.fit(model)


main()
