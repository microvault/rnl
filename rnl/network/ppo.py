"""
Custom PPO (Proximal Policy Optimization) implementation in PyTorch.
This replaces the stable-baselines3 PPO implementation.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class RolloutBuffer:
    """
    Buffer to store trajectories for PPO training.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # Buffers
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_value: float):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get(self, batch_size: Optional[int] = None):
        """
        Get all data from the buffer and optionally shuffle it.
        Returns data in batches if batch_size is specified.
        """
        indices = np.arange(self.buffer_size)
        if batch_size is not None:
            # Shuffle indices
            np.random.shuffle(indices)
            # Generate batches
            for start_idx in range(0, self.buffer_size, batch_size):
                end_idx = min(start_idx + batch_size, self.buffer_size)
                batch_indices = indices[start_idx:end_idx]

                yield {
                    "observations": torch.FloatTensor(
                        self.observations[batch_indices]
                    ).to(self.device),
                    "actions": torch.LongTensor(self.actions[batch_indices]).to(
                        self.device
                    ),
                    "old_values": torch.FloatTensor(self.values[batch_indices]).to(
                        self.device
                    ),
                    "old_log_probs": torch.FloatTensor(
                        self.log_probs[batch_indices]
                    ).to(self.device),
                    "advantages": torch.FloatTensor(self.advantages[batch_indices]).to(
                        self.device
                    ),
                    "returns": torch.FloatTensor(self.returns[batch_indices]).to(
                        self.device
                    ),
                }
        else:
            yield {
                "observations": torch.FloatTensor(self.observations).to(self.device),
                "actions": torch.LongTensor(self.actions).to(self.device),
                "old_values": torch.FloatTensor(self.values).to(self.device),
                "old_log_probs": torch.FloatTensor(self.log_probs).to(self.device),
                "advantages": torch.FloatTensor(self.advantages).to(self.device),
                "returns": torch.FloatTensor(self.returns).to(self.device),
            }

    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: str = "cpu",
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.verbose = verbose
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        # Get observation and action dimensions
        obs_shape = env.observation_space.shape
        self.observation_dim = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        if hasattr(env.action_space, 'n'):
            self.action_dim = env.action_space.n
        else:
            self.action_dim = env.action_space.shape[0]

        # Get number of environments
        if hasattr(env, 'num_envs'):
            self.n_envs = env.num_envs
        else:
            self.n_envs = 1

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps * self.n_envs,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            device=self.device,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )

        self.num_timesteps = 0
        self._last_obs = None

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, None]:
        """
        Predict action given an observation.
        Compatible with stable-baselines3 interface.
        """
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            logits, _ = self.policy(obs_tensor)
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()

            return action.cpu().numpy()[0], None

    def collect_rollouts(self):
        """
        Collect rollouts using the current policy.
        """
        self.policy.eval()
        self.rollout_buffer.reset()

        if self._last_obs is None:
            self._last_obs = self.env.reset()

        for step in range(self.n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(self._last_obs).to(self.device)
                logits, values = self.policy(obs_tensor)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy().flatten()
            log_probs_np = log_probs.cpu().numpy()

            # Step the environment
            new_obs, rewards, dones, truncated, infos = self.env.step(actions_np)

            # Handle both done and truncated
            if isinstance(dones, bool):
                dones = np.array([dones])
            if isinstance(truncated, bool):
                truncated = np.array([truncated])
            
            terminal_flags = np.logical_or(dones, truncated)

            # Store transitions
            for env_idx in range(self.n_envs):
                self.rollout_buffer.add(
                    obs=self._last_obs[env_idx] if self.n_envs > 1 else self._last_obs,
                    action=actions_np[env_idx] if self.n_envs > 1 else actions_np,
                    reward=rewards[env_idx] if self.n_envs > 1 else rewards,
                    done=terminal_flags[env_idx] if self.n_envs > 1 else terminal_flags,
                    value=values_np[env_idx] if self.n_envs > 1 else values_np,
                    log_prob=log_probs_np[env_idx] if self.n_envs > 1 else log_probs_np,
                )

            self._last_obs = new_obs
            self.num_timesteps += self.n_envs

        # Compute returns and advantages
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(self._last_obs).to(self.device)
            _, last_values = self.policy(obs_tensor)
            last_values = last_values.cpu().numpy().flatten()
            
            # Use the first env's last value for single env
            last_value = last_values[0] if self.n_envs > 1 else last_values
            
        self.rollout_buffer.compute_returns_and_advantages(last_value)

    def train(self):
        """
        Update policy using the rollout buffer.
        """
        self.policy.train()

        # Training metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Generate mini-batches
            for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
                # Forward pass
                logits, values = self.policy(rollout_data["observations"])
                values = values.flatten()

                # Get distribution
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(rollout_data["actions"])
                entropy = dist.entropy().mean()

                # Normalize advantages
                advantages = rollout_data["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                ratio = torch.exp(log_probs - rollout_data["old_log_probs"])
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    values_pred = rollout_data["old_values"] + torch.clamp(
                        values - rollout_data["old_values"],
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss_1 = (rollout_data["returns"] - values).pow(2)
                    value_loss_2 = (rollout_data["returns"] - values_pred).pow(2)
                    value_loss = torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = (rollout_data["returns"] - values).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    approx_kl_div = (
                        (rollout_data["old_log_probs"] - log_probs).mean().item()
                    )
                    approx_kl_divs.append(approx_kl_div)
                    clipped = (
                        (ratio - 1.0).abs() > self.clip_range
                    ).float().mean().item()
                    clip_fractions.append(clipped)

            # Check KL divergence for early stopping
            if self.target_kl is not None:
                if np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    if self.verbose > 0:
                        print(
                            f"Early stopping at epoch {epoch} due to reaching max KL divergence: {np.mean(approx_kl_divs):.4f}"
                        )
                    break

        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }

    def learn(self, total_timesteps: int, callback=None):
        """
        Train the agent for a given number of timesteps.
        """
        iteration = 0
        while self.num_timesteps < total_timesteps:
            iteration += 1

            # Collect rollouts
            self.collect_rollouts()

            # Train the policy
            train_metrics = self.train()

            # Call callback if provided
            if callback is not None:
                callback.on_step()

            # Logging
            if self.verbose > 0 and iteration % 10 == 0:
                print(
                    f"Iteration {iteration} | Timesteps: {self.num_timesteps}/{total_timesteps}"
                )
                print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
                print(f"  Entropy: {-train_metrics['entropy_loss']:.4f}")
                print(f"  Approx KL: {train_metrics['approx_kl']:.4f}")
                print(f"  Clip Fraction: {train_metrics['clip_fraction']:.4f}")

    def save(self, path: str):
        """Save the model."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
            },
            path + ".zip",
        )

    @classmethod
    def load(cls, path: str, env=None):
        """Load a saved model."""
        checkpoint = torch.load(path + ".zip")
        # Note: This is a simplified load. In practice, you'd need to
        # reconstruct the policy architecture as well
        # For now, this is a placeholder
        raise NotImplementedError("Load method needs policy architecture reconstruction")

    def get_env(self):
        """Get the environment."""
        return self.env
