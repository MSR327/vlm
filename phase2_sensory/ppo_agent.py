"""
PyTorch Proximal Policy Optimization (PPO) Agent for Phase 2 TransFuser Driving.

Operates over the fixed 144-dimensional observation space:
    s_t = [ z_fusion (128-d) || x_ego (8-d) || x_nav (8-d) ]
Emitting continuous steering and longitudinal acceleration commands.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import phase2_sensory.config as C


class RolloutBuffer:
    """Stores experience transitions for episodic PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.actions)


class ActorCritic(nn.Module):
    """
    Continuous Actor-Critic Network.
    Shared hidden layer architecture with LayerNorm stabilization.
    """
    def __init__(self, obs_dim=C.OBS_DIM, action_dim=C.ACTION_DIM, std_init=C.ACTION_STD_INIT):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Policy Head (Actor)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bounds mean action strictly to [-1.0, +1.0]
        )

        # Trainable exploration log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(std_init))

        # Value Head (Critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        mean = self.actor(state)
        value = self.critic(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    def get_value(self, state):
        return self.critic(state)


class PPOAgent:
    """
    PPO Agent with Generalized Advantage Estimation (GAE) and clipped surrogate objective.
    """
    def __init__(self,
                 obs_dim=C.OBS_DIM,
                 action_dim=C.ACTION_DIM,
                 lr=C.LEARNING_RATE,
                 gamma=C.GAMMA,
                 lam=C.LAMBDA,
                 clip_eps=C.POLICY_CLIP,
                 n_epochs=C.N_EPOCHS,
                 batch_size=C.BATCH_SIZE,
                 device=None):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer = RolloutBuffer()

    def select_action(self, state_np, deterministic=False):
        """
        Takes numpy observation (obs_dim,), returns (action_np, log_prob, value).
        """
        state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.ac(state_t)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
                # Clip to valid vehicle control limits
                action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def remember(self, state, action, log_prob, reward, done, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)

    def compute_gae(self, last_value=0.0):
        """
        Computes Generalized Advantage Estimation (GAE) over the stored buffer.
        """
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [last_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer.values)]
        return advantages, returns

    def learn(self, last_value=0.0):
        """
        Executes PPO policy and value updates over the buffer.
        """
        if len(self.buffer) == 0:
            return 0.0, 0.0

        advantages, returns = self.compute_gae(last_value)

        # Convert buffer to tensors
        states_t = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(np.array(returns), dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(np.array(advantages), dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_samples = len(self.buffer)
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_adv = advantages_t[batch_idx]

                dist, val = self.ac(b_states)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # PPO Clipped Surrogate Loss
                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - C.ENTROPY_COEF * entropy

                # Value Loss (MSE)
                critic_loss = C.VALUE_COEF * F.mse_loss(val.squeeze(-1), b_returns)

                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        self.buffer.clear()
        mean_a_loss = total_actor_loss / max(1, n_updates)
        mean_c_loss = total_critic_loss / max(1, n_updates)
        print(f"[PPOAgent] Weights Updated | Actor Loss: {mean_a_loss:.4f} | Critic Loss: {mean_c_loss:.4f}")
        return mean_a_loss, mean_c_loss

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"[PPOAgent] Checkpoint saved: {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[PPOAgent] Model loaded successfully: {filepath}")
