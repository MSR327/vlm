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


class SquashedNormal:
    """
    Squashed Gaussian distribution (SAC/PPO Tanh-Normal) with exact change-of-variables log_prob.
    Eliminates off-policy clamping discrepancies and boundary probability spikes.
    """
    def __init__(self, loc, scale):
        self.normal = Normal(loc, scale)

    def sample(self):
        u = self.normal.sample()
        action = torch.tanh(u)
        return action, u

    def log_prob(self, action, u):
        log_prob_u = self.normal.log_prob(u)
        log_det = torch.log(1.0 - action.pow(2) + 1e-6)
        return (log_prob_u - log_det).sum(dim=-1)

    def entropy(self):
        return self.normal.entropy().sum(dim=-1)


class RolloutBuffer:
    """Stores experience transitions for episodic PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.u_latents = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.truncateds = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.u_latents.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.truncateds.clear()
        self.values.clear()

    def __len__(self):
        return len(self.actions)


class ActorCritic(nn.Module):
    """
    Continuous Actor-Critic Network with Squashed Gaussian Policy.
    Shared hidden layer architecture with LayerNorm stabilization.
    """
    def __init__(self, obs_dim=C.OBS_DIM, action_dim=C.ACTION_DIM, std_init=C.ACTION_STD_INIT):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Policy Head (Actor): outputs latent mean u in R^2 (squashed strictly to [-1, 1] via tanh)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

        # Trainable exploration log standard deviation (clamped to [-5.0, 0.5])
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
        clamped_log_std = torch.clamp(self.log_std, -5.0, 0.5)
        std = clamped_log_std.exp().expand_as(mean)
        dist = SquashedNormal(mean, std)
        return dist, value

    def get_value(self, state):
        return self.critic(state)


class PPOAgent:
    """
    PPO Agent with Generalized Advantage Estimation (GAE), Squashed Gaussian policy,
    and clipped surrogate objective.
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
        Takes numpy observation (obs_dim,), returns:
            action_np: bounded control in (-1, 1) strictly via tanh
            u_np: unconstrained latent sample
            log_prob: float, exact change-of-variables log-probability
            value: float, state-value prediction
        """
        state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.ac(state_t)
            if deterministic:
                mean = self.ac.actor(state_t)
                action = torch.tanh(mean)
                u = mean
                log_prob = dist.log_prob(action, u)
            else:
                action, u = dist.sample()
                log_prob = dist.log_prob(action, u)

        return (
            action.squeeze(0).cpu().numpy(),
            u.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item()
        )

    def remember(self, state, action, u, log_prob, reward, done, value, truncated=False):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.u_latents.append(u)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.truncateds.append(truncated)
        self.buffer.values.append(value)

    def compute_gae(self, last_value=0.0):
        """
        Computes Generalized Advantage Estimation (GAE) over the stored buffer.
        Differentiates between terminal failure (mask=0) and time-limit truncation (mask=1, bootstraps value).
        """
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        truncateds = self.buffer.truncateds
        values = self.buffer.values + [last_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            # If terminated (crash, lane departure, stall): mask = 0 (no future return)
            # If truncated (max steps timeout) or running: mask = 1 (bootstrap V(s_{t+1}))
            is_truncated = truncateds[t] if t < len(truncateds) else False
            is_done = dones[t]
            mask = 0.0 if (is_done and not is_truncated) else 1.0

            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer.values)]
        return advantages, returns

    def learn(self, last_value=0.0):
        """
        Executes PPO policy and value updates over the buffer with PPO-2 value clipping.
        """
        if len(self.buffer) == 0:
            return 0.0, 0.0

        advantages, returns = self.compute_gae(last_value)

        # Convert buffer to tensors
        states_t = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        u_latents_t = torch.tensor(np.array(self.buffer.u_latents), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(np.array(self.buffer.values), dtype=torch.float32, device=self.device)
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
                b_u = u_latents_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_old_values = old_values_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_adv = advantages_t[batch_idx]

                dist, val = self.ac(b_states)
                log_probs = dist.log_prob(b_actions, b_u)
                entropy = dist.entropy().mean()

                # PPO Clipped Surrogate Loss
                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - C.ENTROPY_COEF * entropy

                # PPO-2 Clipped Value Loss
                val_pred = val.squeeze(-1)
                val_clipped = b_old_values + torch.clamp(val_pred - b_old_values, -self.clip_eps, self.clip_eps)
                vf_loss1 = F.mse_loss(val_pred, b_returns)
                vf_loss2 = F.mse_loss(val_clipped, b_returns)
                critic_loss = C.VALUE_COEF * torch.max(vf_loss1, vf_loss2)

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
