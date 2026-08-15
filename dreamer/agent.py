"""
agent.py — DreamerV3 agent: coordinates all three networks and training phases.

Three training phases happen each iteration (train_step):
  1. World model update:    learn to predict observations, rewards, and continuation
  2. Actor-critic update:   optimize policy and value function on imagined trajectories
  3. EMA target update:     soft-update the critic's EMA target network

Environment interaction (step) is separate and happens in the training loop in train.py.

Key implementation notes:
  - The actor is updated via REINFORCE (not reparameterization) — log_prob is needed.
  - Lambda-returns use the CURRENT critic (critic()) for bootstrap values, detached.
    The EMA copy (critic.forward_ema()) is used ONLY as a regularization target in the
    critic loss — matches DreamerV3 paper (p.6) and the official code (`slowtar: False`).
  - Returns are normalized by ReturnNormalizer before computing the actor loss.
  - features must be .detach()'d from the world model before computing actor/critic losses
    to prevent gradients from flowing back into the world model from the AC update.

⚠️ ACTOR BASELINE (DreamerV3 Eq. 6): the paper's actor objective uses the ADVANTAGE,
   sg((R_lambda - v(s)) / max(1, S)) * log pi + eta * H — i.e. it subtracts the value
   baseline v(s) before normalizing. The guidance below uses raw normalized returns
   (R_lambda / max(1, S)) with NO baseline. That is still an unbiased REINFORCE estimator
   but has higher variance. Subtracting v(s) (a critic value, detached) matches the paper.
   See update_actor_critic().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.world_model import WorldModel
from dreamer.models.actor import Actor
from dreamer.models.critic import Critic
from dreamer.utils.normalizer import ReturnNormalizer
from dreamer.utils.math_utils import lambda_return, symlog
from dreamer.replay_buffer import ReplayBuffer


class DreamerV3(nn.Module):
    """
    Top-level DreamerV3 agent.

    Owns the three networks (world model, actor, critic), their optimizers,
    and the return normalizer. Coordinates all training updates.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        """
        Args:
            obs_dim:    raw observation dimensionality
            action_dim: action dimensionality
            config:     dict of hyperparameters loaded from yaml
        """
        super().__init__()
        # store config, obs_dim, action_dim as attributes
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # compute state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
        self.state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
        # build self.world_model: WorldModel(obs_dim, action_dim, config)
        self.world_model = WorldModel(obs_dim, action_dim, config)
        # build self.actor:  Actor(state_dim, action_dim, hidden_dim, num_layers, min_std)
        self.actor = Actor(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           hidden_dim=config['hidden_dim'],
                           num_layers=config['num_layers'],
                           min_std=config['actor_min_std'])
        # build self.critic: Critic(state_dim, hidden_dim, num_layers, num_bins, bin_range, ema_decay)
        self.critic = Critic(state_dim=self.state_dim,
                             hidden_dim=config['hidden_dim'],
                             num_layers=config['num_layers'],
                             num_bins=config['num_bins'],
                             bin_range=config['bin_range'],
                             ema_decay=config['critic_ema_decay'])
        # build self.return_normalizer: ReturnNormalizer(ret_norm_decay, lower_pct, upper_pct, min_scale)
        self.return_normalizer = ReturnNormalizer(decay=config['ret_norm_decay'],
                                                  lower_pct=config['ret_norm_lower_pct'],
                                                  upper_pct=config['ret_norm_upper_pct'],
                                                  min_scale=config['ret_norm_min'])
        # build self.wm_optimizer:     torch.optim.Adam(world_model.parameters(), lr=config['lr'])
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=config['lr'])
        # build self.actor_optimizer:  torch.optim.Adam(actor.parameters(), lr=config['lr'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr'])
        # build self.critic_optimizer: torch.optim.Adam(critic.parameters(), lr=config['lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])
        # store device from config
        self.device = torch.device(config['device'])

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        state: dict | None,
        training: bool = True,
    ) -> tuple:
        """
        Takes one environment step: encodes obs, updates RSSM state, samples action.
        Called in the training loop for every environment interaction.

        Agent state contract (SUPERSET of RSSM state):
            The dict returned by step() has keys {'h', 'z', 'action'}.
              - 'h', 'z':  RSSM latent state (h_t, z_t)
              - 'action':  the action the actor just sampled — carried to the next call
                           to be fed as prev_action into rssm.observe (matches official
                           DreamerV3, which stores `prevact` in the policy carry).
            RSSM itself does NOT store the action — it's a step-input to observe(),
            never part of RSSM's internal state. The agent owns the action-tracking
            because step() interleaves posterior updates with action sampling across
            gym steps, whereas training uses observe_sequence which reads actions
            straight from the replay buffer.

        Caller contract:
            Pass state=None on env.reset(); otherwise pass back the dict returned by
            the previous step() call.

        Args:
            obs:      shape (obs_dim,) — single raw observation from gym
            state:    previous agent state dict {'h', 'z', 'action'}, or None at episode start
            training: if True, sample action; if False, return mean (for evaluation)
        Returns:
            action:    shape (action_dim,) — numpy array for env.step()
            new_state: updated agent state dict {'h', 'z', 'action'} — 'action' is
                       the action just sampled, used as prev_action on the next call
        """
        # convert obs to tensor: obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        #
        # unpack prev state:
        if state is None:
            # episode start — no previous action exists
            rssm_state = self.world_model.rssm.initial_state(batch_size=1)
            prev_action = torch.zeros(1, self.action_dim, device=self.device)
        else:
            rssm_state = {'h': state['h'], 'z': state['z']}
            prev_action = state['action']

        # prev_action is the action the actor sampled on the PREVIOUS call — zeros only at episode start.
        #
        # encode observation:  embed = self.world_model.encode(obs_t)   shape (1, embed_dim)
        embed = self.world_model.encode(obs_t)
        # posterior step:      new_rssm_state, _ = self.world_model.rssm.observe(embed, prev_action, rssm_state)
        new_rssm_state, _ = self.world_model.rssm.observe(embed, prev_action, rssm_state)
        # state feature:       feature = self.world_model.rssm.get_state_feature(new_rssm_state)  shape (1, state_dim)
        feature = self.world_model.rssm.get_state_feature(new_rssm_state)
        # sample action:       action_t = self.actor.get_action(feature, training=training)       shape (1, action_dim)
        action_t = self.actor.get_action(feature, training=training)
        #
        # pack next agent state (carries new action forward for next call):
        #   new_state = {'h': new_rssm_state['h'],
        #                'z': new_rssm_state['z'],
        #                'action': action_t}
        #
        new_state = {'h': new_rssm_state['h'],
            'z': new_rssm_state['z'],
            'action': action_t}
        # convert to numpy: action = action_t.squeeze(0).cpu().numpy()
        action = action_t.squeeze(0).cpu().numpy()
        # return action, new_state
        return action, new_state

    def train_step(self, replay_buffer: ReplayBuffer) -> dict:
        """
        Performs one complete DreamerV3 training iteration.

        Steps:
            1. Sample a batch from the replay buffer
            2. Update the world model (one gradient step)
            3. Generate imagined rollouts from the learned world model
            4. Update actor and critic using imagined rollouts
            5. Update EMA target critic
            6. Return all metrics for logging

        Args:
            replay_buffer: the ReplayBuffer containing real experience
        Returns:
            dict of scalar metrics: wm losses, actor loss, critic loss, return stats
        """
        # sample a batch: batch = replay_buffer.sample(config['batch_size'])
        batch = replay_buffer.sample(self.config['batch_size'])
        # update world model: wm_info, features = self.world_model.update(batch, wm_optimizer, device, grad_clip)
        info, features = self.world_model.update(batch, self.wm_optimizer, self.device, self.config['grad_clip'])
        # generate imagined rollout from random subset of features:
        #   start_features = features.detach().reshape(-1, state_dim)  (flatten B*T)
        start_features = features.detach().reshape(-1, self.state_dim)
        #   optionally subsample if B*T is too large (take first batch_size rows)
        if start_features.shape[0] > self.config['batch_size']:
            start_features = start_features[:self.config['batch_size']]
        #   rollout = self.world_model.imagine_rollout(start_features, self.actor, horizon)
        rollout = self.world_model.imagine_rollout(start_features, self.actor, self.config['imagination_horizon'])
        # update actor and critic: ac_info = self.update_actor_critic(rollout)
        ac_info = self.update_actor_critic(rollout)
        # update EMA: self.critic.update_ema()
        self.critic.update_ema()
        # merge info dicts and return
        wm_info_prefixed = {f'wm/{k}': v for k, v in info.items()}
        wm_loss = info['recon_loss'] + info['reward_loss'] + info['cont_loss'] + info['kl_dyn'] + info['kl_rep']
        return {'wm_loss': wm_loss, **wm_info_prefixed, **ac_info}

    def compute_lambda_returns(self, rollout: dict) -> torch.Tensor:
        """
        Computes bootstrapped TD(lambda) returns from an imagined rollout.
        Uses the CURRENT critic for bootstrap values (detached, no gradient).

        Matches DreamerV3 paper (p.6) and the official code default (`slowtar: False`
        in imag_loss / repl_loss): "compute returns using the current critic network".
        The EMA copy is used only as a regularization target inside update_actor_critic(),
        never here.

        Args:
            rollout: dict with 'features' (B, H, state_dim), 'rewards' (B, H),
                     'continues' (B, H) — from world_model.imagine_rollout()
        Returns:
            returns: shape (B, H) — lambda-return targets R_t^lambda

        Notes on shapes:
            The lambda_return function needs values of shape (B, H+1):
                v_0 .. v_{H-1}  from critic(features[:, 0..H-1]).mean()
                v_H             from critic(features[:, H-1]).mean() as bootstrap
                    (since we have H imagined steps, we use the last state's value as v_H)

        IMPORTANT: wrap the critic call in torch.no_grad() OR call .detach() on the
        values tensor. The bootstrap targets must not carry a gradient back into
        the critic — otherwise the critic learns to make its own targets smaller
        (a degenerate self-referential loss).
        """
        # extract features, rewards, continues from rollout
        features = rollout['features']  # shape (B, H, state_dim)
        rewards = rollout['rewards']    # shape (B, H)
        continues = rollout['continues']  # shape (B, H)
        
        # get value estimates at each step (no gradient — these are targets):
        #   with torch.no_grad(): values_dist = self.critic(features)
        with torch.no_grad():
            values_dist = self.critic(features)  # shape (B, H)
        # compute value mean: values = values_dist.mean()  shape (B, H)
        values = values_dist.mean()  # shape (B, H) (mean of distribution so does not retunrn sclalar)
        # compute bootstrap value: v_H = values[:, -1:]  shape (B, 1)
        v_H = values[:, -1:]  # shape (B, 1) vakue at horizon step (last imagined state)
        # concatenate to get all H+1 values: values_all = torch.cat([values, v_H], dim=1)  shape (B, H+1)
        values_all = torch.cat([values, v_H], dim=1)  # shape (B, H+1)
        # call lambda_return(rewards, values_all, continues, lambda_, discount)
        returns = lambda_return(rewards, values_all, continues,
                                 lambda_=self.config['lambda_return'], discount=self.config['discount'])
        # return returns of shape (B, H)
        return returns
    
    def update_actor_critic(self, rollout: dict) -> dict:
        """
        Updates actor and critic from an imagined rollout.

        Critic update:
            - Target: symlog-transformed lambda-returns (two-hot encoded internally by log_prob)
            - Loss: -critic(feat).log_prob(symlog(returns)).mean()
            - Regularization: distillation via -critic(feat).log_prob(symlog(critic_ema(feat).mean().detach())).mean()
              (matches official DreamerV3 `slowreg * value.loss(sg(slowvalue.pred()))`)

        Actor update (REINFORCE with value baseline — matches DreamerV3 Eq. 6):
            - Compute advantage: adv = returns - v(s), where v(s) is a DETACHED critic baseline
            - Update return normalizer on RAW returns (not advantage — matches paper)
            - Divide advantage by return-percentile scale S: adv_norm = adv / max(1, S)
            - Loss: -mean( sg(adv_norm) * log_pi(a|s) + eta * entropy )
            - Advantage must be DETACHED before multiplying with log_pi

        Args:
            rollout: dict from world_model.imagine_rollout()
        Returns:
            dict with 'actor_loss', 'critic_loss', 'mean_return', 'return_scale'
        """
        # compute lambda returns: returns = self.compute_lambda_returns(rollout)   shape (B, H)
        returns = self.compute_lambda_returns(rollout)  # shape (B, H)
        # --- critic update ---
        # detach features for critic: feats = rollout['features'].detach()
        feats = rollout['features'].detach()  # shape (B, H, state_dim)
        # compute critic distribution: critic_dist = self.critic(feats)
        critic_dist = self.critic(feats)  # shape (B, H, num_bins)
        # critic loss: -critic_dist.log_prob(symlog(returns.detach())).mean()
        critic_loss = -critic_dist.log_prob(symlog(returns.detach())).mean()
        # add EMA regularization: distillation — critic.forward(feats).log_prob(symlog(critic.forward_ema(feats).mean().detach()))
        # matches official DreamerV3 `slowreg * value.loss(sg(slowvalue.pred()))`
        with torch.no_grad():
            slow_target = self.critic.forward_ema(feats).mean()   # real-space scalar
        critic_reg = -critic_dist.log_prob(symlog(slow_target)).mean()
        critic_loss = critic_loss + self.config['critic_slowreg'] * critic_reg


        # zero_grad → critic_loss.backward() → clip gradients → critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['grad_clip'])
        self.critic_optimizer.step()

        # --- actor update ---
        with torch.no_grad():
            values_baseline = self.critic(feats).mean()  # shape (B, H) — detached baseline for advantage
        # update return normalizer: self.return_normalizer.update(returns)

        advantages = returns - values_baseline  # compute advantage (detached)
        self.return_normalizer.update(returns)
        scale = self.return_normalizer.scale # get current return scale for logging
        # normalize advantage by return-percentile scale (paper: sg(adv / max(1, S)))
        advantages_norm = advantages / scale
        # re-compute action log_probs from the rollout (features already detached from WM):
        #   action_dist = self.actor(feats)
        action_dist = self.actor(feats)  # shape (B, H, action_dim)
        #   log_probs = action_dist.log_prob(rollout['actions'])     shape (B, H)
        log_probs = action_dist.log_prob(rollout['actions'])  # shape (B, H)
        #   entropy   = action_dist.entropy()                        shape (B, H)
        entropy = action_dist.entropy()  # shape (B, H)

        actor_loss = -(log_probs * advantages_norm.detach() + self.config['actor_entropy'] * entropy).mean()
        # zero_grad → actor_loss.backward() → clip gradients → actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['grad_clip'])
        self.actor_optimizer.step()
        # compile and return info dict
        info = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_return': returns.mean().item(),
            'return_scale': scale
        }
        return info

    def save(self, path: str) -> None:
        """
        Saves agent state to disk (all networks and optimizers).

        Args:
            path: file path for the checkpoint (e.g., 'checkpoints/dreamer.pt')
        """
        # create parent directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save a dict with:
        #   'world_model': self.world_model.state_dict()
        #   'actor':       self.actor.state_dict()
        #   'critic':      self.critic.state_dict()
        #   'wm_opt':      self.wm_optimizer.state_dict()
        #   'actor_opt':   self.actor_optimizer.state_dict()
        #   'critic_opt':  self.critic_optimizer.state_dict()
        checkpoint_dict = {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'wm_opt': self.wm_optimizer.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }
        # use torch.save(checkpoint_dict, path)
        torch.save(checkpoint_dict, path)

    def load(self, path: str) -> None:
        """
        Loads agent state from a checkpoint file.

        Args:
            path: file path to the checkpoint
        """
        # load checkpoint: ckpt = torch.load(path, map_location=self.device)
        # load state dicts: world_model, actor, critic, and all optimizers
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt['world_model'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.wm_optimizer.load_state_dict(ckpt['wm_opt'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
