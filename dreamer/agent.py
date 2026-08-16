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
from dreamer.utils import probes


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
                           min_std=config['actor_min_std'],
                           max_std=config.get('actor_max_std', 1.0))
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
        # OP-01: per-network optimizers with reference lr/eps.
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(),
                                             lr=config.get('wm_lr', config['lr']),
                                             eps=config.get('wm_eps', 1e-8))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config.get('actor_lr', config['lr']),
                                                eps=config.get('actor_eps', 1e-8))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=config.get('critic_lr', config['lr']),
                                                 eps=config.get('critic_eps', 1e-8))
        # OP-01: per-network grad clips
        self.wm_grad_clip = config.get('wm_grad_clip', config.get('grad_clip', 100.0))
        self.actor_grad_clip = config.get('actor_grad_clip', config.get('grad_clip', 100.0))
        self.critic_grad_clip = config.get('critic_grad_clip', config.get('grad_clip', 100.0))
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
        # advance probe step counter (once per train_step)
        if not hasattr(self, "_probe_step"): self._probe_step = 0
        self._probe_step += 1
        probes.set_step(self._probe_step)
        # sample a batch: batch = replay_buffer.sample(config['batch_size'])
        batch = replay_buffer.sample(self.config['batch_size'])
        # PROBES: batch stats
        probes.probe("batch_obs", batch['obs'])
        probes.probe("batch_reward", batch['reward'])
        probes.probe("batch_action", batch['action'])
        # update world model: OP-01 wm-specific clip
        info, features = self.world_model.update(batch, self.wm_optimizer, self.device, self.wm_grad_clip)
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
        """Back-compat alias: returns just the lambda-return target (B, H-1) after AC-03 slicing.

        The full contract with weights/base is exposed via compute_targets().
        """
        target, _, _, _ = self.compute_targets(rollout)
        return target

    def compute_targets(self, rollout: dict) -> tuple:
        """
        AC-03 / L-03 / CR-06: computes lambda-return targets, discount-cumprod weights,
        and value baseline aligned with the reference implementation.

        Reference slicing:
            target  = lambda_return(reward[1:], value[:-1], discount[1:], bootstrap=value[-1])
            weights = cumprod([1, discount[:-1]], dim=time)         # weight per step
            base    = value[:-1]                                    # baseline for adv

        Our batch-first shapes (B, H) → returns/weights/base shape (B, H-1).

        Values are computed via the SLOW (EMA) critic, matching reference
        (`self.value(imag_feat).mode()` where value = live critic in ref, but reference
        also uses slow_value for critic distillation; we use live critic here for target,
        slow critic for distillation — same as ref).

        Args:
            rollout: dict from world_model.imagine_rollout()
                     'features' (B, H, sd), 'rewards' (B, H), 'continues' (B, H)
        Returns:
            target:  (B, H-1) — lambda returns for imag steps 1..H-1
            weights: (B, H)   — cumulative discount weights (full length; caller slices)
            base:    (B, H-1) — value baseline for advantage (value at states 0..H-2)
            values:  (B, H)   — critic mean over all H imag features (used for logging)
        """
        features = rollout['features']       # (B, H, sd)
        rewards = rollout['rewards']         # (B, H)
        continues = rollout['continues']     # (B, H) in [0, 1]

        # Value estimates over ALL H imagined features. Detach so bootstrap gradient
        # doesn't leak into the critic. Critic itself is optimized separately below.
        with torch.no_grad():
            values = self.critic(features).mean()  # (B, H)

        # Per-step discount = env discount * predicted continue prob.
        discount = self.config['discount'] * continues  # (B, H)

        # Lambda returns over H-1 steps.
        # Our lambda_return signature: (rewards (B,T), values (B,T+1), continues (B,T), lambda_, discount_scalar).
        # We fold cont into `continues` and pass discount=1.0 so per-step pcont = discount here.
        rewards_used = rewards[:, 1:]                            # (B, H-1)
        # values shape needed: (B, H) — v_1..v_{H-1} plus bootstrap v_H (= values[:, -1])
        values_used = values                                     # (B, H): v_0..v_{H-1}; bootstrap taken as last.
        # For lambda_return we need values of length H (rewards_used is H-1, values must be H-1+1).
        # Pass values[:, 1:] (H-1) concat bootstrap values[:, -1:] (1) → (B, H). Actually values shape (B,H) works
        # only if rewards has H-1 elements: recurrence reads values[:, t+1] for t=0..H-2 -> values[:,1..H-1]
        # and bootstrap = values[:, -1] = values[:, H-1]. So values_used = values (B,H) is correct.
        continues_used = continues[:, 1:] * self.config['discount']  # (B, H-1); fold discount in
        # Use lambda_return with discount=1.0 since we've folded discount into continues_used.
        target = lambda_return(rewards_used, values_used, continues_used,
                               lambda_=self.config['lambda_return'], discount=1.0)  # (B, H-1)

        # Discount-cumprod weights: matches ref torch.cumprod(cat([1, discount[:-1]]), 0)
        ones = torch.ones_like(discount[:, :1])                  # (B, 1)
        weights = torch.cumprod(torch.cat([ones, discount[:, :-1]], dim=1), dim=1).detach()  # (B, H)

        # Baseline: value at states 0..H-2
        base = values[:, :-1]                                    # (B, H-1)

        return target, weights, base, values

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
        # AC-01 / AC-02 / AC-03 / CR-06 rewrite.
        features = rollout['features']       # (B, H, sd) — HAS actor grad through dynamics
        actions_imag = rollout['actions']    # (B, H, ad)

        # Compute targets, weights, and baseline (see compute_targets docstring).
        target, weights, base, values_full = self.compute_targets(rollout)
        # target: (B, H-1) — has grad through rewards -> actions -> actor (dynamics path)
        # weights: (B, H) detached
        # base: (B, H-1) detached
        probes.probe("lambda_returns", target)
        probes.probe("critic_values", values_full)

        # --- return-normalized advantage (matches ref RewardEMA) ---
        # Update normalizer on target (as ref does, on the target tensor pre-detach doesn't matter — RN uses percentiles).
        self.return_normalizer.update(target.detach())
        scale = self.return_normalizer.scale
        adv = (target - base) / scale        # base is detached; target keeps grad
        probes.probe("advantages", adv)
        probes.probe("return_scale", float(scale))

        # --- actor loss ---
        imag_gradient = self.config.get('imag_gradient', 'dynamics')
        # Actor entropy: recompute policy over ALL H features (matches ref that uses [:-1] later).
        # actor sees detached features (matches ref `inp = imag_feat.detach()`).
        policy = self.actor(features.detach())
        entropy = policy.entropy()           # (B, H)
        if imag_gradient == 'dynamics':
            # Dynamics gradient: actor loss = -weights[:-1] * adv
            actor_target = adv               # (B, H-1) — grad through rewards
        elif imag_gradient == 'reinforce':
            log_probs = policy.log_prob(actions_imag)  # (B, H)
            actor_target = log_probs[:, :-1] * adv.detach()  # (B, H-1)
        else:
            raise ValueError(f"Unknown imag_gradient={imag_gradient}")

        # AC-02: discount-cumprod weights applied to actor loss. Slice to H-1 to align with target.
        w = weights[:, :-1]                  # (B, H-1)
        eta = self.config['actor_entropy']
        # Entropy sliced to H-1 to match (ref: entropy[:-1]).
        actor_loss = -(w * (actor_target + eta * entropy[:, :-1])).mean()
        probes.probe("entropy", entropy)
        probes.probe("actor_loss", actor_loss)

        # --- critic update ---
        # Critic uses detached features for first H-1 states (ref: value_input[:-1].detach()).
        feats_for_critic = features[:, :-1].detach()          # (B, H-1, sd)
        critic_dist = self.critic(feats_for_critic)           # (B, H-1)
        target_sg = target.detach()                            # (B, H-1)
        critic_loss_per = -critic_dist.log_prob(symlog(target_sg))  # (B, H-1)
        # Slow-target distillation (ref: value.log_prob(slow.mode().detach()))
        with torch.no_grad():
            slow_mode = self.critic.forward_ema(feats_for_critic).mean()  # real-space
        critic_reg_per = -critic_dist.log_prob(symlog(slow_mode))
        critic_loss_per = critic_loss_per + self.config['critic_slowreg'] * critic_reg_per
        # CR-06: weighted by cumulative discount (ref: torch.mean(weights[:-1] * value_loss)).
        critic_loss = (w * critic_loss_per).mean()
        probes.probe("critic_loss", critic_loss)

        # PROBE: canonical actor mean for PointMass
        if self.obs_dim == 1:
            with torch.no_grad():
                canon_xs = torch.tensor([[-0.5], [0.0], [+0.5]], dtype=torch.float32, device=features.device)
                s0 = self.world_model.rssm.initial_state(3)
                pa = torch.zeros(3, self.action_dim, device=features.device)
                emb = self.world_model.encode(canon_xs)
                ns, _ = self.world_model.rssm.observe(emb, pa, s0)
                cfeat = self.world_model.rssm.get_state_feature(ns)
                cmean = self.actor(cfeat).mean.squeeze(-1)
            probes.probe("canon_actor_mean", cmean)

        # --- backward. Critic loss uses detached features/target, so its graph is disjoint from actor's. ---
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()
        probes.probe("actor_grad_norm", actor_gn)

        info = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_return': target.mean().item(),
            'return_scale': float(scale),
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
