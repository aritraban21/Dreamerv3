"""
world_model.py — World Model: integrates encoder, RSSM, decoder, and predictors.

The world model is trained to predict the future given past observations and actions.
It serves two purposes:
  1. Learning a compact latent representation of the environment (for imagination).
  2. Ensuring the representation is informative (reconstruction + prediction losses).

Training loss (weighted sum):
    L_WM = lambda_pred * L_pred + lambda_dyn * L_dyn + lambda_rep * L_rep

where:
    L_pred = L_recon + L_reward + L_continue   (prediction losses)
    L_dyn  = max(free_bits, KL[sg(posterior) || prior])  (trains prior toward posterior)
    L_rep  = max(free_bits, KL[posterior || sg(prior)])  (trains encoder toward prior)

Critical: sg() means .detach() — stop-gradient.
    - In L_dyn: posterior is detached, prior is not → only the PRIOR is trained.
    - In L_rep: prior is detached, posterior is not → only the POSTERIOR (encoder) is trained.
Getting this backwards causes representation collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.models.encoder import MLPEncoder
from dreamer.models.decoder import MLPDecoder
from dreamer.models.rssm import RSSM
from dreamer.models.predictors import RewardPredictor, ContinuePredictor
from dreamer.utils.distributions import UnimixCategorical
from dreamer.utils.math_utils import symlog


class WorldModel(nn.Module):
    """
    Combines encoder, RSSM, decoder, reward predictor, and continue predictor.
    Owns the full world model loss computation and one gradient update step.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        """
        Args:
            obs_dim:    raw observation dimensionality
            action_dim: action dimensionality
            config:     dict of hyperparameters (from yaml config)
        """
        super().__init__()
        # extract hyperparameters from config
        self.config = config

        # compute state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
        state_dim = self.config['deter_dim'] + self.config['stoch_dim'] * self.config['stoch_classes']
        # build self.encoder:     MLPEncoder(obs_dim, embed_dim, hidden_dim, num_layers)
        self.encoder = MLPEncoder(
            obs_dim=obs_dim,
            embed_dim=self.config['embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers']
            )

        # build self.rssm:        RSSM(embed_dim, action_dim, deter_dim, stoch_dim, stoch_classes, hidden_dim, unimix)
        self.rssm = RSSM(
            embed_dim=self.config['embed_dim'],
            action_dim=action_dim,
            deter_dim=self.config['deter_dim'],
            stoch_dim=self.config['stoch_dim'],
            stoch_classes=self.config['stoch_classes'],
            hidden_dim=self.config['hidden_dim'],
            unimix=self.config['unimix']
        )
        # build self.decoder:     MLPDecoder(state_dim, obs_dim, hidden_dim, num_layers)
        self.decoder = MLPDecoder(
            state_dim=state_dim,
            obs_dim=obs_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers']
        )
        # build self.reward_pred: RewardPredictor(state_dim, hidden_dim, num_layers, num_bins, bin_range)
        self.reward_pred = RewardPredictor(
            state_dim=state_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_bins=self.config['num_bins'],
            bin_range=self.config['bin_range']
        )
        # build self.cont_pred:   ContinuePredictor(state_dim, hidden_dim, num_layers)
        self.cont_pred = ContinuePredictor(
            state_dim=state_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers']
        )
        # store loss weight hyperparameters as attributes: lambda_pred, lambda_dyn, lambda_rep, free_bits
        #   (lambda_dyn/lambda_rep are the DreamerV3 beta_dyn=0.5 / beta_rep=0.1 KL weights)
        self.lambda_pred = self.config['lambda_pred']
        self.lambda_dyn = self.config['lambda_dyn']
        self.lambda_rep = self.config['lambda_rep']
        self.free_bits = self.config['free_bits']

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of observations (or a sequence) through the encoder.

        Args:
            obs: shape (B, obs_dim) or (B, T, obs_dim)
        Returns:
            embed: shape (B, embed_dim) or (B, T, embed_dim)
        """
        # call self.encoder(obs) and return result
        return self.encoder(obs)

    def observe_sequence(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple:
        """
        Full forward pass over a real sequence from the replay buffer.
        Encodes observations and runs the RSSM posterior to get latent states.

        Args:
            obs:      shape (B, T, obs_dim)
            actions:  shape (B, T, action_dim)
            is_first: shape (B, T) — bool
        Returns:
            features:   shape (B, T, state_dim) — concatenated (h_t, z_t.flatten())
            posteriors: dict with 'posterior_logits' (B, T, stoch_dim, stoch_classes)
            priors:     dict with 'prior_logits'     (B, T, stoch_dim, stoch_classes)
        """
        # encode all observations: embeds = self.encode(obs)   shape (B, T, embed_dim)
        embeds = self.encode(obs)
        # get zero initial state: initial_state = self.rssm.initial_state(batch_size=obs.shape[0])
        initial_state = self.rssm.initial_state(batch_size=obs.shape[0])
        # run rssm.observe_sequence(embeds, actions, is_first, initial_state)
        features, posteriors, priors = self.rssm.observe_sequence(embeds, actions, is_first, initial_state)
        # return features, posteriors, priors
        return features, posteriors, priors

    def compute_kl_loss(
        self, posteriors: dict, priors: dict
    ) -> tuple:
        """
        Computes the KL divergence loss with free bits and KL balancing.

        Two terms:
            L_dyn: KL[sg(posterior) || prior]     — posterior detached, trains the PRIOR
            L_rep: KL[posterior || sg(prior)]     — prior detached, trains the POSTERIOR/encoder

        Both terms use free bits: max(free_bits, KL) to prevent trivial zero-KL solutions.
        Weighted sum (DreamerV3 beta weights): lambda_dyn * L_dyn + lambda_rep * L_rep

        Args:
            posteriors: dict with 'posterior_logits' (B, T, stoch_dim, stoch_classes)
            priors:     dict with 'prior_logits'     (B, T, stoch_dim, stoch_classes)
        Returns:
            kl_loss: scalar
            kl_info: dict with 'kl_dyn' and 'kl_rep' for logging
        """
        # extract posterior_logits and prior_logits
        posterior_logits = posteriors['posterior_logits']
        prior_logits = priors['prior_logits']
        # create posterior dist: UnimixCategorical(posterior_logits, unimix)
        posterior = UnimixCategorical(posterior_logits, self.config['unimix'])
        # create prior dist:     UnimixCategorical(prior_logits, unimix)
        prior = UnimixCategorical(prior_logits, self.config['unimix'])
        # --- dynamics loss: sg(posterior) || prior ---
        # create sg_posterior dist: UnimixCategorical(posterior_logits.detach(), unimix)
        sg_posterior = UnimixCategorical(posterior_logits.detach(), self.config['unimix'])
        # compute KL: kl_dyn_raw = sg_posterior.kl_divergence(prior)  shape (B, T, stoch_dim)
        kl_dyn_raw = sg_posterior.kl_divergence(prior)
        # sum over stoch_dim: kl_dyn_raw = kl_dyn_raw.sum(-1)  shape (B, T)
        kl_dyn_raw = kl_dyn_raw.sum(-1)
        # apply free bits: kl_dyn = torch.clamp(kl_dyn_raw, min=self.free_bits).mean()
        kl_dyn = torch.clamp(kl_dyn_raw, min=self.free_bits).mean()
        # --- representation loss: posterior || sg(prior) ---
        # create sg_prior dist: UnimixCategorical(prior_logits.detach(), unimix)
        sg_prior = UnimixCategorical(prior_logits.detach(), self.config['unimix'])
        # compute KL: kl_rep_raw = posterior.kl_divergence(sg_prior)  shape (B, T, stoch_dim)
        kl_rep_raw = posterior.kl_divergence(sg_prior)
        # sum over stoch_dim and apply free bits: kl_rep = torch.clamp(...).mean()
        kl_rep = kl_rep_raw.sum(-1)
        kl_rep = torch.clamp(kl_rep, min=self.free_bits).mean()
        # --- combine ---
        # kl_loss = self.lambda_dyn * kl_dyn + self.lambda_rep * kl_rep
        kl_loss = self.lambda_dyn * kl_dyn + self.lambda_rep * kl_rep
        # return kl_loss, {'kl_dyn': kl_dyn.item(), 'kl_rep': kl_rep.item()}
        return kl_loss, {'kl_dyn': kl_dyn.item(), 'kl_rep': kl_rep.item()}

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple:
        """
        Computes the full world model loss for one batch.

        Args:
            obs:       shape (B, T, obs_dim)
            actions:   shape (B, T, action_dim)
            rewards:   shape (B, T) — raw rewards from environment
            continues: shape (B, T) — 1.0 for non-terminal steps, 0.0 at terminal
            is_first:  shape (B, T)
        Returns:
            loss: scalar
            info: dict of sub-losses for logging
        """
        # run observe_sequence to get features, posteriors, priors
        features, posteriors, priors = self.observe_sequence(obs, actions, is_first)
        # --- reconstruction loss ---
        # obs_pred = self.decoder(features)   shape (B, T, obs_dim) in symlog space
        obs_pred = self.decoder(features)
        # recon_loss = F.mse_loss(obs_pred, symlog(obs))   (targets are symlog-transformed)
        recon_loss = F.mse_loss(obs_pred, symlog(obs))
        # --- reward prediction loss ---
        # reward_dist = self.reward_pred(features)
        reward_dist = self.reward_pred(features)
        # reward_loss = -reward_dist.log_prob(symlog(rewards)).mean()
        reward_loss = -reward_dist.log_prob(symlog(rewards)).mean()
        # --- continue prediction loss ---
        # cont_dist = self.cont_pred(features)
        cont_dist = self.cont_pred(features)
        # cont_loss = -cont_dist.log_prob(continues).mean()
        cont_loss = -cont_dist.log_prob(continues).mean()
        # --- prediction loss total ---
        # pred_loss = recon_loss + reward_loss + cont_loss
        pred_loss = recon_loss + reward_loss + cont_loss
        # --- KL loss ---
        # kl_loss, kl_info = self.compute_kl_loss(posteriors, priors)
        kl_loss, kl_info = self.compute_kl_loss(posteriors, priors)
        # --- total ---
        # total_loss = self.lambda_pred * pred_loss + kl_loss
        total_loss = self.lambda_pred * pred_loss + kl_loss
        # compile info dict for logging (all .item() scalars)
        # return total_loss, info
        info = {
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'cont_loss': cont_loss.item(),
            **kl_info
        }
        return total_loss, info

    def update(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float = 100.0,
    ) -> tuple:
        """
        One world model gradient update step.

        Args:
            batch:     dict from replay_buffer.sample()
            optimizer: world model optimizer (Adam in Part 1)
            device:    torch device
            grad_clip: global gradient norm clip (Part 1 only; use AGC in Part 2)
        Returns:
            info:     dict of losses for logging
            features: shape (B, T, state_dim) — latent states for actor-critic seeding
        """
        # move all batch tensors to device
        for k, v in batch.items():
            batch[k] = v.to(device)
        # extract obs, actions, rewards, continues (1 - done), is_first from batch
        obs = batch['obs']
        actions = batch['action']
        rewards = batch['reward']
        continues = 1.0 - batch['done'].float()
        is_first = batch['is_first']
        # call loss, info = self.compute_loss(obs, actions, rewards, continues, is_first)
        loss, info = self.compute_loss(obs, actions, rewards, continues, is_first)
        # optimizer.zero_grad()
        optimizer.zero_grad()
        # loss.backward()
        loss.backward()
        # clip gradients by global norm: nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        # optimizer.step()
        optimizer.step()
        # re-run observe_sequence (no_grad) to get features for actor-critic seeding
        with torch.no_grad():
            features, _, _ = self.observe_sequence(obs, actions, is_first)
        # return info, features
        return info, features

    @torch.no_grad()
    def imagine_rollout(
        self,
        start_features: torch.Tensor,
        actor,
        horizon: int = 16,
    ) -> dict:
        """
        Generates imagined trajectories starting from real latent states.
        Used in agent.update_actor_critic().

        Args:
            start_features: shape (B, state_dim) — real starting states (from observe_sequence)
            actor:          Actor module
            horizon:        H, imagination steps (16 in paper)
        Returns:
            dict with:
                'features':  shape (B, H, state_dim)
                'actions':   shape (B, H, action_dim)
                'rewards':   shape (B, H) — imagined rewards from reward_pred
                'continues': shape (B, H) — imagined continuation probs from cont_pred
        Notes:
            This method uses no_grad for the predictors and state decoding.
            The RSSM imagine_sequence is called separately with grad enabled in update_actor_critic.
            Call rssm.imagine_sequence with torch.enable_grad() for proper actor gradient flow.
        """
        # convert start_features to initial RSSM state (h, z) by splitting along last dim:
        #   h = start_features[:, :deter_dim]
        h = start_features[:, :self.config['deter_dim']]
        #   z_flat = start_features[:, deter_dim:]
        z_flat = start_features[:, self.config['deter_dim']:]
        #   z = z_flat.reshape(B, stoch_dim, stoch_classes)
        z = z_flat.reshape(z_flat.shape[0], self.config['stoch_dim'], self.config['stoch_classes'])
        # initial_state = {'h': h, 'z': z}
        initial_state = {'h': h, 'z': z}
        # use torch.enable_grad() context and call self.rssm.imagine_sequence(initial_state, actor, horizon)
        with torch.enable_grad():
            imagination = self.rssm.imagine_sequence(initial_state, actor, horizon)
    
        # extract features and actions from the returned dict
        imagined_features = imagination['features']
        imagined_actions = imagination['actions']
        # compute imagined rewards: rewards = self.reward_pred(features).mean()  shape (B, H)
        imagined_rewards = self.reward_pred(imagined_features).mean()
        # compute imagined continues: continues = self.cont_pred(features).probs  shape (B, H)
        imagined_continues = self.cont_pred(imagined_features).probs
        # return {'features': features, 'actions': actions, 'rewards': rewards, 'continues': continues}
        return {
            'features': imagined_features,
            'actions': imagined_actions,
            'rewards': imagined_rewards,
            'continues': imagined_continues
        }