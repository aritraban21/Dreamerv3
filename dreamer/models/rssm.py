"""
rssm.py — Recurrent State-Space Model (RSSM). The architectural core of DreamerV3.

The RSSM maintains a compact latent representation of the environment.
It is the most complex file in the codebase — implement carefully and test thoroughly.

State representation:
    h_t: deterministic recurrent state — computed by the GRU from (h_{t-1}, z_{t-1}, a_{t-1})
    z_t: stochastic categorical state  — 32 independent categoricals with 32 classes each
    s_t = cat(h_t, z_t.flatten())      — the combined feature vector used downstream

Two distributions over z_t:
    Posterior q(z_t | h_t, x_t): uses actual observation x_t via the encoder embedding.
                                  Used during world model training on real data.
    Prior     p(z_t | h_t):      uses only h_t (no observation).
                                  Used for imagination rollouts (no env needed).

Common bugs to watch for:
    - Forgetting stop-gradient (detach) in KL loss (handled in world_model.py)
    - Forgetting straight_through_sample (kills all gradients to actor silently)
    - Not resetting state at episode boundaries (is_first flag in observe_sequence)
    - Detaching intermediate tensors in imagine_sequence (breaks actor gradients)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.utils.distributions import UnimixCategorical


class BlockGRU(nn.Module):
    """
    GRU cell for updating the deterministic state h_t.

    RS-05: LayerNorm-inside GRUCell matching reference (networks.py:GRUCell). This is the
    DreamerV2/V3 stability trick — LayerNorm over the gate pre-activations before splitting
    into reset/candidate/update, plus a learned update-gate bias of -1.

    Input: projected pre-GRU features (via img_in_layers) shape (B, hidden_dim)
    Output: updated h_t                                   shape (B, deter_dim)
    """

    def __init__(self, input_dim: int, deter_dim: int, update_bias: float = -1.0):
        super().__init__()
        self._input_dim = input_dim
        self._deter_dim = deter_dim
        self._update_bias = update_bias
        # ref: Linear(input+state, 3*deter, bias=False) → LayerNorm(3*deter)
        self.linear = nn.Linear(input_dim + deter_dim, 3 * deter_dim, bias=False)
        self.norm = nn.LayerNorm(3 * deter_dim, eps=1e-3)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        reset, cand, update = torch.split(parts, [self._deter_dim] * 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    """
    Full Recurrent State-Space Model with posterior and prior networks.

    See module docstring for architecture overview.
    Implement and test observe() and imagine() individually before observe_sequence()
    and imagine_sequence() — the sequence methods just loop over the step methods.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int = 4096,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 1024,
        unimix: float = 0.01,
    ):
        """
        Args:
            embed_dim:     encoder output size (fed to posterior network)
            action_dim:    environment action dimensionality
            deter_dim:     GRU hidden state size (4096 in paper, 512 in small config)
            stoch_dim:     number of independent categorical latent variables (32 in paper)
            stoch_classes: number of classes per categorical (32 in paper)
            hidden_dim:    MLP hidden layer width for prior/posterior networks
            unimix:        uniform mixture fraction for UnimixCategorical (0.01 in paper)
        """
        super().__init__()
        # store all hyperparameters as attributes (needed in other methods)
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.hidden_dim = hidden_dim
        self.unimix = unimix
        # compute stoch_flat = stoch_dim * stoch_classes (size of flattened z_t)
        self.stoch_flat = self.stoch_dim * self.stoch_classes
        # RS-01: pre-GRU projection matching reference (`_img_in_layers`).
        # cat(z_flat, action) -> Linear(inp, hidden, bias=False) -> LayerNorm -> SiLU -> GRUCell(hidden -> deter)
        self.img_in_layers = nn.Sequential(
            nn.Linear(self.stoch_flat + self.action_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim, eps=1e-3),
            nn.SiLU(),
        )
        # build gru: input is projected hidden_dim now, output is deter_dim
        self.gru = BlockGRU(input_dim=self.hidden_dim, deter_dim=self.deter_dim)
        # build prior_net: MLP that maps deter_dim → stoch_flat
        #   structure: Linear(deter_dim, hidden_dim) → RMSNorm → SiLU → Linear(hidden_dim, stoch_flat)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim, bias=True),
            nn.LayerNorm(self.hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_flat)
        )
        # build posterior_net: MLP that maps (deter_dim + embed_dim) → stoch_flat
        #   structure: Linear(deter_dim + embed_dim, hidden_dim) → RMSNorm → SiLU → Linear(hidden_dim, stoch_flat)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim, bias=True),
            nn.LayerNorm(self.hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_flat)
        )
        

    def initial_state(self, batch_size: int) -> dict:
        """
        Returns a zero-initialized RSSM state for the beginning of a sequence or episode.

        Returns:
            dict with:
                'h': shape (B, deter_dim)                   — zeros
                'z': shape (B, stoch_dim, stoch_classes)    — zeros
        Notes:
            Use torch.zeros and ensure tensors are on the same device as model parameters.
            A convenient way: next(self.parameters()).device
        """
        # get the device from model parameters
        device = next(self.parameters()).device
        # create h = torch.zeros(batch_size, self.deter_dim, device=device)
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        # create z = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        # return {'h': h, 'z': z}
        return {'h': h, 'z': z}

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        prev_state: dict,
    ) -> tuple:
        """
        One posterior step. Uses encoder embedding (from real observation) to compute z_t.
        This is used during world model training where we have access to real observations.

        Steps:
            1. Flatten previous z: z_flat = prev_state['z'].flatten(start_dim=-2)
            2. GRU input: x = cat(z_flat, action, dim=-1)
            3. h = gru(x, prev_state['h'])
            4. Posterior logits: post_logits = posterior_net(cat(h, embed, dim=-1))
               reshape to (B, stoch_dim, stoch_classes)
            5. z = UnimixCategorical(post_logits, unimix).straight_through_sample()
            6. Prior logits: prior_logits = prior_net(h)
               reshape to (B, stoch_dim, stoch_classes)

        Args:
            embed:      shape (B, embed_dim) — encoder output for current observation
            action:     shape (B, action_dim) — action taken at the PREVIOUS step (a_{t-1})
            prev_state: dict with 'h' (B, deter_dim) and 'z' (B, stoch_dim, stoch_classes)
        Returns:
            state: dict with 'h' (B, deter_dim) and 'z' (B, stoch_dim, stoch_classes)
            dists: dict with:
                'posterior_logits': (B, stoch_dim, stoch_classes) — for KL loss
                'prior_logits':     (B, stoch_dim, stoch_classes) — for KL loss
        """
        # flatten z from prev_state: z_flat = prev_state['z'].flatten(start_dim=-2)
        z_flat = prev_state['z'].flatten(start_dim=-2)
        # concatenate z_flat and action as GRU input
        x_t = torch.cat([z_flat, action], dim=-1)
        # RS-01: project through img_in_layers before GRU
        x_t = self.img_in_layers(x_t)
        # compute h via self.gru(gru_input, prev_state['h'])
        h_t = self.gru(x_t, prev_state['h'])
        # compute posterior logits via self.posterior_net(cat(h, embed))
        post_logits = self.posterior_net(torch.cat([h_t, embed], dim=-1))
        # reshape posterior logits to (B, stoch_dim, stoch_classes)
        post_logits = post_logits.view(-1, self.stoch_dim, self.stoch_classes)
        # sample z using UnimixCategorical(posterior_logits, self.unimix).straight_through_sample()
        z_t = UnimixCategorical(post_logits, self.unimix).straight_through_sample()
        # compute prior logits via self.prior_net(h)
        prior_logits = self.prior_net(h_t)
        # reshape prior logits to (B, stoch_dim, stoch_classes)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.stoch_classes)
        # return {'h': h, 'z': z}, {'posterior_logits': ..., 'prior_logits': ...}
        return {'h': h_t, 'z': z_t}, {'posterior_logits': post_logits, 'prior_logits': prior_logits}

    def imagine(
        self,
        action: torch.Tensor,
        prev_state: dict,
    ) -> tuple:
        """
        One prior step. No observation is used — only previous state and action.
        This is used during imagination rollouts (no environment interaction).

        Steps:
            1-3. Same GRU step as observe()
            4. Prior logits: prior_logits = prior_net(h); reshape to (B, stoch_dim, stoch_classes)
            5. z = UnimixCategorical(prior_logits, unimix).straight_through_sample()

        Args:
            action:     shape (B, action_dim) — current action a_t
            prev_state: dict with 'h' and 'z'
        Returns:
            state: dict with 'h' and 'z'
            dists: dict with 'prior_logits' only
        Notes:
            Do NOT detach any tensors here — gradients must flow through states to the actor.
        """
        # flatten z from prev_state
        z_flat = prev_state['z'].flatten(start_dim=-2)
        # concatenate z_flat and action as GRU input
        x_t = torch.cat([z_flat, action], dim=-1)
        # RS-01: project through img_in_layers before GRU
        x_t = self.img_in_layers(x_t)
        # compute h via self.gru
        h_t = self.gru(x_t, prev_state['h'])
        # compute prior logits via self.prior_net(h)
        prior_logits = self.prior_net(h_t)
        # reshape prior logits to (B, stoch_dim, stoch_classes)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.stoch_classes)
        # sample z using straight_through_sample
        z_t = UnimixCategorical(prior_logits, self.unimix).straight_through_sample()
        # return {'h': h, 'z': z}, {'prior_logits': prior_logits}
        return {'h': h_t, 'z': z_t}, {'prior_logits': prior_logits}

    def observe_sequence(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        initial_state: dict,
    ) -> tuple:
        """
        Runs the RSSM posterior over a full sequence of real observations.
        Called once per world model training step.

        Critical detail: when is_first[b, t] is True for batch element b at step t,
        the RSSM state for that element must be reset to initial_state BEFORE the
        observe() call at step t. This ensures episodes don't bleed into each other.

        Args:
            embeds:        shape (B, T, embed_dim) — encoder outputs for the full sequence
            actions:       shape (B, T, action_dim) — actions a_t (action at t is used for step t+1)
            is_first:      shape (B, T) — bool, True at the first step of each episode
            initial_state: zero state dict from initial_state(B)
        Returns:
            features:   shape (B, T, deter_dim + stoch_flat) — cat(h_t, z_t.flatten()) at each t
            posteriors: dict with 'posterior_logits' shape (B, T, stoch_dim, stoch_classes)
            priors:     dict with 'prior_logits'     shape (B, T, stoch_dim, stoch_classes)
        Notes:
            - Loop over the T dimension, calling self.observe() at each step.
            - At each step t, before calling observe, apply is_first masking:
                state['h'] = torch.where(is_first[:, t:t+1], initial_state['h'], state['h'])
                state['z'] = torch.where(is_first[:, t:t+1, None], initial_state['z'], state['z'])
              This resets state for episodes starting at step t without breaking batch parallelism.
            - The action at step t is a_{t-1} (what was done before observing x_t), so use actions[:, t].
        """
        is_first = is_first.bool()  # ensure is_first is a boolean tensor
        # get batch size B and sequence length T from embeds
        B = embeds.shape[0]
        T = embeds.shape[1]
        # initialize current state as a copy of initial_state (both h and z)
        initial_state_copy = initial_state.copy()
        # create empty lists for features, posterior_logits, prior_logits
        features_list = []
        posterior_logits_list = []
        prior_logits_list = []
        # get init_h and init_z from initial_state for the is_first masking
        h_0 = initial_state_copy['h']
        z_0 = initial_state_copy['z']
        state = initial_state_copy.copy()
        # loop over t in range(T):
        for t in range(T):
            # create is_first_t mask: shape (B, 1) from is_first[:, t]
            is_first_t = is_first[:,t].unsqueeze(-1)  # shape (B, 1)
            #     reset h: state['h'] = torch.where(is_first_t, init_h, state['h'])
            h_t = torch.where(is_first_t, h_0, state['h'])
            #     reset z: state['z'] = torch.where(is_first_t.unsqueeze(-1), init_z, state['z'])
            z_t = torch.where(is_first_t.unsqueeze(-1), z_0, state['z'])
            #     RS-02: zero prev_action on is_first (matches ref: prev_action *= (1 - is_first))
            action_t = actions[:, t] * (~is_first_t).float()
            #     call state, dists = self.observe(embeds[:, t], action_t, state)
            state = {'h': h_t, 'z': z_t}
            state, logits = self.observe(embeds[:, t], action_t, state)
            #     append get_state_feature(state) to features list
            features_list.append(self.get_state_feature(state))
            #     append dists['posterior_logits'] to posterior_logits list
            posterior_logits_list.append(logits['posterior_logits'])
            #     append dists['prior_logits'] to prior_logits list
            prior_logits_list.append(logits['prior_logits'])
        # stack all lists along dim=1 to get (B, T, ...) tensors
        features = torch.stack(features_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)
        prior_logits = torch.stack(prior_logits_list, dim=1)
        # return features tensor, {'posterior_logits': stacked}, {'prior_logits': stacked}
        return features, {'posterior_logits': posterior_logits}, {'prior_logits': prior_logits}

    def imagine_sequence(
        self,
        initial_state: dict,
        actor,
        horizon: int = 16,
    ) -> dict:
        """
        Imagines a trajectory of length H by rolling out the prior RSSM.
        No real observations are used. The actor selects actions at each step.

        Used in agent.py → update_actor_critic() for generating training data.

        Args:
            initial_state: dict with 'h' (B, deter_dim) and 'z' (B, stoch_dim, stoch_classes)
                           B = batch_size * batch_length (one starting state per real timestep)
            actor:         Actor module — called with state_feature to get action distribution
            horizon:       H, number of imagination steps (16 in paper)
        Returns:
            dict with:
                'features': shape (B, H, state_dim) — state features at each imagined step
                'actions':  shape (B, H, action_dim) — actions taken at each step
        Notes:
            - Do NOT call .detach() on any tensor here. Gradients must flow backward
              through states and actions to reach the actor parameters.
            - Call actor.get_action(feature, training=True) to sample exploratory actions.
        """
        # initialize current state from initial_state
        state = initial_state.copy()
        # create empty lists for features and actions
        features_list = []
        actions_list = []
        # loop over t in range(horizon):
        for _ in range(horizon):
            # compute feature = self.get_state_feature(state)
            feature = self.get_state_feature(state)
            # AC-01: actor sees detached feat (matches ref `inp = feat.detach()`), so its own
            # gradient at step t doesn't flow back through prior actor parameters via the recurrence.
            # Gradient still reaches earlier actor params via action -> dynamics -> next feat -> reward.
            action = actor.get_action(feature.detach(), training=True)
            #     call state, _ = self.imagine(action, state)
            state, _ = self.imagine(action, state)
            # keep un-detached feature so reward_pred/cont_pred gradient flows to prior actor calls
            features_list.append(feature)
            actions_list.append(action)
        # stack features along dim=1: torch.stack(features, dim=1)
        stacked_features = torch.stack(features_list, dim=1)
        # stack actions along dim=1: torch.stack(actions, dim=1)
        stacked_actions = torch.stack(actions_list, dim=1)
        # return {'features': stacked_features, 'actions': stacked_actions}
        return {'features': stacked_features, 'actions': stacked_actions}

    def get_state_feature(self, state: dict) -> torch.Tensor:
        """
        Concatenates h and z into a single flat feature vector.
        This is the input to all downstream networks (decoder, reward pred, actor, critic).

        Args:
            state: dict with:
                'h': shape (B, deter_dim)
                'z': shape (B, stoch_dim, stoch_classes)
        Returns:
            shape (B, deter_dim + stoch_dim * stoch_classes) — concatenated feature
        """
        # flatten z: z_flat = state['z'].flatten(start_dim=-2)  → (B, stoch_flat)
        z_flat = state['z'].flatten (start_dim=-2)
        # concatenate h and z_flat along last dim: torch.cat([state['h'], z_flat], dim=-1)
        feature = torch.cat([state['h'], z_flat], dim=-1)
        # return the concatenated feature
        return feature
