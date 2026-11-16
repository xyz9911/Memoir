"""Drama-inspired Mamba-2 dynamics for Memoir, using functional PyTorch state updates.

This is an independent functional implementation of the Mamba-2 equations in
state-spaces/mamba v2.2.4 (Apache-2.0). It is not a GRU or a Transformer, and does
not claim the performance of the fused CUDA implementation. Functional caches
are essential here: Memoir backpropagates through recurrent steps and forks
overshooting/imagination branches, whereas official inference step() mutates
its cache in place. Drama's two-layer, width-512, state-16 backbone is adapted
to continuous VLN latents and language conditioning, not its Atari actor or
replay-sampling algorithm. See DRAMA_MAMBA2.md for provenance and deviations.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


WORLD_MODEL_PREFIXES = ('lang_init.', 'prior.', 'posterior.', 'to_vis_embed.',
                        'to_reward.', 'obs_transform.')


class GatedRMSNorm(nn.Module):
    """Mamba-2's norm_before_gate=False, ngroups=1 configuration."""
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value, gate):
        x = value.float() * F.silu(gate.float())
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(value.dtype)


class FunctionalMamba2(nn.Module):
    """Selective SSM with head-shared A, input-dependent B/C/dt and causal conv.

    Cache shapes are (B, inner+2*state, conv_width) and
    (B, heads, head_dim, state). SSM accumulation uses FP32 under autocast.
    No cache is detached or updated in place, so all history remains learnable.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner, self.headdim = d_model * expand, headdim
        if self.d_inner % headdim or min(d_model, d_state, d_conv, headdim) < 1:
            raise ValueError('Invalid Mamba-2 dimensions')
        self.nheads = self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2*self.d_inner + 2*d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv,
                              groups=self.conv_dim, padding=d_conv-1)
        # HF's no_init_weights suppresses nn.init while constructing from a
        # checkpoint. BERT's missing-key initializer does not handle Conv1d.
        # Initialize with Tensor.uniform_ so this new depthwise convolution is
        # valid in both ordinary and HF factories. Loaded checkpoints still
        # replace these values afterwards. This is PyTorch's default fan-in
        # uniform distribution for a depthwise convolution (fan_in=d_conv).
        with torch.no_grad():
            bound = 1 / math.sqrt(d_conv)
            self.conv1d.weight.uniform_(-bound, bound)
            self.conv1d.bias.uniform_(-bound, bound)
        dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(.001), math.log(.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True

    def step(self, u, cache=None):
        if u.ndim != 2 or u.shape[-1] != self.d_model:
            raise ValueError('Mamba-2 step expects (batch, d_model)')
        batch = u.shape[0]
        projected = self.in_proj(u)
        z, xbc, dt_raw = projected.split((self.d_inner, self.conv_dim, self.nheads), -1)
        if cache is None:
            conv = xbc.new_zeros(batch, self.conv_dim, self.d_conv)
            state = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                                device=u.device, dtype=torch.float32)
        else:
            conv, state = cache
            if conv.shape != (batch, self.conv_dim, self.d_conv) or state.shape != (
                    batch, self.nheads, self.headdim, self.d_state):
                raise ValueError('Mamba-2 cache shape does not match current batch')
        conv = torch.cat((conv[..., 1:], xbc.unsqueeze(-1)), -1)
        xbc = (conv * self.conv1d.weight[:, 0, :]).sum(-1)
        xbc = F.silu(xbc + self.conv1d.bias)
        x, b, c = xbc.split((self.d_inner, self.d_state, self.d_state), -1)
        # Disable autocast for the recurrent state. Stable accumulation matters
        # for long tours and avoids state dtype depending on the caller context.
        context = torch.autocast(device_type=u.device.type, enabled=False) if u.device.type in ('cpu', 'cuda') else nullcontext()
        with context:
            dt = F.softplus(dt_raw.float() + self.dt_bias.float())
            decay = torch.exp(-dt * self.A_log.float().exp())
            xh = x.float().reshape(batch, self.nheads, self.headdim)
            state = state.float() * decay[..., None, None] + (
                dt[..., None, None] * xh[..., None] * b.float()[:, None, None, :])
            y = (state * c.float()[:, None, None, :]).sum(-1)
            y = y + self.D.float()[None, :, None] * xh
        y = self.norm(y.reshape(batch, self.d_inner).to(projected.dtype), z)
        return self.out_proj(y), (conv, state)

    def forward(self, inputs, cache=None):
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError('Mamba-2 forward expects a nonempty (B,T,D) sequence')
        outputs = []
        for token in inputs.unbind(1):
            output, cache = self.step(token, cache)
            outputs.append(output)
        return torch.stack(outputs, 1), cache


class DramaMamba2Prior(nn.Module):
    """Language-conditioned causal Mamba-2 replacing only Memoir's TSSM prior.

    The four-tensor layer tuple preserves the author's overshooting batching
    interface: (conv_state, ssm_state, language_keys, language_values). The last
    two slots are cached projections of the fixed instruction, not temporal KV.
    """
    def __init__(self, stoch_size=96, stoch_discrete=False, hidden_size=768,
                 deter_size=672, d_model=512, n_layers=2, d_state=16,
                 headdim=64, dropout=0.1):
        super().__init__()
        self.hidden_size, self.deter_size = hidden_size, deter_size
        self.d_model, self.n_layers = d_model, n_layers
        latent = stoch_size * (stoch_discrete or 1) + deter_size
        self.input_proj = nn.Linear(latent, d_model)
        self.language_q = nn.Linear(d_model, d_model, bias=False)
        self.language_k = nn.Linear(hidden_size, d_model, bias=False)
        self.language_v = nn.Linear(hidden_size, d_model, bias=False)
        self.language_o = nn.Linear(d_model, d_model, bias=False)
        self.language_norm = nn.LayerNorm(d_model)
        self.pre_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList([FunctionalMamba2(d_model, d_state, headdim=headdim)
                                     for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.to_deter = nn.Linear(d_model, deter_size)
        self.dense = nn.Sequential(nn.Linear(deter_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self._norm = nn.LayerNorm(hidden_size)
        self.mu_std = nn.Linear(hidden_size, stoch_size * (stoch_discrete or 2))

    def forward(self, prev_states, encoder_embeds, encoder_attention_mask, past_key_values=None):
        if prev_states.ndim != 3 or prev_states.shape[1] < 1:
            raise ValueError('Dynamics requires nonempty (B,T,768) states')
        if encoder_embeds.ndim != 3 or encoder_embeds.shape[0] != prev_states.shape[0]:
            raise ValueError('Instruction/state batch mismatch')
        mask = encoder_attention_mask.bool()
        if mask.shape != encoder_embeds.shape[:2] or not bool(mask.any(-1).all()):
            raise ValueError('Instruction mask must have at least one valid token per row')
        cache = past_key_values
        if cache is not None and len(cache) != self.n_layers:
            raise ValueError('Mamba-2 layer cache count mismatch')
        if cache is None:
            keys, values = self.language_k(encoder_embeds), self.language_v(encoder_embeds)
        else:
            keys, values = cache[0][2], cache[0][3]
            if keys.shape != (prev_states.shape[0], encoder_embeds.shape[1], self.d_model):
                raise ValueError('Cached language has incompatible batch/length')
        x = self.input_proj(prev_states)
        scores = torch.matmul(self.language_q(x), keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        context = torch.matmul(scores.float().softmax(-1).to(values.dtype), values)
        x = self.language_norm(x + self.language_o(context))
        new_cache = []
        for i, (norm, layer) in enumerate(zip(self.pre_norms, self.layers)):
            layer_cache = None if cache is None else cache[i][:2]
            y, state = layer(norm(x), layer_cache)
            x = x + self.dropout(y)
            new_cache.append((state[0], state[1], keys, values))
        deter = self.to_deter(self.final_norm(x[:, -1]))
        statistics = self.mu_std(F.gelu(self._norm(self.dense(deter))))
        return statistics, deter, tuple(new_cache)


def freeze_encoders(model):
    """Freeze text, panorama, and their shared token-type embeddings completely."""
    for name in ('embeddings', 'lang_encoder', 'img_embeddings'):
        module = getattr(model, name)
        module.requires_grad_(False)
        module.eval()
    model.config.freeze_encoders = True


def freeze_for_world_model_pretraining(model):
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(WORLD_MODEL_PREFIXES):
            parameter.requires_grad_(True)
    freeze_encoders(model)
    return [name for name, p in model.named_parameters() if p.requires_grad]
