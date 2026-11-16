import copy
import ast
import types
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnbert.drama_mamba2 import FunctionalMamba2, DramaMamba2Prior, freeze_for_world_model_pretraining


def parallel_ssd_reference(model, inputs):
    """Independent closed-form semiseparable scan, with parallel causal conv."""
    z, xbc, dt = model.in_proj(inputs).split((model.d_inner, model.conv_dim, model.nheads), -1)
    xbc = F.silu(F.conv1d(xbc.transpose(1, 2), model.conv1d.weight,
        model.conv1d.bias, padding=model.d_conv-1, groups=model.conv_dim)[..., :inputs.shape[1]].transpose(1, 2))
    x, b, c = xbc.split((model.d_inner, model.d_state, model.d_state), -1)
    batch, length, _ = inputs.shape
    x = x.reshape(batch, length, model.nheads, model.headdim)
    dt = F.softplus(dt + model.dt_bias)
    prefix = (-model.A_log.exp() * dt).cumsum(1).transpose(1, 2)
    log_decay = prefix[..., :, None] - prefix[..., None, :]
    decay = log_decay.exp() * torch.ones(length, length, device=x.device).tril()
    bc = torch.einsum('btn,bsn->bts', c, b)
    y = torch.einsum('bhts,bts,bsh,bshp->bthp', decay, bc, dt, x)
    y = y + model.D[None, None, :, None] * x
    y = y.reshape(batch, length, model.d_inner)
    gated = y * F.silu(z)
    normed = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True)+model.norm.eps) * model.norm.weight
    return model.out_proj(normed)


class Mamba2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def block(self):
        return FunctionalMamba2(d_model=16, d_state=4, headdim=8)

    def prior(self):
        return DramaMamba2Prior(stoch_size=4, hidden_size=24, deter_size=20,
            d_model=16, n_layers=2, d_state=4, headdim=8, dropout=0).eval()

    def test_convolution_initializes_when_hf_skips_nn_init(self):
        # Poison allocations instead of relying on allocator-dependent zeros
        # or garbage. HF no_init_weights similarly suppresses nn.init helpers;
        # BERT never reinitializes a missing Conv1d checkpoint key afterwards.
        def skipped_init(tensor, *args, **kwargs):
            with torch.no_grad():
                return tensor.fill_(float('nan'))
        with patch.object(nn.init, 'kaiming_uniform_', skipped_init), patch.object(nn.init, 'uniform_', skipped_init):
            model = self.block()
        for parameter in (model.conv1d.weight, model.conv1d.bias):
            self.assertTrue(bool(torch.isfinite(parameter).all()))
            self.assertLessEqual(float(parameter.abs().max()), model.d_conv ** -.5)
            self.assertGreater(float(parameter.std()), 0.)
        # Explicit initialization must not override a subsequently loaded state.
        original = self.block()
        model.load_state_dict(original.state_dict())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_parallel_closed_form_values_and_gradients(self):
        a, x = self.block(), torch.randn(2, 9, 16)
        b = copy.deepcopy(a)
        output, _ = a(x)
        reference = parallel_ssd_reference(b, x)
        torch.testing.assert_close(output, reference, atol=4e-7, rtol=2e-5)
        output.square().sum().backward(); reference.square().sum().backward()
        for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=4e-4, msg=name)

    def test_official_mamba2_v224_step_parity(self):
        # Extract the pinned upstream method without importing optional CUDA/Triton.
        # It is executed verbatim, with its torch fallbacks (no fused extensions).
        from einops import rearrange, repeat
        source = Path(__file__).parent / 'vendor/mamba2_v2_2_4.py.txt'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mamba2')
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        module = ast.Module(body=[step], type_ignores=[])
        context = dict(torch=torch, F=F, rearrange=rearrange, repeat=repeat,
                       causal_conv1d_update=None, selective_state_update=None)
        exec(compile(module, str(source), 'exec'), context)
        model = self.block()
        official = copy.deepcopy(model)
        official.d_ssm, official.ngroups, official.rmsnorm = official.d_inner, 1, True
        official.act = nn.SiLU()
        official_step = types.MethodType(context['step'], official)
        cache = None
        conv = torch.zeros(2, model.conv_dim, model.d_conv)
        state = torch.zeros(2, model.nheads, model.headdim, model.d_state)
        with torch.no_grad():
            for _ in range(11):
                x = torch.randn(2, 16)
                ours, cache = model.step(x, cache)
                expected, conv, state = official_step(x[:, None], conv, state)
                torch.testing.assert_close(ours, expected[:, 0], atol=3e-7, rtol=1e-5)
                torch.testing.assert_close(cache[0], conv)
                torch.testing.assert_close(cache[1], state, atol=2e-7, rtol=1e-5)

    def test_sequence_step_and_chunk_equivalence(self):
        model, x = self.block(), torch.randn(3, 7, 16)
        full, _ = model(x)
        a, state = model(x[:, :3]); b, state = model(x[:, 3:], state)
        torch.testing.assert_close(full, torch.cat((a, b), 1))
        state = None; values = []
        for t in range(x.shape[1]):
            y, state = model.step(x[:, t], state); values.append(y)
        torch.testing.assert_close(full, torch.stack(values, 1))

    def test_no_future_leakage(self):
        model, x = self.block(), torch.randn(2, 9, 16)
        full, _ = model(x)
        changed = x.clone(); changed[:, 5:] = 100 * torch.randn_like(changed[:, 5:])
        other, _ = model(changed)
        torch.testing.assert_close(full[:, :5], other[:, :5], atol=0, rtol=0)
        self.assertFalse(torch.equal(full[:, 5:], other[:, 5:]))

    def test_language_conditioned_prior_sequence_step_parity(self):
        prior = self.prior(); x = torch.randn(2, 6, 24); lang = torch.randn(2, 5, 24)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        cache = None
        for t in range(6):
            sequential = prior(x[:, t:t+1], lang, mask, cache)
            cache = sequential[2]
            parallel = prior(x[:, :t+1], lang, mask)
            torch.testing.assert_close(sequential[0], parallel[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(sequential[1], parallel[1], atol=1e-6, rtol=1e-5)
        # Padding must not affect dynamics; valid instruction content must.
        modified = lang.clone(); modified[0, 3:] += 100
        torch.testing.assert_close(prior(x, modified, mask)[0], prior(x, lang, mask)[0])
        modified[:, 0] += 100
        self.assertFalse(torch.allclose(prior(x, modified, mask)[0], prior(x, lang, mask)[0]))

    def test_branch_cache_immutable_and_overshoot_stacking(self):
        prior = self.prior(); lang = torch.randn(2, 5, 24); mask = torch.ones(2, 5, dtype=torch.bool)
        _, _, first = prior(torch.randn(2, 1, 24), lang, mask)
        before = [[t.clone() for t in layer] for layer in first]
        branches = [torch.randn(2, 1, 24) for _ in range(3)]
        independent = [prior(x, lang, mask, first) for x in branches]
        stacked_cache = [tuple(torch.cat([tensor]*3, 0) for tensor in layer) for layer in first]
        batched = prior(torch.cat(branches, 0), lang.repeat(3, 1, 1), mask.repeat(3, 1), stacked_cache)
        torch.testing.assert_close(batched[0], torch.cat([v[0] for v in independent], 0), atol=1e-6, rtol=1e-5)
        for layer_idx, layer in enumerate(first):
            for i, tensor in enumerate(layer):
                torch.testing.assert_close(tensor, before[layer_idx][i], atol=0, rtol=0)
                torch.testing.assert_close(batched[2][layer_idx][i],
                    torch.cat([v[2][layer_idx][i] for v in independent], 0), atol=1e-6, rtol=1e-5)
        with torch.inference_mode():
            prior(branches[0], lang, mask, first)
        loss = sum(v[0].square().sum()+v[1].square().sum() for v in independent)
        loss.backward()
        for name, parameter in prior.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_invalid_cache_or_mask_rejected(self):
        p = self.prior(); x = torch.randn(2, 1, 24); lang = torch.randn(2, 3, 24)
        with self.assertRaises(ValueError): p(x, lang, torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError): p(x, lang, torch.ones(2, 3, dtype=torch.bool), [])

    def test_world_model_only_freezes_shared_token_type(self):
        model = nn.Module(); model.config = type('Config', (), {})()
        model.embeddings = nn.Embedding(3, 24)
        model.img_embeddings = nn.Module(); model.img_embeddings.token_type = model.embeddings
        model.lang_encoder = nn.Linear(24, 24)
        model.prior = self.prior(); model.policy = nn.Linear(24, 2)
        trainable = freeze_for_world_model_pretraining(model)
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith('prior.') for n in trainable))
        self.assertFalse(model.embeddings.weight.requires_grad)
        self.assertFalse(model.img_embeddings.training)
        self.assertFalse(model.lang_encoder.training)
        self.assertFalse(model.policy.weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
