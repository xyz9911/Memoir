"""No dataset/checkpoint writes: actual model factory and frozen-encoder optimizer smoke."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnbert.vlnbert_init import get_vlnbert_models
from vlnbert.drama_mamba2 import freeze_for_world_model_pretraining, WORLD_MODEL_PREFIXES


def digest_frozen(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def digest_world(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if name.startswith(WORLD_MODEL_PREFIXES):
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=19)
    opts = p.parse_args()
    a = SimpleNamespace(bert_ckpt_file=opts.initial_checkpoint, tokenizer='bert',
        world_model_backend='drama_mamba2', freeze_encoders=True, initialize_drama_world=True,
        image_feat_size=512, angle_feat_size=4, obj_feat_size=0, num_l_layers=9,
        num_pano_layers=2, num_x_layers=4, graph_sprels=True, fusion='dynamic',
        fix_lang_embedding=True, fix_pano_embedding=True, fix_local_branch=False,
        drama_d_model=512, drama_layers=2, drama_d_state=16, drama_headdim=64, drama_dropout=0.1)
    torch.manual_seed(opts.seed); torch.set_num_threads(2)
    model = get_vlnbert_models(a)
    world_init_sha = digest_world(model)
    convolution_stats = []
    for layer in model.prior.layers:
        for name, p in layer.conv1d.named_parameters():
            assert bool(torch.isfinite(p).all()), 'Uninitialized convolution ' + name
            assert float(p.std()) > 0, 'Degenerate convolution ' + name
            assert float(p.abs().max()) <= layer.d_conv ** -.5, 'Invalid convolution initialization ' + name
        convolution_stats.append({'weight_std': float(layer.conv1d.weight.std()),
                                  'bias_std': float(layer.conv1d.bias.std())})
    original = torch.load(opts.initial_checkpoint, map_location='cpu', weights_only=True)
    for name in ('embeddings.word_embeddings.weight', 'img_embeddings.img_linear.weight',
                 'lang_encoder.layer.0.attention.self.query.weight'):
        assert torch.equal(model.state_dict()[name], original['bert.'+name]), name
    del original
    trainable = freeze_for_world_model_pretraining(model)
    model.to(opts.device).train()
    assert not any(getattr(model, n).training for n in ('embeddings','lang_encoder','img_embeddings'))
    before = digest_frozen(model)
    dev = torch.device(opts.device)
    txt_mask = torch.ones(2, 9, device=dev, dtype=torch.bool)
    text = model.forward_text(torch.randint(0, 100, (2, 9), device=dev), txt_mask)
    pano, _ = model.forward_panorama_per_step(torch.randn(2,36,512,device=dev), None,
        torch.randn(2,36,7,device=dev), torch.zeros(2,36,device=dev,dtype=torch.long),
        torch.tensor([36,36],device=dev), None)
    assert not text.requires_grad and not pano.requires_grad
    obs = pano.mean(1); deter = model.lang_init(text[:,0]); post = model.forward_posterior(deter,obs)
    state = torch.cat([post['stoch'],deter],-1)[:,None]
    loss = 0; cache = None
    for _ in range(3):
        out = model.forward_transit(state,obs,text,txt_mask,cache)
        state = torch.cat([out['posterior_stoch'],out['deter']],-1)[:,None]
        cache = out['past_key_values']
        recon = model.forward_reconstruct(state[:,0])
        loss = loss + recon['reward'].square().mean()+recon['img_embed'].square().mean()
        loss = loss + torch.distributions.kl_divergence(out['posterior_dist'],out['prior_dist']).mean()
    loss = loss + model.forward_transform(obs).square().mean()
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for n,p in model.named_parameters() if p.requires_grad)
    assert all(float(p.grad.abs().sum()) > 0 for n,p in model.named_parameters() if p.requires_grad), 'A trainable parameter has zero gradient'
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4).step()
    after = digest_frozen(model)
    assert before == after, 'Frozen non-world-model weights changed'
    report = {'passed':True, 'loss':float(loss.detach()), 'frozen_sha256':after,
        'seed':opts.seed, 'world_initial_sha256':world_init_sha, 'convolution_initialization':convolution_stats,
        'trainable_tensors':len(trainable),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'device':opts.device}
    if dev.type == 'cuda': report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
    print('DRAMA_FACTORY_SMOKE '+json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
