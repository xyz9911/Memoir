#!/usr/bin/env python3
"""Engineering QA only: native GT trajectories -> frozen encoders -> WM update.

No cache, checkpoint, experiment directory or stage-completion marker is written.
Two official train_fit instructions receive one in-memory optimizer update; one
train_dev instruction is evaluated under no_grad and never optimized. This is a
call-chain probe, not a navigation score or a completed pretraining experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]

import numpy as np
import torch

import drama_pretrain as pre


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def model_digest(model, frozen_only=False):
    """Include buffers and shared-state aliases, not only distinct Parameters."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if frozen_only and name in trainable:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class InMemoryTrajectories(pre.CachedTrajectories):
    """Use the production padding/mask collator without creating an HDF5 file."""
    def __init__(self, records):
        self.ids = list(records)
        self.file = records


def describe(ex):
    return {'instr_id': ex['instr_id'], 'scan': ex['scan'], 'path_id': ex['path_id'],
            'path_length': len(ex['path']), 'tokens': len(ex['instr_encoding'])}


def assert_encoders_frozen(model):
    for name in ('embeddings', 'lang_encoder', 'img_embeddings'):
        module = getattr(model, name)
        require(not module.training, f'{name} left eval mode')
        require(not any(p.requires_grad for p in module.parameters()), f'{name} is trainable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering-qa', action='store_true', required=True,
                        help='Explicitly acknowledge that this does not advance a production stage.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    opts = parser.parse_args()
    require(opts.engineering_qa, 'Only explicit engineering QA is supported')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and
            os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'Prepared offline assets required')
    require(os.environ.get('PYTHONHASHSEED') == '0', 'Launch with PYTHONHASHSEED=0')
    device = torch.device(opts.device)
    require(device.type == 'cuda' and torch.cuda.is_available(),
            'The native panorama helper requires a CUDA runtime')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    cfg = json.loads(Path(opts.config).read_text())
    started = time.monotonic()

    from nav_src.envs.r2r.data_utils import construct_instrs
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from utils.data import ImageFeaturesDB

    split = json.loads(Path(cfg['split_manifest']).read_text())
    fit_scans, dev_scans = set(split['train_fit']), set(split['train_dev'])
    require(fit_scans and dev_scans and not fit_scans & dev_scans, 'Invalid train split')
    tours = json.loads(Path(cfg['tours']).read_text())['train']
    members = {instr for scan_tours in tours.values() for tour in scan_tours for instr in tour}
    data = construct_instrs(str(Path(cfg['annotations']).parent), 'r2r', ['train'],
                           'bert', 512, is_test=False)
    fit = sorted((ex for ex in data if ex['scan'] in fit_scans and ex['instr_id'] in members),
                 key=lambda ex: (len(ex['path']), ex['instr_id']))
    dev = sorted((ex for ex in data if ex['scan'] in dev_scans and ex['instr_id'] in members),
                 key=lambda ex: (len(ex['path']), ex['instr_id']))
    require(len(fit) >= 2 and dev, 'Need two official train_fit samples and one train_dev sample')
    chosen_fit, chosen_dev = [fit[0], fit[-1]], [dev[0]]
    require(len(chosen_fit[0]['path']) < len(chosen_fit[1]['path']),
            'Select distinct short/long real trajectories for the padding check')
    chosen = chosen_fit + chosen_dev
    scans = {ex['scan'] for ex in chosen}
    environment_data = [ex for ex in data if ex['scan'] in scans and ex['instr_id'] in members]
    environment_tours = {scan: tours[scan] for scan in sorted(scans)}
    env = IR2RNavBatch(ImageFeaturesDB(cfg['feature_file'], 512), environment_data,
                      cfg['connectivity'], environment_tours, num_robots=1,
                      batch_size=1, angle_feat_size=4, seed=cfg['seed'], name='engineering_qa')
    model = pre.make_model(cfg, str(device)).eval()
    assert_encoders_frozen(model)
    frozen_before = model_digest(model, frozen_only=True)
    helper = object.__new__(GMapNavAgent)
    helper.args = pre.model_args(cfg)
    helper.store_obs = {}
    encoded = {}
    for ex in chosen:
        record = pre.encode_trajectory(model, env, helper, ex)
        for key, value in record.items():
            require(value.dtype == np.float32 and bool(np.isfinite(value).all()),
                    f'Nonfinite/non-FP32 encoded {key}: {ex["instr_id"]}')
        require(record['visual'].shape == (len(ex['path']), 768), 'Wrong visual shape')
        require(record['language'].shape == (len(ex['instr_encoding']), 768), 'Wrong language shape')
        require(record['distance'].shape == (len(ex['path']),), 'Wrong distance shape')
        require(float(record['distance'][-1]) == 0., 'Terminal GT distance must be zero')
        encoded[ex['instr_id']] = record
        print('DRAMA_DATA_QA_ENCODED ' + json.dumps(describe(ex)), flush=True)
    fit_data = InMemoryTrajectories({ex['instr_id']: encoded[ex['instr_id']] for ex in chosen_fit})
    dev_data = InMemoryTrajectories({ex['instr_id']: encoded[ex['instr_id']] for ex in chosen_dev})
    batch = fit_data.batch([0, 1], device)
    require(batch['mask'].sum(1).tolist() == [len(ex['path']) for ex in chosen_fit], 'Padding mask differs')
    require(not any(t.requires_grad for t in batch.values()), 'Cached encoder input has a gradient graph')

    model.train()
    assert_encoders_frozen(model)
    optimizer = pre.build_optimizer(model, cfg)
    qa_lr = pre.scheduled_lr(1, cfg)
    for group in optimizer.param_groups:
        group['lr'] = qa_lr
    optimizer.zero_grad(set_to_none=True)
    before_update = model_digest(model)
    loss, pieces = pre.wm_loss(model, batch, cfg['overshoot_horizon'], **pre.loss_weights(cfg))
    require(bool(torch.isfinite(loss)), 'Nonfinite real-trajectory loss')
    loss.backward()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                    f'Missing or nonfinite WM gradient: {name}')
            require(float(parameter.grad.detach().abs().sum()) > 0, f'Zero WM gradient: {name}')
        else:
            require(parameter.grad is None, f'Frozen parameter received a gradient: {name}')
    gradient_prefixes = {prefix: sum(float(p.grad.detach().abs().sum()) for name, p in trainable
                                    if name.startswith(prefix)) for prefix in pre.WM_PREFIXES}
    require(all(value > 0 for value in gradient_prefixes.values()), 'A world-model component has zero gradients')
    grad_norm = torch.nn.utils.clip_grad_norm_((p for _, p in trainable), cfg['grad_clip'],
                                             error_if_nonfinite=True)
    optimizer.step()
    require(model_digest(model) != before_update, 'Optimizer did not update the world model')
    frozen_after = model_digest(model, frozen_only=True)
    require(frozen_after == frozen_before, 'Frozen non-world-model weights changed')

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    assert_encoders_frozen(model)
    before_dev = model_digest(model)
    with pre.isolated_rng(917), torch.no_grad():
        dev_loss, dev_pieces = pre.wm_loss(model, dev_data.batch([0], device),
                                         cfg['overshoot_horizon'], **pre.loss_weights(cfg))
    require(not dev_loss.requires_grad and bool(torch.isfinite(dev_loss)), 'Invalid no_grad dev check')
    require(model_digest(model) == before_dev, 'Read-only dev pass changed model state')
    require(all(p.grad is None for p in model.parameters()), 'Dev pass created gradients')
    torch.cuda.synchronize(device)
    report = {'schema': 'memoir-drama-real-data-engineering-qa-v1', 'passed': True,
              'engineering_qa': True, 'formal_training_updates': 0,
              'fit_qa_optimizer_updates': 1, 'dev_optimizer_updates': 0,
              'persistent_training_artifacts_written': False,
              'train_fit': [describe(ex) for ex in chosen_fit],
              'train_dev': [describe(ex) for ex in chosen_dev],
              'fit_qa_loss': float(loss.detach()), 'fit_pieces': {k: float(v.detach()) for k, v in pieces.items()},
              'dev_qa_loss': float(dev_loss), 'dev_pieces': {k: float(v) for k, v in dev_pieces.items()},
              'grad_norm': float(grad_norm), 'gradient_l1_by_component': gradient_prefixes,
              'trainable_tensors': len(trainable),
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'frozen_sha256': frozen_after, 'optimizer': pre.optimizer_contract(cfg),
              'qa_update_lr': qa_lr, 'device': str(device), 'gpu': torch.cuda.get_device_name(device),
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
              'seconds': time.monotonic() - started,
              'source_files': pre.code_manifest(), 'config_sha256': pre.sha(opts.config),
              'input_sha256': {str(Path(cfg[key]).resolve()): pre.sha(cfg[key])
                               for key in ('init_checkpoint', 'annotations', 'tours', 'split_manifest')},
              'runtime': pre.runtime_identity()}
    print('DRAMA_DATA_ENGINEERING_QA ' + json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
