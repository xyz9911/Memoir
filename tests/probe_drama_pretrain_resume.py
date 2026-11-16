#!/usr/bin/env python3
"""Engineering-only CUDA verification of the production WM snapshot/resume path.

Uses two official train_fit GT trajectories in memory. Saves a SHA-verified
frozen base and CompactStore snapshot to an explicitly guarded qa/pretrain
directory, then compares continuous update 2 against a newly constructed model
resuming update 2. No dev optimization, formal cache, or completion marker.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT), str(Path(__file__).resolve().parent)]

import numpy as np
import torch

import drama_pretrain as pre
from probe_drama_data import InMemoryTrajectories, assert_encoders_frozen, describe, model_digest, require


def tree_sha(value):
    h = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(('tensor:' + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(('numpy:' + str(item.dtype) + str(item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            h.update((type(item).__name__ + ':' + repr(item)).encode())
    add(value)
    return h.hexdigest()


def assert_equal(expected, actual, label='state'):
    require(type(expected) is type(actual), f'{label} type differs')
    if torch.is_tensor(expected):
        require(expected.dtype == actual.dtype and expected.shape == actual.shape,
                f'{label} dtype/shape differs')
        require(torch.equal(expected.cpu(), actual.cpu()), f'{label} is not bitwise equal')
    elif isinstance(expected, dict):
        require(expected.keys() == actual.keys(), f'{label} keys differ')
        for key in expected:
            assert_equal(expected[key], actual[key], f'{label}.{key}')
    elif isinstance(expected, (tuple, list)):
        require(len(expected) == len(actual), f'{label} length differs')
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equal(left, right, f'{label}[{index}]')
    else:
        require(expected == actual, f'{label} value differs: {expected!r} vs {actual!r}')


def encode_fit(model, cfg):
    from nav_src.envs.r2r.data_utils import construct_instrs
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from utils.data import ImageFeaturesDB
    split = json.loads(Path(cfg['split_manifest']).read_text())
    fit_scans, dev_scans = set(split['train_fit']), set(split['train_dev'])
    require(fit_scans and not fit_scans & dev_scans, 'Invalid fit/dev split')
    tours = json.loads(Path(cfg['tours']).read_text())['train']
    members = {instr for scan_tours in tours.values() for tour in scan_tours for instr in tour}
    data = construct_instrs(str(Path(cfg['annotations']).parent), 'r2r', ['train'], 'bert', 512, is_test=False)
    fit = sorted((ex for ex in data if ex['scan'] in fit_scans and ex['instr_id'] in members),
                 key=lambda ex: (len(ex['path']), ex['instr_id']))
    require(len(fit) >= 2, 'Need two official fit examples')
    chosen = [fit[0], fit[-1]]
    scans = {ex['scan'] for ex in chosen}
    env = IR2RNavBatch(ImageFeaturesDB(cfg['feature_file'], 512),
                      [ex for ex in data if ex['scan'] in scans and ex['instr_id'] in members],
                      cfg['connectivity'], {scan: tours[scan] for scan in sorted(scans)},
                      num_robots=1, batch_size=1, angle_feat_size=4, seed=cfg['seed'], name='engineering_fit_only')
    helper = object.__new__(GMapNavAgent)
    helper.args = pre.model_args(cfg); helper.store_obs = {}
    model.eval(); assert_encoders_frozen(model)
    records = {ex['instr_id']: pre.encode_trajectory(model, env, helper, ex) for ex in chosen}
    require(all(np.isfinite(value).all() for record in records.values() for value in record.values()),
            'Nonfinite encoded fit trajectory')
    return InMemoryTrajectories(records), [describe(ex) for ex in chosen]


def update(model, optimizer, sampler, data, cfg, step):
    model.train(); assert_encoders_frozen(model)
    optimizer.zero_grad(set_to_none=True)
    indices = sampler.take(2)
    batch = data.batch(indices, next(model.parameters()).device)
    loss, pieces = pre.wm_loss(model, batch, cfg['overshoot_horizon'], **pre.loss_weights(cfg))
    require(bool(torch.isfinite(loss)), 'Nonfinite QA loss')
    loss.backward()
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.abs().sum()) > 0, f'Invalid gradient: {name}')
            parameters.append(parameter)
        else:
            require(parameter.grad is None, f'Frozen gradient: {name}')
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'], error_if_nonfinite=True)
    lr = pre.scheduled_lr(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return {'step': step, 'indices': indices, 'loss': float(loss.detach()), 'grad_norm': float(norm),
            'lr': lr, 'pieces': {key: float(value.detach()) for key, value in pieces.items()}}


def snapshot(model, optimizer, sampler, cfg, identity, step):
    return {'schema': pre.SCHEMA, 'identity_sha256': pre.json_sha(identity), 'step': step,
            'best_dev': float('inf'), 'history': [],
            'trainable_state': {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
            'optimizer': pre.cpu_tree(optimizer.state_dict()), 'scheduler': pre.scheduler_state(step, cfg),
            'sampler': sampler.state(), 'rng': pre.rng_state()}


def run(opts, cfg):
    started = time.monotonic()
    qa_root = Path(opts.qa_root).resolve()
    require(qa_root.is_relative_to(Path(cfg['storage_mount']).resolve() / 'qa' / 'pretrain'),
            'QA root must be inside the selected mount qa/pretrain directory')
    require(not qa_root.exists(), 'Fresh engineering probe refuses an existing QA directory')
    cfg = {**cfg, 'run_dir': str(qa_root / 'local'), 'backup_dir': str(qa_root / 'backup')}
    storage = pre.verify_storage_mount(cfg)
    host = pre.verify_host_volume(cfg['host_volume'])
    pre.space_guard([cfg['storage_mount'], cfg['host_volume']], cfg['min_free_gib'])
    device = torch.device(opts.device)
    require(device.type == 'cuda' and torch.cuda.is_available(), 'Actual CUDA execution required')
    torch.cuda.set_device(device); torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats(device)
    pre.configure_runtime(cfg)
    # cuBLAS keeps a process-local workspace even when no tensors/models remain.
    # Measure it separately; destruction is checked with weakrefs below.
    warm_x = torch.ones(32, 512, device=device, requires_grad=True)
    warm_w = torch.ones(512, 512, device=device, requires_grad=True)
    warm_loss = (warm_x @ warm_w).square().sum(); warm_loss.backward()
    del warm_x, warm_w, warm_loss
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    workspace_baseline = torch.cuda.memory_allocated(device)
    model = pre.make_model(cfg, str(device))
    data, chosen = encode_fit(model, cfg)
    frozen_tensor_sha = model_digest(model, frozen_only=True)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key not in trainable_names}
    # Two frozen bases plus model/Adam snapshot copies, reserving 12 GiB after all writes.
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    pre.space_guard([cfg['storage_mount'], cfg['host_volume']], cfg['min_free_gib'],
                    additional=2 * pre.tensor_bytes(frozen) + 6 * trainable_bytes + 64 * 1024**2)
    local, backup = Path(cfg['run_dir']), Path(cfg['backup_dir'])
    local.mkdir(parents=True); backup.mkdir()
    opts.qa_started = True
    pre.atomic_torch(local / 'frozen-base.pt', frozen); del frozen
    frozen_file_sha = pre.verified_copy(local / 'frozen-base.pt', backup / 'frozen-base.pt')
    inputs = pre.cache_inputs(cfg)
    identity = {'engineering_qa': True, 'formal_training_updates': 0,
                'contract': {'config': cfg, 'source_files': pre.code_manifest(), 'inputs': inputs,
                             'probe_sha256': pre.sha(__file__),
                             'probe_data_helper_sha256': pre.sha(Path(__file__).with_name('probe_drama_data.py')),
                             'fit_examples': chosen, 'in_memory_encoding_sha256': tree_sha(data.file),
                             'runtime': pre.runtime_identity(), 'optimizer': pre.optimizer_contract(cfg)},
                'frozen_base_sha256': frozen_file_sha, 'frozen_tensor_sha256': frozen_tensor_sha,
                'backup_mount': pre.backup_mount_identity(backup, backend='filesystem'),
                'host_mount': host, 'storage_mount': storage}
    pre.write_json(local / 'identity.json', identity); pre.write_json(backup / 'identity.json', identity)
    store = pre.CompactStore(local / 'checkpoints', backup / 'checkpoints', cfg['host_volume'],
                             cfg['min_free_gib'], identity['backup_mount'], host, cfg, storage)
    optimizer = pre.build_optimizer(model, cfg); sampler = pre.Sampler(len(data.ids))
    first = update(model, optimizer, sampler, data, cfg, 1)
    require(model_digest(model, frozen_only=True) == frozen_tensor_sha, 'First update modified frozen state')
    state = snapshot(model, optimizer, sampler, cfg, identity, 1)
    checkpoint_id = store.save(state, {'kind': 'engineering_pretrain_resume'}, step=1)
    del state
    continuous = update(model, optimizer, sampler, data, cfg, 2)
    expected = snapshot(model, optimizer, sampler, cfg, identity, 2)
    require(model_digest(model, frozen_only=True) == frozen_tensor_sha, 'Continuous update modified frozen state')
    expected_model_sha = model_digest(model)
    old_model_ref, old_optimizer_ref = weakref.ref(model), weakref.ref(optimizer)
    old_tensor_refs = [weakref.ref(value) for value in list(model.parameters()) + list(model.buffers())]
    del optimizer, sampler, model
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    released_allocated = torch.cuda.memory_allocated(device)
    require(old_model_ref() is None and old_optimizer_ref() is None and
            all(reference() is None for reference in old_tensor_refs),
            'Previous model/optimizer/parameters remain alive before reconstruction')

    model = pre.make_model(cfg, str(device))
    require(model_digest(model, frozen_only=True) == frozen_tensor_sha, 'Fresh frozen construction differs')
    require(pre.code_manifest() == identity['contract']['source_files'], 'Source changed during QA')
    require(pre.cache_inputs(cfg) == inputs, 'Input assets or runtime changed during QA')
    require(tree_sha(data.file) == identity['contract']['in_memory_encoding_sha256'], 'Encoded trajectories changed')
    for path in (local / 'frozen-base.pt', backup / 'frozen-base.pt'):
        require(pre.sha(path) == frozen_file_sha, 'Frozen base readback SHA mismatch')
    require(json.loads((local / 'identity.json').read_text()) == json.loads((backup / 'identity.json').read_text()) == identity,
            'Frozen identity copy differs')
    frozen = torch.load(backup / 'frozen-base.pt', map_location='cpu', weights_only=True)
    missing = model.load_state_dict(frozen, strict=False)
    require(not missing.unexpected_keys and set(missing.missing_keys) == trainable_names, 'Frozen/trainable partition differs')
    del frozen
    optimizer = pre.build_optimizer(model, cfg)
    state, _, manifest = store.restore(which='latest')
    require(state['identity_sha256'] == pre.json_sha(identity) and state['step'] == manifest['step'] == 1,
            'Snapshot identity or step differs')
    require(set(state['trainable_state']) == trainable_names, 'Snapshot parameter keys differ')
    model.load_state_dict(state['trainable_state'], strict=False)
    optimizer.load_state_dict(state['optimizer'])
    pre.validate_scheduler(state['scheduler'], state['step'], cfg, optimizer)
    sampler = pre.Sampler(len(data.ids), state['sampler'])
    pre.set_rng(state['rng']); del state
    resumed = update(model, optimizer, sampler, data, cfg, 2)
    actual = snapshot(model, optimizer, sampler, cfg, identity, 2)
    assert_equal(continuous, resumed, 'update_2')
    assert_equal(expected, actual, 'full_training_state_after_update_2')
    require(model_digest(model) == expected_model_sha, 'Whole model differs after resumed update')
    require(model_digest(model, frozen_only=True) == frozen_tensor_sha, 'Resumed update changed frozen state')
    torch.cuda.synchronize(device)
    result = {'engineering_qa': True, 'passed': True, 'bitwise_equal': True,
              'formal_training_updates': 0, 'dev_optimizer_updates': 0,
              'continuous_qa_updates': 2, 'resumed_from_update': 1, 'resumed_update': 2,
              'first_update': first, 'continuous_second_update': continuous, 'resumed_second_update': resumed,
              'checkpoint_id': checkpoint_id, 'backup_sha_verified': True,
              'identity_sha256': pre.json_sha(identity), 'frozen_base_sha256': frozen_file_sha,
              'frozen_tensor_sha256': frozen_tensor_sha, 'final_model_sha256': expected_model_sha,
              'final_optimizer_sha256': tree_sha(actual['optimizer']), 'final_rng_sha256': tree_sha(actual['rng']),
              'scheduler': actual['scheduler'], 'final_sampler': actual['sampler'],
              'old_model_optimizer_and_tensors_collected': True,
              'cuda_workspace_baseline_bytes': workspace_baseline,
              'released_cuda_allocated_bytes_before_rebuild': released_allocated,
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
              'seconds': time.monotonic() - started, 'fit_examples': chosen,
              'scope': 'Real fit GT CUDA engineering QA only; no formal cache, stage completion, or model score.'}
    pre.verify_storage_mount(cfg, storage)
    pre.space_guard([cfg['storage_mount'], cfg['host_volume']], cfg['min_free_gib'])
    pre.write_json(qa_root / 'result.json', result)
    print('DRAMA_PRETRAIN_RESUME_ENGINEERING_QA ' + json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering-qa', action='store_true', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--qa-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    opts = parser.parse_args()
    opts.qa_started = False
    require(opts.engineering_qa and os.environ.get('PYTHONHASHSEED') == '0', 'Explicit QA and PYTHONHASHSEED=0 required')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'Offline assets required')
    cfg = json.loads(Path(opts.config).read_text())
    try:
        run(opts, cfg)
    except Exception:
        root = Path(opts.qa_root)
        if opts.qa_started and root.is_dir():
            pre.verify_storage_mount(cfg)
            pre.space_guard([cfg['storage_mount'], cfg['host_volume']], cfg['min_free_gib'])
            pre.write_json(root / 'failure.json', {'engineering_qa': True, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
