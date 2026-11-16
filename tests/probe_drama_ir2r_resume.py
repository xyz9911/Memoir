#!/usr/bin/env python3
"""Engineering only: real MatterSim/CUDA full-state continuation, never a score.

Uses two-episode prefixes of three train_fit author tours. Fresh random Drama
world-model initialization is explicitly confined to this probe. Production
drama_ir2r.py still requires completed world-model pretraining.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts'))
import drama_ir2r as ir
from drama_pretrain import (CompactStore, atomic_torch, code_manifest, json_sha,
    rng_state, seed_all, sha, space_guard, verified_copy, verify_storage_mount, write_json)
from drama_mount_guard import backup_mount_identity


def signature(value):
    if torch.is_tensor(value):
        t = value.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy()
        return {'tensor': hashlib.sha256(raw).hexdigest(), 'dtype': str(t.dtype), 'shape': list(t.shape)}
    if isinstance(value, dict):
        return [[repr(k), signature(v)] for k, v in value.items()]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [signature(v) for v in value]]
    return value


def state_digests(state):
    return {k: json_sha(signature(v)) for k, v in state.items()}


def short_tours(cfg, args):
    from nav_src.envs.r2r.data_utils import construct_instrs
    rows = construct_instrs(args.anno_dir, 'r2r', ['train'], 'bert', args.max_instr_len)
    split = json.loads(Path(cfg['split_manifest']).read_text())
    tours = json.loads(Path(cfg['tours']).read_text())['train']
    subset = {}
    for scan in sorted(split['train_fit']):
        tour = next((tour for tour in tours[scan] if len(tour) >= 2), None)
        if tour:
            subset[scan] = [tour[:2]]
        if len(subset) == 3:
            break
    ids = {i for ts in subset.values() for tour in ts for i in tour}
    data = [x for x in rows if x['instr_id'] in ids]
    ir.require(len(subset) == 3 and len(data) == 6, 'Engineering subset coverage mismatch')
    return data, subset


def make_agent(cfg, data, tours):
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from utils.data import ImageFeaturesDB
    from vlnbert.drama_mamba2 import freeze_encoders
    seed_all(cfg['seed'])
    args = ir.author_args(cfg, {})
    args.bert_ckpt_file = cfg['engineering_initial_checkpoint']
    args.initialize_drama_world = True
    db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    env = IR2RNavBatch(db, data, args.connectivity_dir, tours, num_robots=1,
        batch_size=cfg['batch_size'], angle_feat_size=args.angle_feat_size, seed=args.seed, name='train_fit')
    agent = GMapNavAgent(args, env, rank=0)
    raw = agent.vln_bert.vln_bert
    for p in raw.parameters():
        p.requires_grad_(True)
    freeze_encoders(raw)
    trainable, _ = ir.parameter_partition(raw)
    named = dict(raw.named_parameters())
    groups = [{'params': [p for n, p in named.items() if n in trainable and not getattr(p, '_no_weight_decay', False)],
               'weight_decay': cfg.get('weight_decay', .01)},
              {'params': [p for n, p in named.items() if n in trainable and getattr(p, '_no_weight_decay', False)],
               'weight_decay': 0.}]
    agent.vln_bert_optimizer = torch.optim.AdamW(groups, lr=args.lr)
    agent.optimizers = agent.vln_bert_optimizer
    # A weak reference avoids retaining the complete GPU agent in the hook cycle.
    import weakref
    ref = weakref.ref(agent)
    agent.vln_bert_optimizer.register_step_pre_hook(
        lambda optimizer, a, k: ir.finite_step_guard(optimizer, ref().loss))
    agent.vln_bert.train()
    ir.check_runtime(agent, env)
    return agent


def record_update(agent, progress):
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    original = GMapNavAgent.rollout
    trace = []
    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        trace.append({'feedback': self.feedback,
            'trajectories': [{'instr_id': x['instr_id'], 'scan': x['scan'], 'path': x['path']} for x in result]})
        return result
    try:
        GMapNavAgent.rollout = observed
        agent.train(1, feedback='sample', extended_memory=True)
    finally:
        GMapNavAgent.rollout = original
    progress['completed_updates'] += 1
    ir.require(ir.complete_tour(agent.env), 'Engineering two-episode tours must finish on every update')
    loss = float(agent.loss.detach())
    print('PROBE_UPDATE', json.dumps({'step': progress['completed_updates'], 'loss': loss,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}), flush=True)
    return trace


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('PROBE_RELEASE_GPU', torch.cuda.memory_allocated()/1024**2, flush=True)


def run_probe(cfg, optional_batch2):
    ir.require(os.environ.get('PYTHONHASHSEED') == '0', 'PYTHONHASHSEED=0 required')
    ir.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUBLAS config required')
    storage = verify_storage_mount(cfg)
    host = ir.host_volume_identity(cfg['host_volume'], cfg['host_volume_source'])
    root, backup = Path(cfg['run_dir']), Path(cfg['backup_dir'])
    ir.require(not root.exists() and not backup.exists(), 'Engineering paths already exist; refusing overwrite')
    space_guard([cfg['storage_mount'], cfg['host_volume']], 12)
    root.mkdir(parents=True); backup.mkdir(parents=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    args = ir.author_args(cfg, {})
    data, tours = short_tours(cfg, args)
    agent = make_agent(cfg, data, tours)
    progress = {'completed_updates': 0, 'last_full_step': 0, 'history': [], 'best_metrics': None,
                'best_step': None, 'next_eval': 500, 'elapsed_seconds': 0.}
    raw = agent.vln_bert.vln_bert
    _, changing = ir.parameter_partition(raw)
    frozen = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items() if k not in changing}
    frozen_sig = json_sha(signature(frozen))
    atomic_torch(root/'frozen-base.pt', frozen)
    frozen_sha = verified_copy(root/'frozen-base.pt', backup/'frozen-base.pt')
    identity = {'engineering_only': True, 'not_a_model_score': True,
        'initialization': 'official navigation base plus explicitly fresh random Drama world model; no pretrained-world-model claim',
        'tour_semantics': 'two-episode prefixes of three author train_fit tours; engineering coverage only',
        'tours': tours, 'config': cfg, 'source_files': code_manifest(), 'probe_sha256': sha(__file__),
        'frozen_base_sha256': frozen_sha, 'initial_checkpoint_sha256': sha(cfg['engineering_initial_checkpoint']),
        'host_mount': host, 'storage_mount': storage,
        'feature_sha256': sha(args.img_ft_file), 'split_sha256': sha(cfg['split_manifest']),
        'annotations_sha256': sha(Path(args.anno_dir)/'R2R_train_enc.json'),
        'author_tours_sha256': sha(cfg['tours']), 'environment': ir.static_environment(agent.env),
        'connectivity_sha256': {s: sha(Path(args.connectivity_dir)/(s+'_connectivity.json')) for s in tours},
        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'deterministic_algorithms': True, 'tf32': False}
    identity_sha = json_sha(identity)
    write_json(root/'identity.json', identity); write_json(backup/'identity.json', identity)
    store = CompactStore(root/'checkpoints', backup/'checkpoints', cfg['host_volume'], 12,
        backup_mount_identity(backup, backend='filesystem'), host, cfg, storage)
    with store.lock():
        for _ in range(2):
            record_update(agent, progress)
        progress['last_full_step'] = progress['completed_updates']
        state = ir.runtime_state(agent, progress, identity_sha)
        before_rng = json_sha(signature(rng_state()))
        checkpoint_id = store.save(state, {'engineering_only': True, 'frozen_base_sha256': frozen_sha}, step=2)
        ir.require(before_rng == json_sha(signature(rng_state())), 'Checkpoint IO consumed training RNG')
        del state
        uninterrupted_trace = record_update(agent, progress)
        uninterrupted_state = ir.runtime_state(agent, progress, identity_sha)
        expected = state_digests(uninterrupted_state)
        frozen_after = {k: v for k, v in raw.state_dict().items() if k not in changing}
        ir.require(json_sha(signature(frozen_after)) == frozen_sig, 'Frozen encoder parameters changed')
        del frozen_after, uninterrupted_state, raw, agent
        release_gpu()
        phase1_peak = torch.cuda.max_memory_allocated()/1024**2
        restored = make_agent(cfg, data, tours)
        ir.require(ir.static_environment(restored.env) == identity['environment'],
                   'Reconstructed environment/graph identity changed')
        saved, _, manifest = store.restore(which='latest')
        ir.require(manifest['step'] == 2 and sha(backup/'frozen-base.pt') == frozen_sha,
                   'Backup reference/base mismatch')
        restored_progress = ir.restore_runtime(restored, saved, identity_sha, frozen)
        ir.require(restored_progress['completed_updates'] == 2, 'Restore step mismatch')
        del saved, frozen
        resumed_trace = record_update(restored, restored_progress)
        resumed_state = ir.runtime_state(restored, restored_progress, identity_sha)
        actual = state_digests(resumed_state)
        mismatches = [k for k in expected if expected[k] != actual[k]]
        result = {'engineering_only': True, 'strict_resume_gpu_probe_passed': not mismatches and uninterrupted_trace == resumed_trace,
            'short_tour_memory_probe_not_full_run_capacity_guarantee': True,
            'snapshot_step': 2, 'compared_after_update': 3, 'mismatching_fields': mismatches,
            'actual_navigation_trajectories_equal': uninterrupted_trace == resumed_trace,
            'expected_field_digests': expected, 'actual_field_digests': actual,
            'uninterrupted_trace': uninterrupted_trace, 'resumed_trace': resumed_trace,
            'backup_checkpoint_id': checkpoint_id, 'backup_manifest': manifest,
            'frozen_base_sha256': frozen_sha, 'frozen_parameters_unchanged': True,
            'batch1_peak_allocated_mib': phase1_peak,
            'remaining_allocated_mib': torch.cuda.memory_allocated()/1024**2}
        del resumed_state, restored
        release_gpu()
        write_json(root/'result.json', result)
        if not result['strict_resume_gpu_probe_passed']:
            raise RuntimeError('Strict resume mismatch: '+repr(mismatches))
    if optional_batch2:
        batch2_cfg = {**cfg, 'batch_size': 2}
        torch.cuda.reset_peak_memory_stats()
        batch2 = None
        try:
            batch2 = make_agent(batch2_cfg, data, tours)
            p = {'completed_updates': 0}
            traces = [record_update(batch2, p) for _ in range(2)]
            result['batch2'] = {'engineering_only': True, 'updates': 2, 'fits': True,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2, 'traces': traces}
        except torch.OutOfMemoryError as exc:
            result['batch2'] = {'engineering_only': True, 'fits': False,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2, 'error': str(exc)}
        finally:
            del batch2
            release_gpu()
        write_json(root/'result.json', result)
    print('PROBE_RESULT', json.dumps({k: v for k, v in result.items() if k not in
        {'expected_field_digests','actual_field_digests','uninterrupted_trace','resumed_trace','backup_manifest'}}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--qa-root', required=True)
    p.add_argument('--batch2', action='store_true')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    qa = Path(a.qa_root)
    cfg.update(run_dir=str(qa/'local'), backup_dir=str(qa/'backup'), batch_size=1,
        engineering_only=True, engineering_initial_checkpoint=str(Path(cfg['datasets'])/'R2R/pretrained/grduet_tssm_dropout.pt'))
    os.chdir(REPO)
    started = time.time()
    try:
        run_probe(cfg, a.batch2)
    except Exception:
        if Path(cfg['run_dir']).is_dir():
            write_json(Path(cfg['run_dir'])/'failure.json', {'engineering_only': True,
                'elapsed_seconds': time.time()-started, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
