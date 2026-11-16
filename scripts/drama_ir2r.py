#!/usr/bin/env python3
"""Frozen-encoder Drama IR2R, fitting/selection on complete author tours in train_fit/train_dev.

The update budget and evaluation intervals are minimum thresholds. Checkpoint,
evaluation and graceful stop happen at the next complete tour/update boundary.
No native MatterSim pose or live tour graph is silently serialized/restored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SCHEMA = 'memoir-drama-ir2r-strict-tour-v1'
NUMERICS = {'deterministic_algorithms': True, 'cudnn_deterministic': True,
            'cudnn_benchmark': False, 'matmul_tf32': False, 'cudnn_tf32': False,
            'cublas_workspace_config': ':4096:8'}
ENV_MUTABLE = ('full_data', 'ix', 'tour_batch', 'tour_ended',
               'buffered_state_dict', 'sent_embed_store', 'batch', 'tour_batch_probe')
ENV_FIELDS = set(('angle_feat_size angle_feature batch_size buffered_state_dict '
    'connectivity_dir data data_to_tour_id env extra_obs full_data graphs gt_trajs '
    'hash_distance_map hash_node_map ix name num_robots scans seed sent_embed_store '
    'shortest_distances shortest_paths sim tour_batch tour_data tour_ended').split())
AGENT_FIELDS = set(('args criterion default_gpu device env full_results gmaps logs models '
    'optimizers rank results scanvp_cands store_obs vln_bert vln_bert_optimizer').split())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def assert_runtime_numerics(cfg):
    actual = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cudnn_deterministic': torch.backends.cudnn.deterministic,
              'cudnn_benchmark': torch.backends.cudnn.benchmark,
              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    require(actual == cfg['numerics'] and not torch.is_deterministic_algorithms_warn_only_enabled(),
            'Runtime numerics differ from the GPU-verified deterministic configuration')


def configure_numerics(cfg):
    # Never mutate the input config: the pipeline compares this exact frozen JSON.
    require(cfg.get('numerics') == NUMERICS,
            'Explicit numerics config required; use the GPU-verified deterministic configuration')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process/CUDA initialization')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert_runtime_numerics(cfg)


def digest(value):
    # Preserve dictionary insertion order: graph tie-breaking and sampler order matter.
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pack(value):
    """A weights_only-safe tree. Native objects and live autograd are rejected."""
    if isinstance(value, torch.Tensor):
        require(value.grad_fn is None, 'Live autograd tensor in continuation state')
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, 'Object array in continuation state')
        return {'__ndarray__': True, 'dtype': value.dtype.str,
                'data': torch.from_numpy(value.copy()).clone()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        require(all(type(k) in (int, str, float, tuple) for k in value), 'Unsupported mapping key')
        return {k: pack(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(pack(v) for v in value)
    if type(value) in (type(None), str, bool, int, float):
        return value
    raise RuntimeError('Unsupported continuation object: ' + str(type(value)))


def unpack(value):
    if isinstance(value, dict):
        if value.get('__ndarray__') is True:
            return value['data'].numpy().astype(value['dtype'], copy=True)
        return {k: unpack(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(unpack(v) for v in value)
    return value.clone() if torch.is_tensor(value) else value


def complete_tour(env):
    if env.tour_batch is None:
        return True
    require(len(env.batch) == len(env.tour_batch) == env.batch_size, 'Invalid batch dimensions')
    for cluster, tours in zip(env.batch, env.tour_batch):
        require(len(cluster) == len(tours) == env.num_robots, 'Invalid robot dimensions')
        if any(item is not None and item['instr_id'] != tour[-1]
               for item, tour in zip(cluster, tours)):
            return False
    return True


def canonical_boundary(agent):
    require(complete_tour(agent.env), 'Partial tour checkpoint rejected')
    # _next_minibatch(is_last=True) resets extra_obs before _oracle_phase;
    # the next rollout constructs all graph maps and starts new native episodes.
    agent.gmaps = []
    agent.store_obs = {}  # Write-only diagnostic cache in the bound author source.
    agent.env.extra_obs = None
    agent.loss = 0
    agent.vln_bert_optimizer.zero_grad(set_to_none=True)


def capture_env(env):
    require(complete_tour(env), 'Environment is not at complete tour boundary')
    db = env.env.feat_db
    return {'mutable': {k: pack(getattr(env, k)) for k in ENV_MUTABLE if hasattr(env, k)},
            'feature_cache_keys': list(db._feature_store),
            'native_pose': 'dead: next reset calls newEpisode for every live slot',
            'extra_obs': 'dead: next full-tour reset sets None'}


def restore_env(env, saved):
    require(set(saved['mutable']) >= set(ENV_MUTABLE[:6]), 'Incomplete environment state')
    for k in ENV_MUTABLE:
        if k in saved['mutable']:
            setattr(env, k, unpack(saved['mutable'][k]))
        elif hasattr(env, k):
            delattr(env, k)
    env.extra_obs = None
    require(complete_tour(env), 'Restored partial tour rejected')
    # Restore immutable feature cache insertion order from SHA-bound HDF5.
    db = env.env.feat_db
    db._feature_store.clear()
    for key in saved['feature_cache_keys']:
        scan, viewpoint = key.split('_', 1)
        db.get_image_feature(scan, viewpoint)


def static_environment(env):
    return {'name': env.name, 'batch_size': env.batch_size, 'num_robots': env.num_robots,
            'seed': env.seed, 'data': digest(env.data), 'tours': digest(env.tour_data),
            'data_to_tour_id': digest(env.data_to_tour_id),
            'hash_nodes': digest(env.hash_node_map), 'hash_distances': digest(env.hash_distance_map),
            'graph_order': {s: {'nodes': list(g.nodes),
                'edges': [[a, b, float(d['weight'])] for a, b, d in g.edges(data=True)]}
                for s, g in env.graphs.items()}}


def optimizer_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def finite_step_guard(optimizer, loss=None):
    """Run as a pre-step hook: no corrupt parameter/Adam update can occur."""
    if loss is not None:
        require(torch.is_tensor(loss) and bool(torch.isfinite(loss.detach()).all()),
                'Nonfinite IR2R loss before optimizer step')
    finite_by_device = defaultdict(list)
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                require(not p.grad.is_sparse, 'Sparse gradients unsupported')
                finite_by_device[p.grad.device].append(torch.isfinite(p.grad.detach()).all())
    require(bool(finite_by_device), 'No trainable gradients before optimizer step')
    require(all(bool(torch.stack(flags).all()) for flags in finite_by_device.values()),
            'Nonfinite gradient: optimizer step rejected before parameter mutation')


def parameter_partition(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # Include every buffer in the changing delta; tied parameters appear once in
    # named_parameters but all state_dict aliases must follow the same partition.
    aliases = {n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    changing = aliases | set(dict(model.named_buffers(remove_duplicate=False)))
    return trainable, changing


def runtime_state(agent, progress, identity_sha, dev_env=None):
    from drama_pretrain import rng_state
    canonical_boundary(agent)
    model = agent.vln_bert.vln_bert
    _, changing = parameter_partition(model)
    return {'schema': SCHEMA, 'identity_sha256': identity_sha,
            'strict_resume_supported': True, 'step': progress['completed_updates'],
            'boundary': 'complete_optimizer_update_and_all_batch_tours_complete',
            'trainable_state': {k: pack(v) for k, v in model.state_dict().items() if k in changing},
            'optimizer': pack(agent.vln_bert_optimizer.state_dict()),
            'optimizer_names': optimizer_names(model, agent.vln_bert_optimizer),
            'optimizer_class': type(agent.vln_bert_optimizer).__module__ + '.' + type(agent.vln_bert_optimizer).__qualname__,
            'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()},
            'module_modes': {n: m.training for n, m in agent.vln_bert.named_modules()},
            'environment': capture_env(agent.env),
            'dev_environment': capture_env(dev_env) if dev_env is not None else None,
            'scanvp_cands': pack(agent.scanvp_cands),
            'progress': pack(progress), 'logs': pack(dict(agent.logs)),
            'losses': pack(getattr(agent, 'losses', [])), 'feedback': getattr(agent, 'feedback', None),
            'rng': rng_state(), 'scheduler': None, 'amp_scaler': None}


def restore_runtime(agent, state, identity_sha, frozen, dev_env=None):
    from drama_pretrain import set_rng
    require(state['schema'] == SCHEMA and state['identity_sha256'] == identity_sha,
            'Strict resume identity/schema mismatch')
    require(state['boundary'] == 'complete_optimizer_update_and_all_batch_tours_complete',
            'Strict resume requires complete tour boundary')
    require(state['step'] == state['progress']['completed_updates'], 'Checkpoint step disagreement')
    model = agent.vln_bert.vln_bert
    _, changing = parameter_partition(model)
    require(set(state['trainable_state']) == changing, 'Delta parameter/buffer keys differ')
    require(not set(frozen) & changing, 'Frozen/delta keys overlap')
    require(state['optimizer_names'] == optimizer_names(model, agent.vln_bert_optimizer),
            'Optimizer parameter order changed')
    require(state['requires_grad'] == {n: p.requires_grad for n, p in model.named_parameters()},
            'Frozen parameter identity changed')
    require(state['optimizer_class'] == type(agent.vln_bert_optimizer).__module__ + '.' + type(agent.vln_bert_optimizer).__qualname__,
            'Optimizer class changed')
    model.load_state_dict({**frozen, **state['trainable_state']}, strict=True)
    agent.vln_bert_optimizer.load_state_dict(state['optimizer'])
    restore_env(agent.env, state['environment'])
    require((dev_env is None) == (state['dev_environment'] is None), 'Validation state contract changed')
    if dev_env is not None:
        restore_env(dev_env, state['dev_environment'])
    agent.scanvp_cands = unpack(state['scanvp_cands'])
    agent.logs = defaultdict(list, unpack(state['logs']))
    agent.losses = unpack(state['losses'])
    agent.feedback = state['feedback']
    modules = dict(agent.vln_bert.named_modules())
    require(set(modules) == set(state['module_modes']), 'Module set changed')
    for n, mode in state['module_modes'].items():
        modules[n].training = mode
    canonical_boundary(agent)
    set_rng(state['rng'])  # Last: construction, native simulator setup and IO cannot consume continuation RNG.
    return unpack(state['progress'])


def evaluate_navigation(agent, dev_env):
    from drama_pretrain import rng_state, set_rng, seed_all
    saved_rng = rng_state()
    saved = {k: getattr(agent, k) for k in
             ('env', 'gmaps', 'scanvp_cands', 'store_obs', 'logs', 'results', 'full_results')}
    saved_dynamic = {k: getattr(agent, k) for k in ('feedback', 'loss', 'losses') if hasattr(agent, k)}
    modes = {n: m.training for n, m in agent.vln_bert.named_modules()}
    try:
        seed_all(917)  # Comparable stochastic WM evaluation; training RNG is restored below.
        agent.env = dev_env
        agent.scanvp_cands = {}
        agent.store_obs = {}
        agent.logs = defaultdict(list)
        with torch.no_grad():
            agent.test(use_dropout=False, feedback='argmax', extended_memory=True, iters=None)
        predictions = agent.get_results()
        require({p['instr_id'] for p in predictions} == set(dev_env.data),
                'Full train_dev navigation coverage failed')
        metrics, _ = dev_env.eval_metrics(predictions)
        require(all(math.isfinite(float(metrics[k])) for k in ('sr', 'spl')), 'Nonfinite dev metrics')
        return {k: float(v) for k, v in metrics.items() if np.isscalar(v)}, len(predictions)
    finally:
        for k, v in saved.items():
            setattr(agent, k, v)
        for k in ('feedback', 'loss', 'losses'):
            if k in saved_dynamic:
                setattr(agent, k, saved_dynamic[k])
            elif hasattr(agent, k):
                delattr(agent, k)
        for n, m in agent.vln_bert.named_modules():
            m.training = modes[n]
        set_rng(saved_rng)


def select_split(items, tours, scans):
    scans = set(scans)
    annotations = [x for x in items if x['scan'] in scans]
    selected_tours = {s: copy.deepcopy(tours[s]) for s in tours if s in scans}
    ids = [x['instr_id'] for x in annotations]
    tour_ids = [i for ts in selected_tours.values() for t in ts for i in t]
    membership = set(tour_ids)
    require(set(selected_tours) == scans, 'Missing split scans in author tours')
    require(len(ids) == len(set(ids)), 'Duplicate annotation instruction IDs')
    require(membership <= set(ids), 'Tour/annotation instruction coverage mismatch')
    selected = [x for x in annotations if x['instr_id'] in membership]
    require(membership == {x['instr_id'] for x in selected}, 'Effective author-tour coverage mismatch')
    scan_by_id = {x['instr_id']: x['scan'] for x in selected}
    require(all(scan_by_id[i] == s for s, ts in selected_tours.items() for t in ts for i in t),
            'Instruction appears in a different scan tour')
    return selected, selected_tours


def host_volume_identity(path, expected_source, expected=None):
    from drama_mount_guard import backup_mount_identity
    p = Path(path)
    require(p.is_dir() and not p.is_symlink(), 'Host volume absent or symlinked')
    ident = backup_mount_identity(p, backend='filesystem')
    require(Path(ident['mount_point']) == p.resolve(), 'Host volume fell back to parent Linux filesystem')
    require(ident['filesystem'] in {'9p', 'drvfs'}, 'Host volume must be an explicit Windows mounted drive')
    require(ident['source'].casefold() == expected_source.casefold(), 'Wrong host volume mount source')
    if expected is not None:
        require(ident == expected, 'Host volume mount identity changed')
    return ident


def verify_pretraining_assets(metadata, assets, args, cfg):
    contract = metadata['identity']['contract']
    previous = contract['config']
    bound = contract['cache_inputs']['inputs']
    for field, current in [('feature_file', args.img_ft_file),
                           ('annotations', str(Path(args.anno_dir)/'R2R_train_enc.json')),
                           ('tours', cfg['tours']), ('split_manifest', cfg['split_manifest'])]:
        original = str(Path(previous[field]).resolve())
        require(assets[str(Path(current).resolve())] == bound[original],
                'IR2R input differs from pretrained encoder cache source: ' + field)
    for original, expected in bound.items():
        if original.endswith('_connectivity.json'):
            current = str((Path(args.connectivity_dir)/Path(original).name).resolve())
            require(assets.get(current) == expected, 'Connectivity differs from pretrained cache source')


def author_args(cfg, architecture):
    from nav_src.iter_nav.parser import parse_args
    old = sys.argv
    argv = ['drama_ir2r', '--root_dir', cfg['datasets'], '--output_dir', cfg['run_dir'],
            '--batch_size', str(cfg.get('batch_size', 1)), '--seed', str(cfg.get('seed', 0)),
            '--iters', str(cfg.get('minimum_updates', 20000)), '--lr', str(cfg.get('lr', 1e-5)),
            '--mastermind', '--multimodal_history', '--redundant_view', '--teacher_aug',
            '--exp_bw', '--share_graph', '--teacher_teleport', '--include_neighbours',
            '--env_memory_drop_replace', '--fix_lang_embedding', '--fix_pano_embedding']
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old
    args.bert_ckpt_file = None  # All raw model keys are loaded strictly from verified pretraining artifacts below.
    args.initialize_drama_world = True
    args.world_model_backend = 'drama_mamba2'
    args.freeze_encoders = True
    for key, default in [('d_model', 512), ('layers', 2), ('d_state', 16), ('headdim', 64), ('dropout', 0.1)]:
        setattr(args, 'drama_' + key, architecture.get(key, default))
    return args


def build_environments(args, cfg):
    from nav_src.envs.r2r.data_utils import construct_instrs
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from utils.data import ImageFeaturesDB
    split = json.loads(Path(cfg['split_manifest']).read_text())
    require(len(split['train_fit']) == 49 and len(split['train_dev']) == 12 and
            not set(split['train_fit']) & set(split['train_dev']), 'Unexpected frozen 49/12 split')
    instructions = construct_instrs(args.anno_dir, 'r2r', ['train'], 'bert', args.max_instr_len)
    tours = json.loads(Path(cfg['tours']).read_text())['train']
    envs, coverage = [], {}
    for name in ('train_fit', 'train_dev'):
        data, subset_tours = select_split(instructions, tours, split[name])
        effective = {x['instr_id'] for x in data}
        annotations = {x['instr_id'] for x in instructions if x['scan'] in set(split[name])}
        coverage[name] = {'scope': 'all original author tours within scan split',
            'annotation_instructions': len(annotations), 'effective_instructions': len(effective),
            'excluded_instr_ids': sorted(annotations-effective), 'excluded_count': len(annotations-effective),
            'exclusion_reason': 'annotation instruction ID absent from official author tour membership; no synthetic tours added',
            'tour_count': sum(len(ts) for ts in subset_tours.values())}
        # Separate feature DBs ensure evaluation cannot mutate the training cache.
        db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
        envs.append(IR2RNavBatch(db, data, args.connectivity_dir, subset_tours, num_robots=1,
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, seed=args.seed, name=name))
    require([coverage[k]['effective_instructions'] for k in ('train_fit', 'train_dev')] == [11136, 2889],
            'Frozen author-tour protocol counts differ from 11136 fit / 2889 dev')
    return envs, coverage


def check_runtime(agent, train_env):
    require(AGENT_FIELDS <= set(vars(agent)) <= AGENT_FIELDS | {'feedback', 'loss', 'losses'},
            'Unknown/missing persistent agent fields')
    require(ENV_FIELDS <= set(vars(train_env)) <= ENV_FIELDS | {'batch', 'tour_batch_probe'},
            'Unknown/missing persistent environment fields')
    require(agent.env is train_env and train_env.num_robots == 1 and agent.rank == 0,
            'Unsupported environment/robot/rank')
    require(agent.models is agent.vln_bert and agent.optimizers is agent.vln_bert_optimizer,
            'Model/optimizer aliases differ')
    require(set(vars(train_env.env)) == {'feat_db', 'image_w', 'image_h', 'vfov', 'sims'},
            'Unknown native simulator wrapper state')
    require(set(vars(train_env.env.feat_db)) == {'img_feat_size', 'img_ft_file', '_feature_store'},
            'Unknown feature adapter state')
    require(type(agent.criterion) is torch.nn.CrossEntropyLoss and agent.criterion.weight is None,
            'Unexpected stateful criterion')


def train(cfg, resume=False, stop_after=None):
    from drama_pretrain import (CompactStore, load_selected_weights, sha, json_sha, write_json,
        code_manifest, atomic_torch, verified_copy, seed_all, space_guard, verify_storage_mount)
    from drama_mount_guard import backup_mount_identity
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from vlnbert.drama_mamba2 import freeze_encoders
    require(cfg.get('backup_backend') == 'filesystem', 'Explicit filesystem backup selection required')
    require(cfg.get('min_free_gib', 0) >= 12, 'Host reserve must be at least 12 GiB')
    require(all(isinstance(cfg.get(k, d), int) and cfg.get(k, d) > 0
                for k, d in [('minimum_updates', 20000), ('eval_every', 500), ('save_every', 100)]),
            'Update/evaluation/archive thresholds must be positive integers')
    require(cfg.get('batch_size', 1) in (1, 2), 'Only separately identified batch 1 or 2 runs supported')
    require(os.environ.get('PYTHONHASHSEED') == '0', 'Run with PYTHONHASHSEED=0 for deterministic graph ordering')
    configure_numerics(cfg)
    host = host_volume_identity(cfg['host_volume'], cfg['host_volume_source'])
    storage = verify_storage_mount(cfg)
    run, backup = Path(cfg['run_dir']), Path(cfg['backup_dir'])
    require(run.resolve() != backup.resolve() and not run.resolve().is_relative_to(backup.resolve()) and
            not backup.resolve().is_relative_to(run.resolve()), 'Backup must be separate and nonnested')
    require(not run.is_symlink() and not backup.is_symlink(), 'Symlink run/backup rejected')
    if resume:
        require((run/'identity.json').is_file() and (backup/'identity.json').is_file(), 'Strict resume requires identities')
    else:
        require(not run.exists() and not backup.exists(), 'Fresh experiment refuses existing paths')
        space_guard([run.parent, backup.parent, cfg['host_volume']], cfg['min_free_gib'])
    def guard(additional=0):
        assert_runtime_numerics(cfg)
        host_volume_identity(cfg['host_volume'], cfg['host_volume_source'], host)
        require(verify_storage_mount(cfg) == storage, 'Dedicated Linux storage identity changed')
        space_guard([run, backup, cfg['host_volume']], cfg['min_free_gib'], additional=additional)
    completion = json.loads((Path(cfg['pretrain_run'])/'completion.json').read_text())
    require(completion.get('complete') is True, 'Full world-model pretraining must complete before IR2R')
    selected, metadata = load_selected_weights(cfg['pretrain_run'], cfg['pretrain_backup'], which='best')
    require(any(k.startswith('prior.layers.0.') for k in selected), 'Selected checkpoint is not Drama Mamba2')
    architecture = metadata['identity']['contract']['architecture']
    require(sha(cfg['split_manifest']) == metadata['identity']['contract']['split_manifest_sha256'],
            'Pretrain/IR2R frozen split differs')
    if not resume:
        run.mkdir(); backup.mkdir()
    seed_all(cfg.get('seed', 0))
    args = author_args(cfg, architecture)
    (train_env, dev_env), coverage = build_environments(args, cfg)
    agent = GMapNavAgent(args, train_env, rank=0)
    model = agent.vln_bert.vln_bert
    model.load_state_dict(selected, strict=True); del selected
    for p in model.parameters():
        p.requires_grad_(True)
    freeze_encoders(model)
    trainable, changing = parameter_partition(model)
    named = dict(model.named_parameters())
    opt_groups = [{'params': [p for n, p in named.items() if n in trainable and not getattr(p, '_no_weight_decay', False)],
                   'weight_decay': cfg.get('weight_decay', 0.01)},
                  {'params': [p for n, p in named.items() if n in trainable and getattr(p, '_no_weight_decay', False)],
                   'weight_decay': 0.0}]
    agent.vln_bert_optimizer = torch.optim.AdamW(opt_groups, lr=args.lr)
    agent.optimizers = agent.vln_bert_optimizer
    agent.vln_bert_optimizer.register_step_pre_hook(
        lambda optimizer, positional, keywords: finite_step_guard(optimizer, agent.loss))
    agent.vln_bert.train()
    check_runtime(agent, train_env)
    files = [Path(args.img_ft_file), Path(args.anno_dir)/'R2R_train_enc.json', Path(cfg['tours']), Path(cfg['split_manifest'])]
    files += sorted(Path(args.connectivity_dir).glob('*_connectivity.json'))
    assets = {str(p.resolve()): sha(p) for p in files}
    verify_pretraining_assets(metadata, assets, args, cfg)
    contract = {'schema': SCHEMA, 'config': cfg, 'source_files': code_manifest(),
        'assets': assets, 'pretraining': metadata,
        'trainable_names': sorted(trainable), 'delta_names': sorted(changing),
        'environment': {e.name: static_environment(e) for e in (train_env, dev_env)},
        'selection': 'full author tours within train_dev: SR then SPL, lexicographic; no val_unseen accesses',
        'tour_membership': coverage,
        'minimum_updates': cfg.get('minimum_updates', 20000),
        'boundary_schedule': 'thresholds deferred to first complete tour/update boundary',
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED')}}
    contract['optimizer'] = {'class': 'torch.optim.AdamW', 'lr': args.lr,
        'weight_decay': cfg.get('weight_decay', 0.01), 'no_weight_decay_group': 0.0,
        'gradient_clip': 40.0, 'finite_loss_and_gradients_checked_before_step': True}
    if resume:
        identity = json.loads((run/'identity.json').read_text())
        require(identity['contract'] == contract and json.loads((backup/'identity.json').read_text()) == identity,
                'Strict resume contract differs')
        require(identity['host_mount'] == host, 'Host mount changed since experiment creation')
        require(identity['storage_mount'] == storage, 'Dedicated storage mount changed since experiment creation')
    else:
        frozen = {k: pack(v) for k, v in model.state_dict().items() if k not in changing}
        frozen_bytes = sum(x.numel()*x.element_size() for x in frozen.values())
        guard(2*frozen_bytes)
        atomic_torch(run/'frozen-base.pt', frozen); del frozen
        base_sha = verified_copy(run/'frozen-base.pt', backup/'frozen-base.pt')
        identity = {'contract': contract, 'frozen_base_sha256': base_sha,
            'backup_mount': backup_mount_identity(backup, backend='filesystem'),
            'host_mount': host, 'storage_mount': storage}
        write_json(run/'identity.json', identity); write_json(backup/'identity.json', identity)
    for path in (run/'frozen-base.pt', backup/'frozen-base.pt'):
        require(sha(path) == identity['frozen_base_sha256'], 'Frozen base checksum mismatch')
    frozen = torch.load(run/'frozen-base.pt', map_location='cpu', weights_only=True)
    identity_sha = json_sha(identity)
    store = CompactStore(run/'checkpoints', backup/'checkpoints', cfg['host_volume'],
                         cfg['min_free_gib'], identity['backup_mount'])
    # Every publish and prune also validates the real Windows host mount.
    original_check = store.backup_check
    def storage_check():
        guard(); original_check()
    store.backup_check = storage_check
    progress = {'completed_updates': 0, 'last_full_step': 0, 'next_eval': cfg.get('eval_every', 500),
                'best_metrics': None, 'best_step': None, 'history': [], 'elapsed_seconds': 0.0}
    stop = [False]
    def signal_stop(signum, frame):
        stop[0] = True
    previous_signals = {s: signal.signal(s, signal_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    clock = time.monotonic()
    elapsed_at_start = 0.
    def report(phase, **extra):
        data = {**{k: v for k, v in progress.items() if k != 'history'},
            'evaluation_count': len(progress['history']), 'schema': SCHEMA, 'phase': phase, 'strict_resume_supported': True,
            'minimum_updates': cfg.get('minimum_updates', 20000),
            'elapsed_seconds': elapsed_at_start + time.monotonic()-clock,
            'complete_tour_boundary': complete_tour(train_env),
            'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2, **extra}
        write_json(run/'progress.json', data)
        print('IR2R_PROGRESS', json.dumps(data), flush=True)
    def save(best=False):
        progress['last_full_step'] = progress['completed_updates']
        progress['elapsed_seconds'] = elapsed_at_start + time.monotonic()-clock
        state = runtime_state(agent, progress, identity_sha, dev_env)
        # Reserve space for both new snapshot copies before old verified copies can be pruned.
        def tensor_bytes(x):
            if torch.is_tensor(x): return x.numel()*x.element_size()
            if isinstance(x, dict): return sum(tensor_bytes(v) for v in x.values())
            if isinstance(x, (list, tuple)): return sum(map(tensor_bytes, x))
            return 0
        guard(2*tensor_bytes(state) + 32*1024**2)
        checkpoint_id = store.save(state, {'kind': 'complete_ir2r_trainable_delta_state',
            'frozen_base_sha256': identity['frozen_base_sha256']}, step=state['step'], is_best=best)
        report('checkpoint_verified', checkpoint_id=checkpoint_id, backup_verified=True)
    try:
        with store.lock():
            if resume:
                state, _, manifest = store.restore(which='latest')
                require(manifest['step'] == state['step'], 'Manifest/runtime step mismatch')
                progress = restore_runtime(agent, state, identity_sha, frozen, dev_env)
                elapsed_at_start = progress['elapsed_seconds']
                report('resumed')
            else:
                save()
            del frozen
            while True:
                step = progress['completed_updates']
                reached = step >= cfg.get('minimum_updates', 20000)
                stopping = stop[0] or (stop_after is not None and step >= stop_after)
                if complete_tour(train_env) and (reached or stopping):
                    report('complete' if reached else 'stopped_at_complete_tour_boundary')
                    write_json(run/'completion.json', {'complete': reached, 'actual_updates': step,
                        'minimum_updates': cfg.get('minimum_updates', 20000), 'last_full_step': progress['last_full_step'],
                        'best_step': progress['best_step'], 'best_metrics': progress['best_metrics'],
                        'strict_resume_supported': True})
                    break
                guard()
                agent.train(1, feedback='sample', extended_memory=True)
                require(torch.is_tensor(agent.loss) and bool(torch.isfinite(agent.loss).all()), 'Nonfinite IR2R loss')
                loss = float(agent.loss.detach())
                progress['completed_updates'] += 1
                step = progress['completed_updates']
                boundary = complete_tour(train_env)
                report('training', loss=loss)
                if not boundary:
                    continue
                canonical_boundary(agent)
                final = step >= cfg.get('minimum_updates', 20000)
                do_eval = step >= progress['next_eval'] or final
                is_best = False
                if do_eval:
                    report('evaluating_train_dev')
                    metrics, count = evaluate_navigation(agent, dev_env)
                    entry = {'actual_update': step, 'scheduled_update': progress['next_eval'],
                             'split': 'train_dev', 'scope': 'full author tours within train_dev',
                             'instructions': count, 'annotation_instructions': coverage['train_dev']['annotation_instructions'],
                             'excluded_instr_ids': coverage['train_dev']['excluded_instr_ids'],
                             'excluded_count': coverage['train_dev']['excluded_count'], **metrics}
                    progress['history'].append(entry)
                    old = progress['best_metrics']
                    is_best = old is None or (metrics['sr'], metrics['spl']) > (old['sr'], old['spl'])
                    if is_best:
                        progress['best_metrics'] = metrics; progress['best_step'] = step
                    progress['next_eval'] = (step // cfg.get('eval_every', 500) + 1)*cfg.get('eval_every', 500)
                    write_json(run/'dev-history.json', progress['history'])
                if do_eval or step-progress['last_full_step'] >= cfg.get('save_every', 100) or stop[0] or (stop_after is not None and step >= stop_after):
                    save(is_best)
    finally:
        for s, handler in previous_signals.items():
            signal.signal(s, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--require-resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Engineering stop threshold; still completes current tour/update')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    os.chdir(REPO)
    train(cfg, resume=a.require_resume, stop_after=a.stop_after)


if __name__ == '__main__':
    main()
