"""CPU continuation equivalence; these do not claim MatterSim/GPU verification."""
from collections import defaultdict
import copy
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import drama_ir2r as ir
from drama_pretrain import rng_state, seed_all


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class Features:
    def __init__(self):
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        return self._feature_store.setdefault(scan+'_'+viewpoint, np.array([len(viewpoint)], dtype=np.float32))


class ToyEnv:
    def __init__(self):
        self.full_data = [[['a', 'b', 'c', 'd']], [['e', 'f']]]
        self.ix = 0
        self.tour_batch = None
        self.tour_ended = np.ones((1, 1), dtype=bool)
        self.batch_size = self.num_robots = 1
        self.buffered_state_dict = {}
        self.sent_embed_store = {}
        self.extra_obs = None
        self.env = SimpleNamespace(feat_db=Features())

    def take(self):
        if ir.complete_tour(self):
            if self.ix == len(self.full_data):
                random.shuffle(self.full_data)
                self.ix = 0
            self.tour_batch = [self.full_data[self.ix][0:1]]
            self.ix += 1
            self.tour_batch_probe = np.zeros((1, 1), dtype=np.int64)
        else:
            self.tour_batch_probe += 1
        item = self.tour_batch[0][0][self.tour_batch_probe[0, 0]]
        self.batch = [[{'instr_id': item}]]
        self.env.feat_db.get_image_feature('scan', item)
        self.buffered_state_dict[item] = np.array([random.random(), np.random.rand()], dtype=np.float32)
        return item


class Raw(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3)
        self.alias = self.frozen  # Shared alias must never leak into trainable delta.
        self.head = torch.nn.Linear(3, 2)
        self.register_buffer('counter', torch.tensor(0))
        self.frozen.requires_grad_(False)

    def forward(self, x):
        self.counter += 1
        return self.head(self.frozen(x))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vln_bert = Raw()


def make_agent():
    wrapper = Wrapper()
    optimizer = torch.optim.AdamW([p for p in wrapper.vln_bert.parameters() if p.requires_grad], lr=.01)
    return SimpleNamespace(vln_bert=wrapper, vln_bert_optimizer=optimizer, env=ToyEnv(),
        gmaps=[], store_obs={}, scanvp_cands={}, logs=defaultdict(list), losses=[],
        feedback='sample', loss=0, results={}, full_results=[])


def update(a, progress):
    a.vln_bert_optimizer.zero_grad(set_to_none=True)
    loss = 0
    for mode in ('teacher', 'sample'):
        item = a.env.take()
        a.scanvp_cands[item] = {'next': len(item)}
        x = torch.rand(3) + random.random() + np.random.rand()
        loss = loss + a.vln_bert.vln_bert(x).square().sum()
    loss.backward()
    a.vln_bert_optimizer.step()
    a.loss = loss
    a.logs['IL_loss'].append(float(loss.detach()))
    progress['completed_updates'] += 1


def test_full_state_delta_resume_matches_uninterrupted_cpu_updates():
    seed_all(137)
    a = make_agent()
    progress = {'completed_updates': 0, 'last_full_step': 0, 'best_metrics': None,
                'history': [], 'next_eval': 500, 'elapsed_seconds': 0.}
    update(a, progress); update(a, progress)
    assert ir.complete_tour(a.env)
    _, changing = ir.parameter_partition(a.vln_bert.vln_bert)
    assert changing == {'head.weight', 'head.bias', 'counter'}
    frozen = {k: v.clone() for k, v in a.vln_bert.vln_bert.state_dict().items() if k not in changing}
    state = ir.runtime_state(a, progress, 'identity')
    # Production weights_only roundtrip, including NumPy sampler/cache and Adam state.
    stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    for _ in range(7):
        update(a, progress)
    expected = (copy.deepcopy(a.vln_bert.state_dict()), copy.deepcopy(a.vln_bert_optimizer.state_dict()),
                ir.pack(a.env.full_data), a.env.ix, copy.deepcopy(a.env.buffered_state_dict), rng_state())
    b = make_agent()  # Consumes RNG and has different weights/native initial state.
    restored = ir.restore_runtime(b, saved, 'identity', frozen)
    assert restored['completed_updates'] == 2
    for _ in range(7):
        update(b, restored)
    actual = (b.vln_bert.state_dict(), b.vln_bert_optimizer.state_dict(),
              ir.pack(b.env.full_data), b.env.ix, b.env.buffered_state_dict, rng_state())
    assert equal(expected, actual)
    assert restored['completed_updates'] == progress['completed_updates'] == 9


def test_partial_tour_and_wrong_identity_are_rejected():
    seed_all(0); a = make_agent(); p = {'completed_updates': 0}
    update(a, p)
    with pytest.raises(RuntimeError, match='Partial tour'):
        ir.runtime_state(a, p, 'id')
    update(a, p); state = ir.runtime_state(a, p, 'id')
    with pytest.raises(RuntimeError, match='identity'):
        ir.restore_runtime(a, state, 'wrong', {})
    state['step'] = 99
    with pytest.raises(RuntimeError, match='step disagreement'):
        ir.restore_runtime(a, state, 'id', {})


def test_live_autograd_and_unsupported_native_objects_rejected():
    with pytest.raises(RuntimeError, match='autograd'):
        ir.pack(torch.tensor(1., requires_grad=True)*2)
    with pytest.raises(RuntimeError, match='Unsupported'):
        ir.pack(object())


def test_finite_loss_bad_gradient_cannot_mutate_parameter_or_adam_state():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = torch.optim.AdamW([p], lr=.1)
    opt.register_step_pre_hook(lambda optimizer, args, kwargs:
                              ir.finite_step_guard(optimizer, torch.tensor(1.)))
    p.grad = torch.tensor([1., float('nan')])
    before = p.detach().clone()
    with pytest.raises(RuntimeError, match='Nonfinite gradient'):
        opt.step()
    assert torch.equal(p, before) and not opt.state
    p.grad = torch.tensor([1., 2.])
    opt.step()
    assert not torch.equal(p, before) and opt.state


def test_evaluation_restores_training_state_and_rng_on_success_and_failure():
    seed_all(31); a = make_agent()
    dev = SimpleNamespace(data={'dev0': {}}, eval_metrics=lambda p: ({'sr': 50., 'spl': 40.}, {}))
    before_rng = rng_state(); before_env = a.env; before_cache = a.scanvp_cands
    def test(**kwargs):
        random.random(); np.random.rand(); torch.rand(3)
        a.scanvp_cands['dev-only'] = {}
        a.gmaps = ['dev']; a.losses = ['dev']; a.feedback = 'argmax'
        a.vln_bert.eval()
    a.test = test
    a.get_results = lambda: [{'instr_id': 'dev0'}]
    metrics, count = ir.evaluate_navigation(a, dev)
    assert metrics == {'sr': 50., 'spl': 40.} and count == 1
    assert a.env is before_env and a.scanvp_cands is before_cache and a.scanvp_cands == {}
    assert a.vln_bert.training and a.losses == [] and a.feedback == 'sample'
    assert equal(before_rng, rng_state())
    a.get_results = lambda: [{'instr_id': 'wrong'}]
    with pytest.raises(RuntimeError, match='coverage'):
        ir.evaluate_navigation(a, dev)
    assert a.env is before_env and equal(before_rng, rng_state())


def test_split_uses_complete_author_tours_and_rejects_missing_annotations():
    rows = [{'scan': 'fit', 'instr_id': 'f'}, {'scan': 'dev', 'instr_id': 'd'}]
    tours = {'fit': [['f']], 'dev': [['d']]}
    selected, groups = ir.select_split(rows, tours, ['fit'])
    assert selected == rows[:1] and groups == {'fit': [['f']]}
    with pytest.raises(RuntimeError, match='coverage'):
        ir.select_split(rows, {'fit': [['unknown']]}, ['fit'])
    rows.append({'scan': 'fit', 'instr_id': 'f_omitted_fourth_instruction'})
    selected, groups = ir.select_split(rows, tours, ['fit'])
    assert selected == rows[:1]
    assert groups == {'fit': [['f']]}  # No singleton/synthetic tour changes.


def test_host_guard_rejects_parent_fallback_and_changed_mount(tmp_path, monkeypatch):
    import drama_mount_guard
    identity = {'mount_point': '/', 'filesystem': 'ext4', 'source': '/dev/sdd'}
    monkeypatch.setattr(drama_mount_guard, 'backup_mount_identity', lambda *a, **k: identity)
    with pytest.raises(RuntimeError, match='parent Linux'):
        ir.host_volume_identity(tmp_path, 'E:\\')
    identity.update(mount_point=str(tmp_path), filesystem='9p', source='E:\\')
    assert ir.host_volume_identity(tmp_path, 'E:\\') == identity
    with pytest.raises(RuntimeError, match='Wrong host volume'):
        ir.host_volume_identity(tmp_path, 'C:\\')
    with pytest.raises(RuntimeError, match='identity changed'):
        ir.host_volume_identity(tmp_path, 'E:\\', {**identity, 'source': 'D:\\'})


def test_feature_semantics_must_match_pretraining_cache_even_with_new_paths(tmp_path):
    old = tmp_path/'old'; new = tmp_path/'new'
    previous = {'feature_file': str(old/'features.hdf5'), 'annotations': str(old/'R2R_train_enc.json'),
                'tours': str(old/'tours.json'), 'split_manifest': str(old/'split.json')}
    inputs = {p: 'sha-'+Path(p).name for p in previous.values()}
    metadata = {'identity': {'contract': {'config': previous, 'cache_inputs': {'inputs': inputs}}}}
    args = SimpleNamespace(img_ft_file=str(new/'features.hdf5'), anno_dir=str(new), connectivity_dir=str(new/'connectivity'))
    cfg = {'tours': str(new/'tours.json'), 'split_manifest': str(new/'split.json')}
    assets = {str(new/Path(p).name): s for p, s in inputs.items()}
    ir.verify_pretraining_assets(metadata, assets, args, cfg)
    assets[args.img_ft_file] = 'different-features'
    with pytest.raises(RuntimeError, match='feature_file'):
        ir.verify_pretraining_assets(metadata, assets, args, cfg)
