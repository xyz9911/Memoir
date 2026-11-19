#!/usr/bin/env python3
"""Frozen Memoir encoders and Mamba-2 world-model pretraining on all author train rows."""
import argparse
from array import array
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import drama_pretrain as pre
from colab_full_storage import FullStore


REPO = Path(__file__).resolve().parents[1]
AUTHOR_CONFIG = REPO / "pretrain/config/r2r_rssm_pretrain.json"
DEFAULT_CONFIG = REPO / "configs/colab_full_pretrain.json"
BASE_FILE = "pretrained/grduet_tssm_dropout.pt"
SCHEMA = "memoir-full-pretraining-v1"


def author_paths():
    data = json.loads(AUTHOR_CONFIG.read_text())["train_datasets"]["R2R"]
    fields = ("train_traj_file", "train_aug_traj_file", "img_ft_file", "scanvp_cands_file", "connectivity_dir")
    return {field: Path(data[field]).relative_to("datasets/R2R").as_posix() for field in fields}


class TrainingRows:
    """Index the author's ordered union without materializing a large JSONL as Python objects."""
    def __init__(self, original, augmented):
        self.original = []
        for item in json.loads(Path(original).read_text()):
            if len(item["instr_encodings"]) != len(item["instructions"]):
                raise ValueError(f"Instruction count differs for {item['path_id']}")
            for index, tokens in enumerate(item["instr_encodings"]):
                self.original.append({"scan": item["scan"], "path": item["path"], "heading": item["heading"],
                                      "instr_encoding": tokens, "instr_id": f"{item['path_id']}_{index}"})
        self.augmented = Path(augmented).open("rb")
        self.offsets = array("Q")
        offset = 0
        for line in self.augmented:
            self.offsets.append(offset)
            offset += len(line)
        if not self.offsets or not self.original:
            raise ValueError("Both author trajectory sources must be populated.")

    def __len__(self):
        return len(self.original) + len(self.offsets)

    def __getitem__(self, idx):
        if idx < len(self.original):
            return self.original[idx]
        self.augmented.seek(self.offsets[idx - len(self.original)])
        return json.loads(self.augmented.readline())


class FrozenInputs:
    def __init__(self, model, assets, rows, pano_batch_size):
        from pretrain.data.common import get_view_rel_angles
        from pretrain.data.dataset import R2RTextPathData
        from utils.data import load_nav_graphs, new_simulator

        self.model = model
        self.rows = rows
        self.device = next(model.parameters()).device
        paths = author_paths()
        conn = assets / paths["connectivity_dir"]
        self.graphs = load_nav_graphs(str(conn), (conn / "scans.txt").read_text().split())
        self.sim = new_simulator(str(conn))
        self.features = h5py.File(assets / paths["img_ft_file"], "r")
        self.candidates = json.loads((assets / paths["scanvp_cands_file"]).read_text())
        # The exact candidate order, panorama layout and relative-angle code used in author RSSM.
        self.db = object.__new__(R2RTextPathData)
        self.db.scanvp_cands = self.candidates
        self.db.image_feat_size = 512
        self.db.angle_feat_size = 4
        self.db.all_point_rel_angles = [get_view_rel_angles(i) for i in range(36)]
        self.db.get_scanvp_feature = self.feature
        # Fixed batches make cached and recomputed panorama embeddings identical across restarts.
        # Group strictly within a scan. Initial headings are horizontal, and subsequent views come
        # from the author's incoming edge at the predecessor panorama.
        by_viewpoint = defaultdict(lambda: set(range(12, 24)))
        for key, edges in self.candidates.items():
            scan, vp = key.split("_")
            by_viewpoint[scan, vp]
            for neighbor, edge in edges.items():
                by_viewpoint[scan, neighbor].add(edge[0])
        by_scan = defaultdict(list)
        for (scan, vp), views in sorted(by_viewpoint.items()):
            by_scan[scan].extend((scan, vp, idx) for idx in sorted(views))
        self.pano_groups = []
        self.group_id = {}
        self.visual_cache = {}
        for items in by_scan.values():
            for start in range(0, len(items), pano_batch_size):
                group = items[start:start + pano_batch_size]
                self.group_id.update(dict.fromkeys(group, len(self.pano_groups)))
                self.pano_groups.append(group)

    @lru_cache(maxsize=2048)
    def feature(self, scan, vp):
        return self.features[f"{scan}_{vp}"][:, :512].astype(np.float32)

    @lru_cache(maxsize=16384)
    def start_view(self, scan, vp, heading):
        self.sim.newEpisode([scan], [vp], [float(heading)], [0.0])
        return self.sim.getState()[0].viewIndex

    @lru_cache(maxsize=2048)
    def distances(self, scan, goal):
        return nx.single_source_dijkstra_path_length(self.graphs[scan], goal, weight="weight")

    @torch.no_grad()
    def batch(self, indices):
        if any(m.training for m in (self.model.embeddings, self.model.lang_encoder, self.model.img_embeddings)):
            raise RuntimeError("All encoders must stay frozen in eval mode")
        rows = [self.rows[i] for i in indices]
        tokens = [torch.as_tensor(row["instr_encoding"], dtype=torch.long) for row in rows]
        ids = pad_sequence(tokens, batch_first=True).to(self.device)
        lengths = torch.tensor([len(t) for t in tokens], device=self.device)
        text_mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
        lang = self.model.forward_text(ids, text_mask)
        length = max(len(row["path"]) for row in rows)
        mask = torch.zeros(len(rows), length, dtype=torch.bool)
        distance = torch.zeros(len(rows), length)
        keys = []
        for i, row in enumerate(rows):
            scan, path = row["scan"], row["path"]
            if len(path) < 2:
                raise ValueError(f"Invalid author RSSM trajectory {row.get('instr_id')}")
            goal_dist = self.distances(scan, path[-1])
            view = self.start_view(scan, path[0], row["heading"])
            example = []
            for t, vp in enumerate(path):
                example.append((scan, vp, view))
                nxt = path[min(t + 1, len(path) - 1)]
                distance[i, t] = goal_dist[nxt]
                if t + 1 < len(path):
                    view = self.candidates[f"{scan}_{vp}"][nxt][0]
            mask[i, :len(path)] = True
            keys.append(example)

        missing = dict.fromkeys(self.group_id[k] for example in keys for k in example if k not in self.visual_cache)
        for group_id in missing:
            group = self.pano_groups[group_id]
            img, loc, typ, lens, _ = self.db.get_batch_pano_fts(*zip(*group))
            pano, valid = self.model.forward_panorama_per_step(img, None, loc, typ, lens, None)
            mean = ((pano * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)).cpu()
            self.visual_cache.update(zip(group, mean.unbind(0)))
        visual = pad_sequence([torch.stack([self.visual_cache[k] for k in example]) for example in keys],
                              batch_first=True)
        return {"language": lang, "text_mask": text_mask, "visual": visual.to(self.device),
                "mask": mask.to(self.device), "distance": distance.to(self.device)}


def backward_update(model, batch, microbatch, horizon, weights):
    """All microbatches share the 32-trajectory negative pool and per-item normalization."""
    mask, visual = batch["mask"], batch["visual"]
    counts = mask.long().sum(1).tolist()
    transitions = sum(count - 1 for count in counts)
    overs = sum(min(t - 1, horizon - 1) for count in counts for t in range(1, count))
    normalizers = (transitions, 2 * transitions + overs, overs)
    pool = F.normalize(model.obs_transform(visual[mask]), dim=-1)
    leaf_pool = pool.detach().requires_grad_()
    first = 0
    total = 0.0
    parts = {}
    for pos in range(0, len(counts), microbatch):
        sl = slice(pos, pos + microbatch)
        mb = {key: value[sl] for key, value in batch.items()}
        loss, values = pre.wm_loss(model, mb, horizon, **weights, contrastive_pool=leaf_pool,
                                   first_label=first, normalizers=normalizers)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite WM loss")
        loss.backward()
        total += loss.detach().item()
        for k, v in values.items():
            parts[k] = parts.get(k, 0.0) + v.detach().item()
        first += sum(counts[sl])
    pool.backward(leaf_pool.grad)
    return total, parts


def assets_manifest(root, cfg, paths):
    manifest = json.loads((REPO / "configs/colab-full-assets.json").read_text())
    if manifest["revision"] != cfg["author_revision"]:
        raise RuntimeError("Author dataset revision differs")
    files = manifest["files"]
    required = dict(cfg["required_sha256"])
    prefix = paths["connectivity_dir"] + "/"
    required.update({name: meta["sha256"] for name, meta in files.items() if name.startswith(prefix)})
    if not required.get(prefix + "scans.txt") or any(files[name]["sha256"] != expected for name, expected in required.items()):
        raise RuntimeError("Asset manifests disagree")
    result = {name: pre.sha(root / name) for name in required}
    if result != required:
        raise RuntimeError(f"Author data checksum mismatch: {[k for k, v in result.items() if v != required[k]]}")
    return result


def git_revision():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        # A transport of an archived revision sets this alongside the audited source content hashes.
        sha = os.environ.get("MEMOIR_SOURCE_SHA", "")
        if len(sha) != 40 or any(x not in "0123456789abcdef" for x in sha):
            raise RuntimeError("A source archive requires its full MEMOIR_SOURCE_SHA")
        return sha


def initialization_audit(model, base):
    original = torch.load(base, weights_only=True, mmap=True, map_location="cpu")
    if any(k.removeprefix("bert.").startswith("prior.layers.0.") for k in original):
        raise RuntimeError("Expected the author's TSSM initializer")
    copied, fresh = {}, []
    for name, value in model.state_dict().items():
        if name.startswith(pre.WM_PREFIXES):
            fresh.append(name)
            continue
        key = "bert." + name.replace("exp_encoder.", "global_encoder.", 1).replace("exp_sap_head.", "global_sap_head.", 1)
        if key in original:
            if not torch.equal(value.detach().cpu(), original[key]):
                raise RuntimeError(f"Initializer differs from the author's weights: {name}")
            copied[name] = key
        else:
            if name.startswith(("embeddings.", "lang_encoder.", "img_embeddings.")):
                raise RuntimeError(f"Missing frozen encoder weight: {name}")
            fresh.append(name)
    return {"source": BASE_FILE, "copied_parameters_and_buffers": copied, "fresh": fresh,
            "fresh_world_model_prefixes": list(pre.WM_PREFIXES), "frozen_encoders": ["embeddings", "lang_encoder", "img_embeddings"]}


def contract(cfg, assets, rows, mount, initialization):
    runtime = pre.runtime_identity()
    # Reinstalled wheels and HF caches can occupy different absolute paths on a replacement VM.
    runtime["bert_config"].pop("resolved_path")
    if runtime["mattersim"]:
        runtime["mattersim"].pop("path")
    source = pre.code_manifest()
    return {"schema": SCHEMA, "config": cfg, "source_revision": git_revision(),
            "source_files": source, "author_config_sha256": pre.sha(AUTHOR_CONFIG),
            "assets": assets, "rows": {"r2r": len(rows.original), "prevalent": len(rows.offsets)},
            "sampler": "uniform shuffled union without replacement; each input instruction or JSONL row kept",
            "architecture": {k: cfg[k] for k in ("d_model", "layers", "d_state", "headdim")},
            "initialization": initialization,
            "objective": "frozen eval-mode text/panorama; author GT paths, batched positive panoramas, KL, overshoot and next-goal-distance",
            "runtime": runtime, "backup_mount": {k: mount[k] for k in ("backend", "mount_root", "filesystem", "source")}}


def mirror_base(run, backup, frozen_names, model, store, resume):
    source, mirror = run / "frozen-base.pt", backup / "frozen-base.pt"
    if resume:
        identity = json.loads((backup / "identity.json").read_text())
        if pre.sha(mirror) != identity["frozen_base_sha256"]:
            raise RuntimeError("Frozen base checksum mismatch")
        pre.verified_copy(mirror, source)
    else:
        frozen = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in frozen_names}
        pre.space_guard([run, backup], store.min_free_gib, pre.tensor_bytes(frozen) + 32 * 2**20)
        pre.atomic_torch(source, frozen)
        store.backup_check()
        pre.verified_copy(source, mirror)
    return pre.sha(source)


def restore_weights(model, trainable, state, frozen_path):
    frozen = torch.load(frozen_path, weights_only=True, map_location="cpu")
    if set(state["trainable_state"]) != set(trainable) or set(frozen) & set(trainable):
        raise RuntimeError("Snapshot parameter partition differs")
    frozen.update(state["trainable_state"])
    model.load_state_dict(frozen, strict=True)


def finalize(model, cfg, stage, backup, identity, store):
    store.backup_check()
    file = stage / "final-model.pt"
    remote_file = backup / file.name
    # Reconstruct a missing local export from the verified remote bytes if final publication was interrupted.
    if remote_file.exists():
        pre.verified_copy(remote_file, file)
        expected = model.state_dict()
        actual = torch.load(file, weights_only=True, map_location="cpu")
        if set(expected) != set(actual) or any(not torch.equal(v.detach().cpu(), actual[k]) for k, v in expected.items()):
            raise RuntimeError("Existing final export differs from terminal training state")
    else:
        data = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        pre.space_guard((stage, backup), store.min_free_gib, pre.tensor_bytes(data) + 32 * 2**20)
        pre.atomic_torch(file, data)
    store.backup_check()
    digest = pre.verified_copy(file, remote_file)
    result = {"complete": True, "selection": "fixed-final-step", "step": cfg["steps"], "model_file": file.name,
              "model_sha256": digest, "identity_sha256": pre.json_sha(identity), "assets": identity["contract"]["assets"],
              "architecture": identity["contract"]["architecture"]}
    store.backup_check()
    pre.write_json(backup / "handoff.json", result)
    if json.loads((backup / "handoff.json").read_text()) != result:
        raise RuntimeError("Handoff readback failed")
    pre.write_json(stage / "handoff.json", result)
    print("PRETRAIN_COMPLETE", json.dumps({k: result[k] for k in ("step", "model_sha256", "identity_sha256")}), flush=True)


def run(a):
    cfg = json.loads(a.config.read_text())
    if os.environ.get("PYTHONHASHSEED") != "0" or not torch.cuda.is_available():
        raise RuntimeError("Launch on a CUDA GPU with PYTHONHASHSEED=0")
    pre.configure_runtime(cfg)
    root = a.asset_root.resolve()
    stage, backup = a.run_root.resolve() / "pretrain", a.backup_root.resolve() / "pretrain"
    # CheckpointStore locks and re-verifies the explicit persistent mount before writes/pruning.
    store = FullStore(stage / "checkpoints", backup / "checkpoints", backend=a.backup_backend,
                      min_free_gib=cfg["min_free_gib"], keep_last=1)
    with store.lock():
        existing = [p for d in (backup, stage) for p in (d / "identity.json", d / "frozen-base.pt", d / "handoff.json") if p.exists()]
        if not a.require_resume and (existing or any(store.backup_dir.joinpath("snapshots").iterdir())
                                     or any(store.local_dir.joinpath("snapshots").iterdir())):
            raise RuntimeError("Run artifacts already exist. Select strict resume or a new run.")
        if a.require_resume and not (backup / "identity.json").is_file():
            raise RuntimeError("Strict resume needs the persistent training identity.")
        paths = author_paths()
        assets = assets_manifest(root, cfg, paths)
        rows = TrainingRows(root / paths["train_traj_file"], root / paths["train_aug_traj_file"])
        if (len(rows.original), len(rows.offsets)) != (14039, 1069620):
            raise RuntimeError("Author instruction row counts differ")
        print("PRETRAIN_DATA", json.dumps({"r2r": len(rows.original), "prevalent": len(rows.offsets), "total": len(rows)}), flush=True)
        model = pre.make_model({**cfg, "init_checkpoint": str(root / BASE_FILE)})
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        initialization = initialization_audit(model, root / BASE_FILE)
        run_contract = contract(cfg, assets, rows, store.mount, initialization)
        frozen_names = model.state_dict().keys() - trainable.keys()
        digest = mirror_base(stage, backup, frozen_names, model, store, a.require_resume)
        identity = {"contract": run_contract, "frozen_base_sha256": digest, "trainable_names": sorted(trainable)}
        if a.require_resume:
            if json.loads((backup / "identity.json").read_text()) != identity:
                raise RuntimeError("Strict resume identity differs")
            if (stage / "identity.json").exists() and json.loads((stage / "identity.json").read_text()) != identity:
                raise RuntimeError("Local identity differs from mounted backup")
        else:
            store.backup_check()
            pre.write_json(backup / "identity.json", identity)
        pre.write_json(stage / "identity.json", identity)
        identity_sha = pre.json_sha(identity)
        optimizer = pre.build_optimizer(model, cfg)
        step, width = 0, cfg["microbatch"]
        if a.require_resume:
            state, _, checkpoint = store.restore()
            if state["identity_sha256"] != identity_sha or state["step"] != checkpoint["step"]:
                raise RuntimeError("Training snapshot identity or step mismatch")
            restore_weights(model, trainable, state, stage / "frozen-base.pt")
            optimizer.load_state_dict(state["optimizer"])
            step, width = state["step"], state["microbatch"]
            pre.validate_scheduler(state["scheduler"], step, cfg, optimizer)
            sampler = pre.Sampler(len(rows), state["sampler"])
            restored_rng = state["rng"]
            del state
        else:
            sampler = pre.Sampler(len(rows))
            restored_rng = None

        def save():
            state = {"schema": SCHEMA, "identity_sha256": identity_sha, "step": step, "microbatch": width,
                     "trainable_state": {k: v.detach().cpu() for k, v in trainable.items()},
                     "optimizer": pre.cpu_tree(optimizer.state_dict()), "scheduler": pre.scheduler_state(step, cfg),
                     "sampler": sampler.state(), "rng": pre.rng_state(), "mount_at_save": store.mount}
            store.save(state, {"kind": "frozen-base-plus-all-trainable-weights"}, step=step)

        if not a.require_resume:
            save()
        if a.stop_after is not None and a.stop_after <= step < cfg["steps"]:
            return
        if step < cfg["steps"]:
            data = FrozenInputs(model, root, rows, cfg["pano_batch_size"])
            if restored_rng is not None:
                pre.set_rng(restored_rng)
            model.train()
            started = time.monotonic()
            while step < cfg["steps"]:
                indices = sampler.take(cfg["batch_size"])
                batch = data.batch(indices)
                before = pre.rng_state()
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        loss, pieces = backward_update(model, batch, width, cfg["overshoot_horizon"], pre.loss_weights(cfg))
                        break
                    except torch.cuda.OutOfMemoryError:
                        if width <= 1:
                            raise
                        width = max(1, width // 2)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    pre.set_rng(before)
                    print("PRETRAIN_MICROBATCH_RETRY", width, flush=True)
                grad = torch.nn.utils.clip_grad_norm_(trainable.values(), cfg["grad_clip"], error_if_nonfinite=True)
                lr = pre.scheduled_lr(step + 1, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                step += 1
                record = {"step": step, "target": cfg["steps"], "batch": len(indices), "microbatch": width, "loss": loss,
                          "lr": lr, "grad_norm": float(grad), "seconds": time.monotonic() - started,
                          "peak_mib": torch.cuda.max_memory_allocated() / 2**20, **pieces}
                pre.write_json(stage / "progress.json", record)
                print("PRETRAIN_PROGRESS", json.dumps(record), flush=True)
                stop = a.stop_after is not None and step >= a.stop_after
                if step % cfg["save_every"] == 0 or step == cfg["steps"] or stop:
                    save()
                if stop and step < cfg["steps"]:
                    return
        finalize(model, cfg, stage, backup, identity, store)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--asset-root", type=Path, required=True, help="Author datasets/R2R directory")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--backup-root", type=Path, required=True)
    p.add_argument("--backup-backend", choices=("drive", "filesystem"), required=True)
    p.add_argument("--require-resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Commit complete training state and exit after this optimizer step")
    run(p.parse_args())
