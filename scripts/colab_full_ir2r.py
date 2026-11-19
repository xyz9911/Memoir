#!/usr/bin/env python3
"""Run frozen-encoder Mamba-2 navigation on the complete author IR2R training split."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BUDGET = {8: 10000, 4: 20000, 2: 40000}
OOM_EXIT = 86
ARCH = {"d_model": 768, "layers": 4, "d_state": 16, "headdim": 64}
OFFICIAL = {
    "annotations/R2R_train_enc.json": "7745f32c24e40b75107c11593ec768eb2b26a5ab784675346abef3c9edf46d5c",
    "tours_iVLN.json": "0ef3b8582e4389a800a4eabc37b81cc15486f86bccd748d168e0270015a49125",
}
SELECTION = "first-complete-tour-boundary-at-or-after-fixed-update-budget"


def require(test, message):
    if not test:
        raise RuntimeError(message)


def verify_handoff(path):
    from drama_pretrain import sha
    header = json.loads((path / "handoff.json").read_text())
    require(header.get("complete") is True and header.get("step") == 5000 and
            header.get("selection") == "fixed-final-step", "Incomplete or unselected pretraining handoff")
    require(all(header.get("architecture", {}).get(k) == v for k, v in ARCH.items()),
            "Expected 4-layer, 768-wide Mamba-2 checkpoint")
    require(header.get("model_file") == "final-model.pt", "Unexpected pretraining export filename")
    model_file = path / header["model_file"]
    require(sha(model_file) == header["model_sha256"], "Pretraining export checksum mismatch")
    require(len(header["identity_sha256"]) == 64, "Missing pretraining identity")
    return header, model_file


def tour_coverage(rows, tours):
    annotations = {row["instr_id"]: row["scan"] for row in rows}
    require(len(annotations) == len(rows), "Duplicate source instructions")
    members = set()
    for scan, scan_tours in tours.items():
        for tour in scan_tours:
            require(tour and all(annotations.get(i) == scan for i in tour),
                    "Author tour references unknown instruction or other scan")
            members.update(tour)
    require(set(annotations.values()) == set(tours), "Tour and annotation scans differ")
    return {"scans": len(tours), "annotation_instructions": len(rows),
            "effective_tour_instructions": len(members), "tours": sum(map(len, tours.values())),
            "annotations_outside_author_tours": sorted(annotations.keys() - members)}


def make_env(args, handoff):
    from drama_pretrain import sha
    from nav_src.envs.r2r.data_utils import construct_instrs
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from utils.data import ImageFeaturesDB
    r2r = Path(args.root_dir) / "R2R"
    rows = construct_instrs(args.anno_dir, args.dataset, ["train"], args.tokenizer, args.max_instr_len, is_test=False)
    tours = json.loads((r2r / "tours_iVLN.json").read_text())["train"]
    coverage = tour_coverage(rows, tours)
    require((coverage["scans"], coverage["tours"], coverage["annotation_instructions"],
             coverage["effective_tour_instructions"]) == (61, 183, 14039, 14025), "Released IR2R training membership differs")
    feature = Path(args.img_ft_file).relative_to(r2r).as_posix()
    graphs = [f"connectivity/{scan}_connectivity.json" for scan in tours]
    assets = {name: sha(r2r / name) for name in [*OFFICIAL, feature, *graphs]}
    require(all(assets[k] == v for k, v in OFFICIAL.items()), "Released train annotation/tour checksum differs")
    for name in ("annotations/R2R_train_enc.json", feature, *graphs):
        require(assets[name] == handoff["assets"].get(name), f"IR2R/pretraining asset mismatch: {name}")
    env = IR2RNavBatch(ImageFeaturesDB(args.img_ft_file, args.image_feat_size), rows, args.connectivity_dir,
                       tours, num_robots=1, batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,
                       seed=args.seed, name="train")
    return env, coverage, assets


def make_agent(args, env, pretrained):
    import torch
    from drama_ir2r import check_runtime, finite_step_guard, parameter_partition
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from vlnbert.drama_mamba2 import freeze_encoders
    agent = GMapNavAgent(args, env, rank=0)
    model = agent.vln_bert.vln_bert
    model.load_state_dict(pretrained, strict=True)
    model.requires_grad_(True)
    freeze_encoders(model)
    agent.vln_bert.train()
    for module in (model.embeddings, model.lang_encoder, model.img_embeddings):
        require(not module.training and all(not p.requires_grad for p in module.parameters()),
                "Encoder must stay frozen in training mode")
    trainable, delta = parameter_partition(model)
    named = dict(model.named_parameters())
    groups = [{"params": [p for n, p in named.items() if n in trainable and
                          bool(getattr(p, "_no_weight_decay", False)) == exempt],
               "weight_decay": 0. if exempt else .01} for exempt in (False, True)]
    # Foreach AdamW materializes parameter-sized tensor lists on CUDA during step().
    # Use the single-tensor implementation so a larger physical rollout batch can fit.
    agent.vln_bert_optimizer = torch.optim.AdamW(groups, lr=args.lr, foreach=False)
    agent.optimizers = agent.vln_bert_optimizer
    agent.optimizers.register_step_pre_hook(lambda opt, pos, kw: finite_step_guard(opt, agent.loss))
    check_runtime(agent, env)
    return agent, {k: v for k, v in pretrained.items() if k not in delta}, sorted(trainable)


def roots(cli):
    stage = f"ir2r-b{cli.batch_size}"
    return cli.run_root.resolve() / stage, cli.backup_root.resolve() / stage


def train(cli):
    import torch
    from colab_full_storage import FullStore
    from colab_full_pretrain import git_revision
    from drama_ir2r import (NUMERICS, author_args, canonical_boundary, complete_tour, configure_numerics,
                            digest, restore_runtime, runtime_state, static_environment)
    from drama_mount_guard import validate_separate_roots
    from drama_pretrain import (atomic_torch, code_manifest, json_sha, runtime_identity, seed_all,
                                space_guard, tensor_bytes, verified_copy, write_json)
    require(os.environ.get("PYTHONHASHSEED") == "0", "Set PYTHONHASHSEED=0 before starting Python")
    require(torch.cuda.is_available(), "IR2R training requires a CUDA GPU")
    configure_numerics({"numerics": NUMERICS})
    run, backup = roots(cli)
    validate_separate_roots(run, backup)
    if cli.require_resume:
        require((backup / "identity.json").is_file() and (backup / "checkpoints/STORE.json").is_file(),
                "Strict resume requires an existing durable experiment")
    else:
        require(not run.exists() and not backup.exists(), "Fresh training refuses existing experiment paths")
    store = FullStore(run / "checkpoints", backup / "checkpoints", backend=cli.backup_backend, keep_last=1)
    with store.lock():
        if not cli.require_resume:
            require(not any((p / "identity.json").exists() or (p / "checkpoints/refs.json").exists()
                            for p in (run, backup)), "Fresh run found published state under the writer lock")
        pre = cli.pretrain.resolve()
        if not all((pre / f).is_file() for f in ("handoff.json", "final-model.pt")):
            pre = cli.backup_root.resolve() / "pretrain"
        header, model_file = verify_handoff(pre)
        seed_all(0)
        batch = cli.batch_size
        target = BUDGET[batch]
        args = author_args({"datasets": str(cli.datasets.resolve()), "run_dir": str(run), "seed": 0,
                            "batch_size": batch, "minimum_updates": target, "lr": 1e-5},
                           header["architecture"])
        env, coverage, assets = make_env(args, header)
        # The terminal pretraining export uses torch.save's ZIP format. The local
        # mapping lets the kernel reclaim pages once parameters are copied to GPU.
        weights = torch.load(model_file, weights_only=True, map_location="cpu",
                             mmap=(pre == cli.pretrain.resolve()))
        agent, frozen, trainable = make_agent(args, env, weights)
        del weights
        runtime = runtime_identity()
        runtime["bert_config"].pop("resolved_path")
        if runtime["mattersim"]:
            runtime["mattersim"].pop("path")
        mount = {k: store.mount[k] for k in ("backend", "mount_root", "filesystem", "source")}
        contract = {"stage": "memoir-full-ir2r-v1", "pretrain_handoff_sha256": json_sha(header),
                    "pretrain_model_sha256": header["model_sha256"], "architecture": header["architecture"],
                    "assets": assets, "tour_coverage": coverage, "environment_sha256": digest(static_environment(env)),
                    "source_commit": git_revision(), "source_files": code_manifest(),
                    "runtime": runtime, "gpu": torch.cuda.get_device_name(),
                    "capability": list(torch.cuda.get_device_capability()),
                    "batch_size": batch, "minimum_updates": target, "seed": 0, "selection": SELECTION,
                    "trainable_names": trainable, "optimizer": "AdamW", "foreach": False, "lr": args.lr, "clip_grad_norm": 40.,
                    "weight_decay": .01, "mamba_no_weight_decay_group": 0., "train_alg": args.train_alg,
                    "teacher_ml_weight": args.ml_weight, "save_every": cli.save_every,
                    "backup_mount": mount}
        if cli.require_resume:
            identity = json.loads((backup / "identity.json").read_text())
            require(identity["contract"] == contract and identity["store_id"] == store.run_id,
                    "Strict resume rejected changed model/source/data/optimizer/runtime/store")
            if (run / "identity.json").exists():
                require(json.loads((run / "identity.json").read_text()) == identity, "Local/durable identities disagree")
            write_json(run / "identity.json", identity)
        else:
            identity = {"contract": contract, "store_id": store.run_id, "creation_mount": store.mount}
            write_json(run / "identity.json", identity)
            verified_copy(run / "identity.json", backup / "identity.json")
        identity_sha = json_sha(identity)
        progress = {"completed_updates": 0, "last_full_step": 0, "loss_history": []}
        if cli.require_resume:
            state, head, manifest = store.restore("latest")
            require(manifest["step"] == state["step"] and head["pretrain_model_sha256"] == header["model_sha256"],
                    "Checkpoint metadata disagrees with pretrained base")
            progress = restore_runtime(agent, state, identity_sha, frozen)
            del state
        del frozen

        def save():
            progress["last_full_step"] = progress["completed_updates"]
            state = runtime_state(agent, progress, identity_sha)
            store.save(state, {"pretrain_model_sha256": header["model_sha256"]}, step=state["step"])
            print("IR2R_CHECKPOINT", state["step"], flush=True)
            store.backup_check()
            for directory in (run, backup):
                write_json(directory / "training-loss.json", progress["loss_history"])

        if not cli.require_resume:
            save()
        stopped = [False]

        def request_stop(signum, frame):
            stopped[0] = True

        signals = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                step = progress["completed_updates"]
                stopping = stopped[0] or (cli.stop_after is not None and step >= cli.stop_after)
                if complete_tour(env) and (stopping or step >= target):
                    if step != progress["last_full_step"]:
                        save()
                    break
                agent.train(1, feedback="sample", extended_memory=True)
                loss = float(agent.loss.detach())
                step += 1
                progress["completed_updates"] = step
                progress["loss_history"].append(loss)
                boundary = complete_tour(env)
                print("IR2R_PROGRESS", json.dumps({"update": step, "target": target, "batch": batch,
                      "loss": loss, "tour_boundary": boundary, "unix_time": time.time(),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}), flush=True)
                if boundary:
                    canonical_boundary(agent)
                    if step - progress["last_full_step"] >= cli.save_every:
                        save()
        finally:
            for sig, handler in signals.items():
                signal.signal(sig, handler)

        if step >= target:
            store.backup_check()
            final_file = run / "final-model.pt"
            state = agent.vln_bert.vln_bert.state_dict()
            export_size = tensor_bytes(state) + 32 * 1024**2
            if not (backup / "final-model.pt").exists():
                # Filesystem backups can share a physical device with local outputs.
                copies = 2 if run.stat().st_dev == backup.stat().st_dev else 1
                space_guard((run, backup), store.min_free_gib, copies * export_size)
                atomic_torch(final_file, {k: v.detach().cpu() for k, v in state.items()})
                store.backup_check()
                model_sha = verified_copy(final_file, backup / final_file.name)
            else:
                # A process can stop after publishing the model but before publishing completion.
                # Confirm the durable model really equals this restored final training state.
                if not final_file.exists():
                    space_guard((run,), store.min_free_gib, export_size)
                model_sha = verified_copy(backup / final_file.name, final_file)
                published = torch.load(final_file, weights_only=True, map_location="cpu", mmap=True)
                require(published.keys() == state.keys() and
                        all(torch.equal(published[k], v.detach().cpu()) for k, v in state.items()),
                        "Existing final export differs from final training state")
                del published
            result = {"complete": True, "step": step, "minimum_updates": target, "batch_size": batch,
                      "model_file": final_file.name, "model_sha256": model_sha, "identity_sha256": identity_sha,
                      "architecture": header["architecture"], "selection": SELECTION,
                      "tour_coverage": coverage, "validation_accesses": 0}
            store.backup_check()
            write_json(run / "completion.json", result)
            verified_copy(run / "completion.json", backup / "completion.json")
            print("IR2R_COMPLETE", json.dumps(result), flush=True)
        else:
            print("IR2R_STOPPED", step, "resume with --require-resume", flush=True)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, required=True, help="Parent of R2R/")
    p.add_argument("--pretrain", type=Path, required=True, help="Pretraining run containing handoff.json")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--backup-root", type=Path, required=True)
    p.add_argument("--backup-backend", choices=("drive", "filesystem"), required=True)
    p.add_argument("--batch-size", type=int, choices=BUDGET, default=8)
    p.add_argument("--auto-batch", action="store_true", help="Start at 8. Only CUDA OOM starts a new lower-batch experiment")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--require-resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Stop at the next complete optimizer/tour boundary")
    args = p.parse_args(argv)
    if args.save_every <= 0 or (args.stop_after is not None and args.stop_after < 1):
        p.error("save and stop thresholds must be positive")
    if args.auto_batch and (args.batch_size != 8 or args.require_resume):
        p.error("--auto-batch starts new experiments at batch 8. To resume, specify the existing batch explicitly.")
    return args


def main(argv):
    cli = parse(argv)
    if cli.auto_batch:
        common = [x for x in argv if x != "--auto-batch"]
        for batch in BUDGET:
            result = subprocess.run([sys.executable, "-u", __file__, *common, "--batch-size", str(batch)])
            if result.returncode != OOM_EXIT:
                return result.returncode
            print(f"IR2R_BATCH_OOM batch={batch}; its checkpoints stay in ir2r-b{batch}", flush=True)
        return OOM_EXIT
    import torch
    try:
        train(cli)
    except torch.cuda.OutOfMemoryError as exc:
        if "cuda" not in str(exc).lower():
            raise
        print(f"CUDA_OOM ir2r-b{cli.batch_size}: {exc}", file=sys.stderr, flush=True)
        return OOM_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
