"""Verified checkpoint copies for an interruptible, single-writer training run.

Use one run directory on a *mounted* persistent drive as ``backup_dir``. Mount
validation belongs to the caller. A local flock prevents competing processes on
one VM; it is not a distributed lock. Never run two VMs against the same run.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator
import uuid

import torch


FORMAT = "vln-checkpoint-store-v1"
ID_RE = re.compile(r"^step-\d{12,}-[0-9a-f]{32}$")


class CheckpointError(RuntimeError):
    """A checkpoint or its store identity failed validation."""


class BackupError(CheckpointError):
    """The durable copy failed; the local recovery point is retained."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            # Some mounted-drive filesystems do not implement directory fsync.
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    def __init__(
        self, local_dir: str | Path, backup_dir: str | Path, *,
        keep_local: int = 2, keep_backup: int = 5,
    ) -> None:
        for name, count in (("keep_local", keep_local), ("keep_backup", keep_backup)):
            if type(count) is not int or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.local_dir = Path(local_dir).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        if (self.local_dir.is_relative_to(self.backup_dir)
                or self.backup_dir.is_relative_to(self.local_dir)):
            raise ValueError("local_dir and backup_dir must be separate, non-nested directories")
        self.keep_local, self.keep_backup = keep_local, keep_backup
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_file = None
        with self.lock():
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            identities = []
            for root in (self.local_dir, self.backup_dir):
                marker = root / "STORE.json"
                if marker.exists():
                    try:
                        identity = json.loads(marker.read_bytes())
                        if (marker.is_symlink() or identity.get("format") != FORMAT
                                or not isinstance(identity.get("run_id"), str)
                                or not re.fullmatch(r"[0-9a-f]{32}", identity["run_id"])):
                            raise ValueError("invalid identity")
                    except (ValueError, AttributeError, OSError) as error:
                        raise CheckpointError(f"Invalid store identity: {marker}") from error
                    identities.append(identity["run_id"])
            if len(set(identities)) > 1:
                raise CheckpointError("local and backup directories belong to different runs")
            self.run_id = identities[0] if identities else uuid.uuid4().hex
            for root in (self.local_dir, self.backup_dir):
                if not (root / "STORE.json").exists():
                    _atomic_bytes(root / "STORE.json", _json_bytes({"format": FORMAT, "run_id": self.run_id}))
                snapshots = root / "snapshots"
                if snapshots.is_symlink():
                    raise CheckpointError("snapshots directory must not be a symlink")
                snapshots.mkdir(exist_ok=True)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold a reentrant local, process-exclusive, nonblocking run lock."""
        with self._thread_lock:
            if self._lock_depth == 0:
                self._lock_file = (self.local_dir / ".writer.lock").open("a+b")
                try:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    self._lock_file.close()
                    self._lock_file = None
                    raise CheckpointError("another process holds this local run's writer lock") from error
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    self._lock_file.close()
                    self._lock_file = None

    def _verify(self, directory: Path, checkpoint_id: str | None = None) -> dict[str, Any]:
        if directory.is_symlink() or not directory.is_dir():
            raise CheckpointError(f"invalid checkpoint directory: {directory}")
        try:
            for name in ("manifest.json", "COMMITTED", "state.pt", "head.pt"):
                if (directory / name).is_symlink() or not (directory / name).is_file():
                    raise ValueError(f"missing or symlinked {name}")
            raw = (directory / "manifest.json").read_bytes()
            if (directory / "COMMITTED").read_text().strip() != hashlib.sha256(raw).hexdigest():
                raise ValueError("manifest digest mismatch")
            manifest = json.loads(raw)
            identity = manifest["checkpoint_id"]
            if (manifest["format"] != FORMAT or manifest["run_id"] != self.run_id
                    or not isinstance(identity, str) or not ID_RE.fullmatch(identity)
                    or (checkpoint_id is not None and identity != checkpoint_id)
                    or type(manifest["step"]) is not int or manifest["step"] < 0
                    or type(manifest["created_ns"]) is not int
                    or type(manifest["is_best"]) is not bool):
                raise ValueError("invalid checkpoint metadata")
            if set(manifest["files"]) != {"state.pt", "head.pt"}:
                raise ValueError("unexpected checkpoint file list")
            for name, expected in manifest["files"].items():
                file = directory / name
                if file.stat().st_size != expected["size"] or _sha256(file) != expected["sha256"]:
                    raise ValueError(f"checksum mismatch for {name}")
            return manifest
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise CheckpointError(f"Invalid checkpoint {directory}: {error}") from error

    def _snapshots(self, root: Path, *, best_only: bool = False) -> list[tuple[Path, dict[str, Any]]]:
        result = []
        for directory in (root / "snapshots").iterdir():
            if not ID_RE.fullmatch(directory.name):
                continue
            try:
                manifest = self._verify(directory, directory.name)
            except CheckpointError:
                continue
            if not best_only or manifest["is_best"]:
                result.append((directory, manifest))
        return sorted(result, key=lambda item: (item[1]["created_ns"], item[0].name), reverse=True)

    def _pointer(self, root: Path, which: str) -> tuple[Path, dict[str, Any]] | None:
        try:
            # One atomic document publishes both references. The separately
            # named JSON files are compatibility mirrors and never authoritative.
            path = root / "refs.json"
            if path.is_symlink():
                return None
            refs = json.loads(path.read_bytes())
            if refs["format"] != FORMAT or refs["run_id"] != self.run_id:
                return None
            pointer = refs[which]
            identity = pointer["checkpoint_id"]
            if (pointer["run_id"] != self.run_id or not isinstance(identity, str)
                    or not ID_RE.fullmatch(identity)):
                return None
            directory = root / "snapshots" / identity
            manifest = self._verify(directory, identity)
            if (pointer["manifest_sha256"] != _sha256(directory / "manifest.json")
                    or (which == "best" and not manifest["is_best"])):
                return None
            return directory, manifest
        except (OSError, ValueError, KeyError, TypeError, CheckpointError):
            return None

    def _update_pointers(self, root: Path, directory: Path, *, is_best: bool) -> None:
        def reference(path: Path) -> dict[str, str]:
            return {"run_id": self.run_id, "checkpoint_id": path.name,
                    "manifest_sha256": _sha256(path / "manifest.json")}

        latest = reference(directory)
        best = self._pointer(root, "best")
        if best is None:
            previous_best = self._snapshots(root, best_only=True)
            best = previous_best[0] if previous_best else None
        selected_best = latest if is_best else (reference(best[0]) if best else None)
        refs = {"format": FORMAT, "run_id": self.run_id, "latest": latest, "best": selected_best}
        writes = [(root / "latest.json", _json_bytes(latest))]
        if selected_best is not None:
            writes.append((root / "best.json", _json_bytes(selected_best)))
        # Commit last. A process killed between the mirrors leaves the old
        # refs.json intact, so latest and best still belong to one transaction.
        writes.append((root / "refs.json", _json_bytes(refs)))
        targets = [path for path, _ in writes]
        previous = {path: path.read_bytes() if path.exists() else None for path in targets}
        changed = []
        try:
            for path, content in writes:
                changed.append(path)
                _atomic_bytes(path, content)
        except BaseException:
            # A copy failure occurs before this transaction. If a pointer write
            # itself fails, restore the previous refs as far as the FS permits.
            for path in reversed(changed):
                try:
                    if previous[path] is None:
                        path.unlink(missing_ok=True)
                    else:
                        _atomic_bytes(path, previous[path])
                except OSError:
                    pass  # Completed immutable snapshots remain discoverable.
            raise

    def _copy_snapshot(self, source: Path, root: Path) -> Path:
        manifest = self._verify(source, source.name)
        destination = root / "snapshots" / source.name
        if destination.exists():
            existing = self._verify(destination, source.name)
            if existing != manifest:
                raise CheckpointError("checkpoint ID collision")
            return destination
        staging = root / "snapshots" / f".pending-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for name in ("state.pt", "head.pt", "manifest.json"):
                shutil.copyfile(source / name, staging / name)
                with (staging / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            # Publish COMMITTED only after the copied bytes have been checked.
            for name, expected in manifest["files"].items():
                if (staging / name).stat().st_size != expected["size"] or _sha256(staging / name) != expected["sha256"]:
                    raise CheckpointError(f"copied {name} failed checksum verification")
            if _sha256(staging / "manifest.json") != _sha256(source / "manifest.json"):
                raise CheckpointError("copied manifest failed checksum verification")
            _atomic_bytes(staging / "COMMITTED", (_sha256(staging / "manifest.json") + "\n").encode())
            self._verify(staging, source.name)
            os.replace(staging, destination)
            _sync_dir(destination.parent)
            self._verify(destination, source.name)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def save(
        self, state: dict, head_payload: dict, *, step: int,
        is_best: bool = False, metrics: dict | None = None,
    ) -> str:
        """Synchronously publish a snapshot; return only after a verified backup.

        A BackupError stops the caller before another training step. No retention
        cleanup happens after failure. The local snapshot can still be recovered.
        """
        if type(step) is not int or step < 0 or type(is_best) is not bool:
            raise ValueError("step must be a nonnegative integer and is_best a bool")
        if not isinstance(state, dict) or not isinstance(head_payload, dict):
            raise ValueError("state and head_payload must be dictionaries")
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary or None")
        _json_bytes(metrics)  # Reject nonfinite/unserializable metrics before writing.
        with self.lock():
            identity = f"step-{step:012d}-{uuid.uuid4().hex}"
            staging = self.local_dir / "snapshots" / f".pending-{uuid.uuid4().hex}"
            destination = self.local_dir / "snapshots" / identity
            staging.mkdir()
            try:
                for name, payload in (("state.pt", state), ("head.pt", head_payload)):
                    with (staging / name).open("xb") as stream:
                        torch.save(payload, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                manifest = {
                    "format": FORMAT, "run_id": self.run_id, "checkpoint_id": identity,
                    "step": step, "created_ns": time.time_ns(), "is_best": is_best,
                    "metrics": metrics or {},
                    "files": {name: {"sha256": _sha256(staging / name), "size": (staging / name).stat().st_size}
                              for name in ("state.pt", "head.pt")},
                }
                _atomic_bytes(staging / "manifest.json", _json_bytes(manifest))
                _atomic_bytes(staging / "COMMITTED", (_sha256(staging / "manifest.json") + "\n").encode())
                self._verify(staging, identity)
                os.replace(staging, destination)
                _sync_dir(destination.parent)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            self._update_pointers(self.local_dir, destination, is_best=is_best)
            try:
                backup = self._copy_snapshot(destination, self.backup_dir)
                self._update_pointers(self.backup_dir, backup, is_best=is_best)
            except Exception as error:
                raise BackupError(f"Backup failed; local checkpoint retained at {destination}: {error}") from error
            self._prune()
            return identity

    def _prune(self) -> None:
        # Cleanup is deliberately conservative: only verified checkpoints owned
        # by this run can be removed; unknown/corrupt/incomplete files survive.
        for root, count in ((self.local_dir, self.keep_local), (self.backup_dir, self.keep_backup)):
            snapshots = self._snapshots(root)
            protected = {path.name for path, _ in snapshots[:count]}
            for which in ("latest", "best"):
                pointed = self._pointer(root, which)
                if pointed:
                    protected.add(pointed[0].name)
                elif which == "best":
                    previous_best = next((path for path, meta in snapshots if meta["is_best"]), None)
                    if previous_best:
                        protected.add(previous_best.name)
            for directory, _ in snapshots:
                if directory.name in protected:
                    continue
                if root == self.local_dir:
                    try:
                        self._verify(self.backup_dir / "snapshots" / directory.name, directory.name)
                    except CheckpointError:
                        continue  # Never discard an unbacked local recovery point.
                shutil.rmtree(directory)

    def restore(self, which: str = "latest") -> tuple[dict, dict, dict[str, Any]]:
        """Restore a verified copy, preferring backup and falling back on damage.

        ``best`` only considers snapshots explicitly saved with is_best=True.
        ``latest`` uses creation time (not training step) for fallback snapshots.
        Returned manifest additionally contains the local snapshot path.
        """
        if which not in {"latest", "best"}:
            raise ValueError("which must be 'latest' or 'best'")
        with self.lock():
            attempted = []
            requested_checkpoint_exists = False
            for root in (self.backup_dir, self.local_dir):
                requested_checkpoint_exists |= (root / f"{which}.json").exists()
                if which == "latest":
                    requested_checkpoint_exists |= any(
                        ID_RE.fullmatch(path.name) for path in (root / "snapshots").iterdir()
                    )
                pointed = self._pointer(root, which)
                candidates = ([pointed] if pointed else []) + self._snapshots(root, best_only=which == "best")
                seen = set()
                for directory, manifest in candidates:
                    if directory.name in seen:
                        continue
                    seen.add(directory.name)
                    try:
                        if root == self.backup_dir:
                            local = self.local_dir / "snapshots" / directory.name
                            if local.exists():
                                try:
                                    self._verify(local, local.name)
                                except CheckpointError:
                                    # Preserve damaged data for diagnosis, outside retention.
                                    os.replace(local, local.with_name(f".corrupt-{local.name}-{uuid.uuid4().hex}"))
                            local = self._copy_snapshot(directory, self.local_dir)
                        else:
                            local = directory
                        self._verify(local, directory.name)
                        # Loading is from the verified local bytes, never pickle unrestricted mode.
                        state = torch.load(local / "state.pt", map_location="cpu", weights_only=True)
                        head = torch.load(local / "head.pt", map_location="cpu", weights_only=True)
                        if not isinstance(state, dict) or not isinstance(head, dict):
                            raise CheckpointError("checkpoint payloads must be dictionaries")
                        return state, head, {**manifest, "local_path": str(local)}
                    except Exception as error:
                        attempted.append(f"{directory}: {error}")
            if attempted or requested_checkpoint_exists:
                raise CheckpointError("No loadable checkpoint: " + (
                    "; ".join(attempted) or "existing checkpoint references or files failed validation"
                ))
            raise FileNotFoundError(f"No valid {which} checkpoint in {self.backup_dir} or {self.local_dir}")
