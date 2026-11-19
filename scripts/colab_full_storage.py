"""Verified checkpoint storage for the full-data experiment on Linux."""
import os
from pathlib import Path

from drama_mount_guard import MountedCheckpointStore, backup_mount_identity, validate_backup_root
from drama_pretrain import space_guard, tensor_bytes


class FullStore(MountedCheckpointStore):
    def __init__(self, local: Path, backup: Path, *, backend: str,
                 min_free_gib: float = 4, keep_last: int = 2):
        local, backup = Path(local).resolve(), Path(backup).resolve()
        if backend not in {"drive", "filesystem"} or min_free_gib < 0:
            raise ValueError("Choose drive or filesystem and a nonnegative free-space reserve")
        # Verify the mounted parent before creating any missing stage directories.
        parent = next(path for path in (backup, *backup.parents) if path.exists())
        self.mount = backup_mount_identity(parent, backend=backend)
        required_mount = os.environ.get("MEMOIR_BACKUP_MOUNT")
        if required_mount and Path(self.mount["mount_point"]) != Path(required_mount).resolve():
            raise RuntimeError(f"Expected backup mount {required_mount}, found {self.mount['mount_point']}")
        backup.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        self.min_free_gib = min_free_gib

        def check():
            validate_backup_root(backup, backend=backend, expected_identity=self.mount)
            space_guard((local, backup), min_free_gib)

        super().__init__(local, backup, backup_check=check, keep_local=keep_last, keep_backup=keep_last)

    def save(self, state, head_payload, **kwargs):
        self.backup_check()
        size = tensor_bytes(state) + tensor_bytes(head_payload) + 16 * 1024**2
        shared = self.local_dir.stat().st_dev == self.backup_dir.stat().st_dev
        space_guard((self.local_dir, self.backup_dir), self.min_free_gib, size * (2 if shared else 1))
        return super().save(state, head_payload, **kwargs)

    def restore(self, which="latest"):
        self.backup_check()
        return super().restore(which)
