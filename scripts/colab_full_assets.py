"""Stage the pinned Memoir pretraining and IR2R assets under datasets/R2R."""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote


MANIFEST = Path(__file__).resolve().parents[1] / "configs/colab-full-assets.json"


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {spec['bytes']}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")


def stage_file(root, url_prefix, name, spec, verify_only):
    path = root / name
    if path.exists() or verify_only:
        verify(path, spec)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if not partial.exists() or partial.stat().st_size != spec["bytes"]:
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "4", "--connect-timeout", "20", "--speed-time", "90",
            "--speed-limit", "4096", "--continue-at", "-", "--output", str(partial),
            f"{url_prefix}/{quote(name)}",
        ], check=True)
    verify(partial, spec)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/R2R"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}"
    files = manifest["files"]
    # Start the long visual feature transfer before the small graph downloads.
    names = sorted(files, key=lambda name: files[name]["bytes"], reverse=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {name: pool.submit(stage_file, args.root, url, name, files[name], args.verify_only)
                for name in names}
        for name, job in jobs.items():
            job.result()
    print(f"Verified {len(names)} files ({sum(f['bytes'] for f in files.values()):,} bytes) "
          f"from {manifest['repository']} at {manifest['revision']} under {args.root}")


if __name__ == "__main__":
    main()
