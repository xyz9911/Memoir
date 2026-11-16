#!/usr/bin/env bash
# WSL deployment entry point. No automatic fallback to a different run or volume.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"
source /home/xxl/projects/memoir-drama-env-20261116/activate-memoir.sh
export PYTHONPATH="$repo_dir:$script_dir:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
worker=0
if [[ "${1:-}" == --worker ]]; then worker=1; shift; fi
resume=()
if [[ "${1:-}" == --require-resume ]]; then resume=(--require-resume); shift; fi
if (( $# )); then printf 'Unexpected argument: %s\n' "$1" >&2; exit 2; fi
python - <<'PY'
import json
import subprocess
from pathlib import Path
from drama_pretrain import verify_storage_mount, space_guard
for name in ('drama-pretrain.json','drama-ir2r.json','drama-evaluate.json'):
    cfg=json.loads((Path('configs')/name).read_text())
    verify_storage_mount(cfg)
    space_guard([cfg['storage_mount'],cfg['host_volume']],cfg['min_free_gib'])
status=subprocess.check_output(['git','status','--porcelain'],text=True)
if status.strip():
    raise RuntimeError('Commit the deployed source/configuration before launching: '+status)
PY
run_root=/home/xxl/memoir-disk-20261116/runs/drama-v1
mkdir -p -- "$run_root"
if (( ! worker )); then
    if tmux has-session -t memoir-drama 2>/dev/null; then
        printf 'tmux session memoir-drama already exists; inspect it before starting another writer.\n' >&2
        exit 1
    fi
    printf -v task_command '%q ' bash "$script_dir/run_drama_background.sh" --worker "${resume[@]}"
    tmux new-session -d -s memoir-drama -c "$repo_dir" "$task_command"
    printf 'Started tmux memoir-drama; log: %s/pipeline.log\n' "$run_root"
    exit 0
fi
exec 9>"$run_root/launcher.lock"
flock -n 9 || { printf 'Another pipeline writer holds the launcher lock.\n' >&2; exit 1; }
exec >>"$run_root/pipeline.log" 2>&1
printf 'LAUNCH utc=%s commit=%s resume=%s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "${resume[*]:-false}"
set +e
python -u scripts/drama_pipeline.py \
    --pretrain-config configs/drama-pretrain.json \
    --ir2r-config configs/drama-ir2r.json \
    --eval-config configs/drama-evaluate.json \
    --journal "$run_root/pipeline.json" "${resume[@]}"
result=$?
set -e
python - "$run_root/launcher-exit.json" "$result" <<'PY'
import sys,time
from drama_pretrain import write_json
write_json(sys.argv[1],{'exit_code':int(sys.argv[2]),'finished_unix':time.time()})
PY
printf 'PIPELINE_EXIT code=%s utc=%s\n' "$result" "$(date -u +%FT%TZ)"
exit "$result"
