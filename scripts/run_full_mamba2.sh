#!/usr/bin/env bash
# Run both stages in order. On recovery, choose the phase explicitly.
set -euo pipefail
mode="${1:?choose start, resume-pretrain, or resume-ir2r}"
: "${MEMOIR_DATASETS:?set the parent of R2R/}"
: "${MEMOIR_RUN_ROOT:?set a new local experiment root}"
: "${MEMOIR_BACKUP_ROOT:?set the persistent experiment root}"
: "${MEMOIR_BACKUP_BACKEND:?choose drive or filesystem}"

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
common=(--run-root "$MEMOIR_RUN_ROOT" --backup-root "$MEMOIR_BACKUP_ROOT"
        --backup-backend "$MEMOIR_BACKUP_BACKEND")
pre=(python -u scripts/colab_full_pretrain.py --asset-root "$MEMOIR_DATASETS/R2R" "${common[@]}")
post=(python -u scripts/colab_full_ir2r.py --datasets "$MEMOIR_DATASETS"
      --pretrain "$MEMOIR_RUN_ROOT/pretrain" "${common[@]}")

case "$mode" in
  start|resume-pretrain)
    pre_flags=()
    if [[ "$mode" == resume-pretrain ]]; then pre_flags+=(--require-resume); fi
    "${pre[@]}" "${pre_flags[@]}"
    # The IR2R entry point verifies the full SHA and step-5000 handoff.
    exec "${post[@]}" --auto-batch
    ;;
  resume-ir2r)
    exec "${post[@]}" --require-resume --batch-size "${2:?set the original physical batch}"
    ;;
  *)
    echo "Unknown phase: $mode" >&2
    exit 2
    ;;
esac
