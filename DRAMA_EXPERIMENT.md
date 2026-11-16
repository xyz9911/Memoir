# Frozen-encoder Mamba-2 experiment

This branch replaces Memoir's temporal world-model prior with a differentiable
Mamba-2 recurrence inspired by Drama. It retains Memoir's continuous latent,
language conditioning, observation reconstruction and navigation interfaces.
It is not a reproduction of Drama's Atari agent. See
[model details and pinned sources](vlnbert/DRAMA_MAMBA2.md).

## Fixed training protocol

1. Load the released Memoir encoder/navigation initialization. Initialize all
   world-model modules afresh, including the Mamba-2 prior.
2. Freeze language and panorama encoders in eval mode. Cache their FP32 outputs
   along ground-truth training trajectories. No navigation gradient enters the
   encoders.
3. Train only the world model for **6000 optimizer updates**, batch **32**:
   observation contrastive reconstruction, goal-distance prediction, posterior
   KL, and five-step overshooting. Use the author's AdamW implementation,
   learning rate 5e-5, betas (0.9, 0.98), epsilon 1e-6, weight decay 0.01,
   gradient clipping 5, 100-step warmup and linear decay. Mamba parameters marked
   `_no_weight_decay` are excluded from decay. Select on full train_dev loss.
4. Load that selected world model and train IR2R with the encoders still frozen.
   Train the world model and navigation modules together. Use the author's
   teacher-plus-sampled DAgger update, AdamW at 1e-5, and at least **20000 updates**.
   Finish at a complete tour/update boundary. The fixed JSON uses batch **2**.
   Evaluate full train_dev tours at the first complete tour boundary after each
   500-update threshold; select by SR, then SPL.
5. Compare the selected IR2R model with the author's released IR2R checkpoint on
   the same 2889 train_dev instructions. Save complete-tour trajectories and
   independently recomputable SR, SPL, NE and nDTW. Each tour has its own fixed
   seed and retains memory across its instructions. This final evaluation has a
   different stochastic stream from training-time selection. The reference was
   trained on the original full training set and selected by its authors on
   val_unseen; this is a released-reference comparison, not a matched-budget
   ablation or blind test. This evaluator does not read val_unseen data.

This is a new world-model-only pretraining objective, not the author's complete
joint navigation/language pretraining. Its loss normalization and negative pool
are documented in the implementation and must not be presented as equivalent.

The frozen scan partition uses seed 20261003: 49 train_fit scans and 12 train_dev
scans. Preserve official tours. Exclude 14 annotation IDs absent from those tours
(13 fit, 1 dev), recording their IDs in cache/run identities. The actual scope is
11136 fit instructions in 147 tours and 2889 dev instructions in 36 tours.
No val_unseen data is read by this experiment. Internal train_dev results are not
blind-test results and cannot be compared directly with a different split.

## Storage and continuation

Configs explicitly select a persistent ext4 loop filesystem on the Windows G
volume. The launcher checks the exact mount, backing image and host volume before
writing. Missing mounts fail rather than falling back to the nearly full WSL root
volume. Both host and work filesystem retain at least 12 GiB free, plus the budget
needed to publish two new checkpoint copies before pruning.

Checkpoints contain every changing model tensor, optimizer state, RNG states,
sampler/environment position, selection history and scheduler where applicable.
An immutable frozen base is saved once and SHA-bound to every compact snapshot.
Local and filesystem-backup snapshots are read back and SHA-verified before
retaining only latest and best. Two directories on this disk protect continuation
from process failure, not physical disk loss.

Run from the repository in the prepared Python environment:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python -u scripts/drama_pipeline.py \
  --pretrain-config configs/drama-pretrain.json \
  --ir2r-config configs/drama-ir2r.json \
  --eval-config configs/drama-evaluate.json \
  --journal /home/xxl/memoir-disk-20261116/runs/drama-v1/pipeline.json
```

Use a detached tmux session for the command. For continuation, keep source,
assets, configs, environment and mounts unchanged and add `--require-resume`.
The queue verifies completed artifacts and resumes an interrupted training stage
strictly; it does not silently initialize a replacement run.

The prepared WSL deployment can use `bash scripts/run_drama_background.sh`.
It verifies mounts, disk reserves and a clean committed repository, then creates
the `memoir-drama` tmux session. Logs are in `runs/drama-v1/pipeline.log` on the
work disk. Append `--require-resume` only when continuing an existing pipeline.
Numerical configuration is explicit: deterministic algorithms, TF32 disabled,
cuDNN deterministic mode and the fixed cuBLAS workspace setting. Runtime identity
also binds the actual offline BERT configuration and MatterSim binary hashes.

## Verification

```bash
python -m pytest -q tests/test_drama_mamba2.py tests/test_drama_pretrain.py tests/test_drama_ir2r.py
```

The official Mamba reference test additionally needs the pinned einops version in
`tests/requirements-drama.txt`. Engineering GPU probes are separate from full
training and must not be interpreted as navigation scores. Actual run status,
source commit, configurations, asset hashes and engineering evidence are archived
by the experiment operator before launch.
