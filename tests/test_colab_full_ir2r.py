import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_full_ir2r as full


def arguments(tmp_path):
    return ["--datasets", str(tmp_path / "data"), "--pretrain", str(tmp_path / "pretrain"),
            "--run-root", str(tmp_path / "runs"), "--backup-root", str(tmp_path / "backup"),
            "--backup-backend", "drive"]


def test_author_annotation_and_tour_populations_remain_distinct():
    rows = [{"instr_id": "1", "scan": "s"}, {"instr_id": "2", "scan": "s"},
            {"instr_id": "3", "scan": "t"}, {"instr_id": "4", "scan": "s"}]
    coverage = full.tour_coverage(rows, {"s": [["1", "2"]], "t": [["3"]]})
    assert coverage == {"scans": 2, "tours": 2, "annotation_instructions": 4,
                        "effective_tour_instructions": 3, "annotations_outside_author_tours": ["4"]}
    with pytest.raises(RuntimeError, match="unknown instruction or other scan"):
        full.tour_coverage(rows, {"s": [["1", "3"]]})


def test_handoff_rejects_corrupt_or_partial_pretraining(tmp_path):
    model = tmp_path / "final-model.pt"
    model.write_bytes(b"fixture")
    header = {"complete": True, "step": 5000, "selection": "fixed-final-step",
              "model_file": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "identity_sha256": "b" * 64, "architecture": full.ARCH}
    manifest = tmp_path / "handoff.json"
    manifest.write_text(json.dumps(header))
    assert full.verify_handoff(tmp_path) == (header, model)
    for k, bad in (("complete", False), ("step", 4999), ("architecture", {"d_model": 512}),
                   ("model_sha256", "0" * 64)):
        manifest.write_text(json.dumps({**header, k: bad}))
        with pytest.raises(RuntimeError):
            full.verify_handoff(tmp_path)


@pytest.mark.parametrize("return_codes,expected_batches", [
    ([0], [8]), ([full.OOM_EXIT, 0], [8, 4]), ([full.OOM_EXIT, full.OOM_EXIT, 0], [8, 4, 2]),
    ([4], [8]), ([full.OOM_EXIT, 14], [8, 4]), ([full.OOM_EXIT] * 3, [8, 4, 2])])
def test_only_explicit_cuda_oom_falls_back(tmp_path, monkeypatch, return_codes, expected_batches):
    invoked = []
    def run(cmd):
        invoked.append(int(cmd[-1]))
        return SimpleNamespace(returncode=return_codes[len(invoked)-1])
    monkeypatch.setattr(full.subprocess, "run", run)
    result = full.main([*arguments(tmp_path), "--auto-batch"])
    assert invoked == expected_batches
    assert result == return_codes[-1]
    for batch in invoked:
        assert full.BUDGET[batch] * batch == 80000
        local, durable = full.roots(full.parse([*arguments(tmp_path), "--batch-size", str(batch)]))
        assert local.name == durable.name == f"ir2r-b{batch}"


def test_resume_does_not_fall_back_or_start_fresh(tmp_path):
    with pytest.raises(SystemExit):
        full.parse([*arguments(tmp_path), "--auto-batch", "--require-resume"])
