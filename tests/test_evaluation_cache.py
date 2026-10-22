"""Unit tests for evaluation cache provenance and validation."""

import json
from pathlib import Path
import pytest

from src.utils.provenance import (
    CACHE_SCHEMA_VERSION,
    build_cache_provenance,
    compute_file_sha256,
    resolve_cache_state,
    validate_cache_provenance,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verifies that compute_file_sha256 accurately hashes file contents and detects missing files."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("hello mri world", encoding="utf-8")

    h = compute_file_sha256(test_file)
    assert isinstance(h, str)
    assert len(h) == 64

    # Different content produces different hash
    test_file2 = tmp_path / "sample2.txt"
    test_file2.write_text("hello mri world 2", encoding="utf-8")
    assert compute_file_sha256(test_file2) != h

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "does_not_exist.txt")


def test_build_and_validate_provenance_success(tmp_path: Path) -> None:
    """Verifies that matching provenance records validate successfully."""
    ckpt = tmp_path / "model.pt"
    cfg = tmp_path / "config.yaml"
    man = tmp_path / "manifest.json"

    ckpt.write_bytes(b"model_weights_123")
    cfg.write_text("in_channels: 1\nnum_classes: 4", encoding="utf-8")
    man.write_text("[]", encoding="utf-8")

    prov = build_cache_provenance(ckpt, cfg, man, schema_version=CACHE_SCHEMA_VERSION)
    cache_payload = {
        "provenance": prov,
        "results": {"dummy": 1},
    }

    is_valid, reason = validate_cache_provenance(cache_payload, prov)
    assert is_valid is True
    assert reason is None


def test_validate_provenance_missing_provenance(tmp_path: Path) -> None:
    """Verifies that cached data lacking a provenance section is rejected."""
    ckpt = tmp_path / "model.pt"
    cfg = tmp_path / "config.yaml"
    man = tmp_path / "manifest.json"
    ckpt.write_bytes(b"w")
    cfg.write_text("c", encoding="utf-8")
    man.write_text("m", encoding="utf-8")

    prov = build_cache_provenance(ckpt, cfg, man)

    # Missing entirely
    is_valid, reason = validate_cache_provenance({"results": {}}, prov)
    assert is_valid is False
    assert "Missing 'provenance'" in str(reason)

    # Non-dict cached data
    is_valid, reason = validate_cache_provenance(["not", "a", "dict"], prov)
    assert is_valid is False
    assert "not a dictionary" in str(reason)


def test_validate_provenance_mismatches(tmp_path: Path) -> None:
    """Verifies that mismatches in checkpoint, config, manifest, or schema version invalidate the cache."""
    ckpt = tmp_path / "model.pt"
    cfg = tmp_path / "config.yaml"
    man = tmp_path / "manifest.json"
    ckpt.write_bytes(b"w")
    cfg.write_text("c", encoding="utf-8")
    man.write_text("m", encoding="utf-8")

    expected_prov = build_cache_provenance(ckpt, cfg, man, schema_version="1.0.0")

    # 1. Checkpoint mismatch
    tampered_ckpt_prov = dict(expected_prov)
    tampered_ckpt_prov["checkpoint_sha256"] = "0" * 64
    is_valid, reason = validate_cache_provenance({"provenance": tampered_ckpt_prov}, expected_prov)
    assert is_valid is False
    assert "Checkpoint SHA-256 mismatch" in str(reason)

    # 2. Config mismatch
    tampered_cfg_prov = dict(expected_prov)
    tampered_cfg_prov["config_sha256"] = "1" * 64
    is_valid, reason = validate_cache_provenance({"provenance": tampered_cfg_prov}, expected_prov)
    assert is_valid is False
    assert "Config SHA-256 mismatch" in str(reason)

    # 3. Manifest mismatch
    tampered_man_prov = dict(expected_prov)
    tampered_man_prov["manifest_sha256"] = "2" * 64
    is_valid, reason = validate_cache_provenance({"provenance": tampered_man_prov}, expected_prov)
    assert is_valid is False
    assert "Manifest SHA-256 mismatch" in str(reason)

    # 4. Schema version mismatch
    tampered_ver_prov = dict(expected_prov)
    tampered_ver_prov["schema_version"] = "99.0.0"
    is_valid, reason = validate_cache_provenance({"provenance": tampered_ver_prov}, expected_prov)
    assert is_valid is False
    assert "Schema version mismatch" in str(reason)


def test_resolve_cache_state(tmp_path: Path) -> None:
    """Verifies cache state resolution under valid reuse, force recomputation, and invalid states."""
    ckpt = tmp_path / "model.pt"
    cfg = tmp_path / "config.yaml"
    man = tmp_path / "manifest.json"
    ckpt.write_bytes(b"w")
    cfg.write_text("c", encoding="utf-8")
    man.write_text("m", encoding="utf-8")

    prov = build_cache_provenance(ckpt, cfg, man, schema_version="1.0.0")
    cache_file = tmp_path / "mitigation_results.json"

    # 1. Cache file does not exist
    store, is_reused, msg = resolve_cache_state(cache_file, prov, force_recompute=False)
    assert is_reused is False
    assert store == {"provenance": prov}
    assert "not found" in msg

    # Write a valid cache
    valid_payload = {
        "provenance": prov,
        "postproc_alone": {"condition_1": {"mean": 0.85}},
    }
    cache_file.write_text(json.dumps(valid_payload), encoding="utf-8")

    # 2. Valid cache reuse
    store, is_reused, msg = resolve_cache_state(cache_file, prov, force_recompute=False)
    assert is_reused is True
    assert "postproc_alone" in store
    assert "reusing valid entries" in msg

    # 3. Force recompute overrides valid cache
    store, is_reused, msg = resolve_cache_state(cache_file, prov, force_recompute=True)
    assert is_reused is False
    assert "postproc_alone" not in store
    assert store == {"provenance": prov}
    assert "Forced recompute requested" in msg

    # 4. Invalidation on tampered checkpoint hash
    stale_prov = dict(prov)
    stale_prov["checkpoint_sha256"] = "abcde" * 12 + "1234"
    cache_file.write_text(json.dumps({"provenance": stale_prov, "postproc_alone": {}}), encoding="utf-8")
    store, is_reused, msg = resolve_cache_state(cache_file, prov, force_recompute=False)
    assert is_reused is False
    assert "recomputing all conditions" in msg
    assert store == {"provenance": prov}
