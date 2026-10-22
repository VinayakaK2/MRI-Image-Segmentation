"""Cache provenance tracking and cryptographic validation for evaluation artifacts."""

import hashlib
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

CACHE_SCHEMA_VERSION = "1.0.0"


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes the SHA-256 cryptographic hexadecimal digest of a file in streaming chunks."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {filepath}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_cache_provenance(
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    manifest_path: Union[str, Path],
    schema_version: str = CACHE_SCHEMA_VERSION,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Constructs a cryptographic provenance record for evaluation cache artifacts."""
    ckpt_p = Path(checkpoint_path)
    cfg_p = Path(config_path)
    man_p = Path(manifest_path)

    provenance = {
        "schema_version": str(schema_version),
        "checkpoint_path": str(ckpt_p.as_posix()),
        "checkpoint_sha256": compute_file_sha256(ckpt_p),
        "config_path": str(cfg_p.as_posix()),
        "config_sha256": compute_file_sha256(cfg_p),
        "manifest_path": str(man_p.as_posix()),
        "manifest_sha256": compute_file_sha256(man_p),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if extra_metadata:
        provenance["metadata"] = extra_metadata

    return provenance


def validate_cache_provenance(
    cached_data: Any,
    expected_provenance: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Validates cached evaluation data against expected cryptographic provenance.

    Returns:
        (True, None) if provenance is present and all cryptographic hashes match.
        (False, reason_string) if provenance is missing, invalid, or mismatched.
    """
    if not isinstance(cached_data, dict):
        return False, "Cached data is not a dictionary"

    cached_prov = cached_data.get("provenance")
    if not cached_prov or not isinstance(cached_prov, dict):
        return False, "Missing 'provenance' metadata section in cache"

    # 1. Schema version check
    cached_version = cached_prov.get("schema_version")
    expected_version = expected_provenance.get("schema_version")
    if cached_version != expected_version:
        return False, f"Schema version mismatch: cached '{cached_version}' vs expected '{expected_version}'"

    # 2. Checkpoint SHA-256 check
    cached_ckpt = cached_prov.get("checkpoint_sha256")
    expected_ckpt = expected_provenance.get("checkpoint_sha256")
    if cached_ckpt != expected_ckpt:
        return False, f"Checkpoint SHA-256 mismatch: cached '{cached_ckpt}' vs expected '{expected_ckpt}'"

    # 3. Config SHA-256 check
    cached_cfg = cached_prov.get("config_sha256")
    expected_cfg = expected_provenance.get("config_sha256")
    if cached_cfg != expected_cfg:
        return False, f"Config SHA-256 mismatch: cached '{cached_cfg}' vs expected '{expected_cfg}'"

    # 4. Manifest SHA-256 check
    cached_man = cached_prov.get("manifest_sha256")
    expected_man = expected_provenance.get("manifest_sha256")
    if cached_man != expected_man:
        return False, f"Manifest SHA-256 mismatch: cached '{cached_man}' vs expected '{expected_man}'"

    return True, None


def resolve_cache_state(
    cache_path: Union[str, Path],
    expected_provenance: Dict[str, Any],
    force_recompute: bool = False,
) -> Tuple[Dict[str, Any], bool, str]:
    """Resolves existing cache state against expected cryptographic provenance.

    Args:
        cache_path: Path to the JSON cache file.
        expected_provenance: The authoritative provenance dictionary for the current run.
        force_recompute: If True, bypasses and invalidates any existing cache.

    Returns:
        A tuple of (cache_data_dict, is_reused_flag, diagnostic_message).
    """
    import json

    fresh_store = {"provenance": expected_provenance}

    if force_recompute:
        return fresh_store, False, "Forced recompute requested; existing cache bypassed."

    p = Path(cache_path)
    if not p.is_file():
        return fresh_store, False, f"Cache file not found at {p}; initializing clean state."

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as err:
        return fresh_store, False, f"Failed to load cache JSON ({err}); initializing clean state."

    is_valid, reason = validate_cache_provenance(data, expected_provenance)
    if not is_valid:
        return fresh_store, False, f"Cache provenance invalid ({reason}); recomputing all conditions."

    # Cache is valid and matches all cryptographic criteria
    return data, True, "Cache validated successfully against cryptographic provenance; reusing valid entries."
