#!/usr/bin/env python3
"""Environment diagnostics and hardware verification script for Cardiac MRI project."""

import os
import platform
import sys
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def check_python_environment() -> Dict[str, Any]:
    """Collects Python runtime and interpreter information."""
    return {
        "version": sys.version.replace("\n", " "),
        "version_info": tuple(sys.version_info),
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }


def check_pytorch_and_cuda() -> Dict[str, Any]:
    """Checks PyTorch installation, CUDA availability, and device properties."""
    info: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch
        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            info["device_count"] = count
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        else:
            info["cpu_fallback"] = True
    except ImportError:
        pass

    return info


def check_dependencies() -> Dict[str, Dict[str, Any]]:
    """Checks the availability and version of key project dependencies."""
    packages = [
        ("numpy", "Core numerical computing"),
        ("nibabel", "Neuroimaging/NIfTI I/O"),
        ("scipy", "Scientific computing & signal processing"),
        ("matplotlib", "Plotting & visual evaluation"),
        ("torchvision", "Vision utilities"),
        ("yaml", "PyYAML configuration loader"),
        ("tqdm", "Progress bars"),
        ("pytest", "Testing framework"),
        ("pandas", "DataFrames & tabular metrics"),
        ("sklearn", "Scikit-learn splitting & metrics"),
        ("skimage", "Scikit-image morphology & processing"),
        ("monai", "Medical Open Network for AI (optional)"),
    ]

    results = {}
    for pkg_name, description in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "available")
            results[pkg_name] = {
                "available": True,
                "version": ver,
                "description": description,
            }
        except ImportError:
            results[pkg_name] = {
                "available": False,
                "version": None,
                "description": description,
            }
    return results


def check_directories_and_dataset() -> Dict[str, Any]:
    """Validates project folder structure and dataset configuration status."""
    root = get_project_root()
    expected_dirs = [
        "configs",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/corruptions",
        "src/preprocessing",
        "src/postprocessing",
        "src/utils",
        "tests",
        "scripts",
        "checkpoints",
        "experiments",
        "reports",
        "data/raw",
        "data/processed",
    ]

    dir_status = {}
    for d in expected_dirs:
        dir_path = root / d
        dir_status[d] = {
            "exists": dir_path.exists(),
            "is_dir": dir_path.is_dir() if dir_path.exists() else False,
        }

    # Check dataset path configuration
    config_file = root / "configs" / "baseline_unet.yaml"
    dataset_status = {
        "config_found": config_file.exists(),
        "raw_dir_configured": None,
        "raw_dir_exists": False,
        "raw_file_count": 0,
        "sample_files": [],
    }

    if config_file.exists():
        try:
            import yaml
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            raw_rel = cfg.get("dataset", {}).get("raw_dir", "data/raw/ACDC")
            dataset_status["raw_dir_configured"] = raw_rel
            raw_path = root / raw_rel
            if raw_path.exists() and raw_path.is_dir():
                dataset_status["raw_dir_exists"] = True
                files = list(raw_path.glob("*"))
                # Filter out .gitkeep
                data_files = [f.name for f in files if f.name != ".gitkeep"]
                dataset_status["raw_file_count"] = len(data_files)
                dataset_status["sample_files"] = data_files[:5]
        except Exception as e:
            dataset_status["config_read_error"] = str(e)

    return {
        "root": str(root),
        "directories": dir_status,
        "dataset": dataset_status,
    }


def print_diagnostic_report() -> None:
    """Prints a structured, human-readable environment diagnostic report."""
    print("=" * 70)
    print("      CARDIAC MRI ROBUSTNESS PROJECT - ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    # 1. Python Environment
    py = check_python_environment()
    print("\n[1] Python Environment:")
    print(f"  - Version:      {py['version']}")
    print(f"  - Executable:   {py['executable']}")
    print(f"  - OS Platform:  {py['platform']} ({py['architecture']})")

    # 2. PyTorch & Hardware
    torch_info = check_pytorch_and_cuda()
    print("\n[2] PyTorch & Hardware Acceleration:")
    if torch_info["installed"]:
        print(f"  - PyTorch Version:   {torch_info['version']}")
        print(f"  - CUDA Available:    {torch_info['cuda_available']}")
        if torch_info["cuda_available"]:
            print(f"  - GPU Devices Count: {torch_info['device_count']}")
            for d in torch_info["devices"]:
                print(f"    * Device {d['index']}: {d['name']} | VRAM: {d['total_memory_gb']} GB | SMs: {d['multi_processor_count']} | Capability: {d['compute_capability']}")
        else:
            print("  - Hardware Device:   CPU execution mode active (No CUDA device found)")
    else:
        print("  - PyTorch: NOT INSTALLED")

    # 3. Dependencies
    deps = check_dependencies()
    print("\n[3] Dependency Availability:")
    for name, d in deps.items():
        status_icon = "[OK]  " if d["available"] else "[MISS]"
        ver_str = d["version"] if d["available"] else "Not installed"
        print(f"  {status_icon} {name:<12} : {ver_str:<15} ({d['description']})")

    # 4. Project Paths & Dataset
    dir_info = check_directories_and_dataset()
    print("\n[4] Project Paths & Structure:")
    print(f"  - Project Root: {dir_info['root']}")
    all_dirs_ok = all(s["exists"] for s in dir_info["directories"].values())
    print(f"  - Core Directories Status: {'All present (OK)' if all_dirs_ok else 'Some missing'}")

    ds = dir_info["dataset"]
    print("\n[5] Dataset Configuration Status:")
    print(f"  - Config File Found:       {ds['config_found']}")
    print(f"  - Configured Raw Data Dir: {ds['raw_dir_configured']}")
    print(f"  - Raw Data Dir Exists:     {ds['raw_dir_exists']}")
    print(f"  - Raw Data File Count:     {ds['raw_file_count']}")
    if ds["raw_file_count"] == 0:
        print("  - Notice: Raw ACDC dataset has not been populated yet.")
        print("            Expected in Phase 1 (NIfTI patient archives).")
    else:
        print(f"  - Sample Files:            {', '.join(ds['sample_files'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_diagnostic_report()
