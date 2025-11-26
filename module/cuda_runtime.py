"""
Embedded CUDA runtime helper.

Adds bundled CUDA/cuDNN DLL (or .so/.dylib) directories to the process search path
so that torch / faster-whisper can run without system-wide CUDA installs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

_REGISTERED = False

# Official zlibwapi download (linked from NVIDIA docs)
def check_zlib_exists(dest_dir: Path):
    """Check that zlibwapi.dll exists in libs; warn if missing."""
    zlib_path = dest_dir / "zlibwapi.dll"
    if not zlib_path.exists():
        print("WARNING: zlibwapi.dll not found in libs/. Please add it to enable CUDA acceleration on Windows.")

def get_nvidia_package_paths() -> List[Path]:
    """Find installed nvidia pip package library paths."""
    paths = []
    try:
        import nvidia.cudnn
        import nvidia.cublas
        
        # Add bin/lib directories from the packages
        for module in [nvidia.cudnn, nvidia.cublas]:
            base_path = Path(module.__file__).parent
            # Check for bin (Windows) or lib (Linux)
            if (base_path / "bin").exists():
                paths.append(base_path / "bin")
            if (base_path / "lib").exists():
                paths.append(base_path / "lib")
            # Sometimes DLLs are in the root of the package
            paths.append(base_path)
            
    except ImportError:
        print("Warning: nvidia-cudnn-cu12 or nvidia-cublas-cu12 not installed via pip.")
        print("Please run: pip install -r requirements.txt")
        
    return paths

def ensure_libs_exist():
    """Ensure zlibwapi exists."""
    libs_dir = PROJECT_ROOT / "libs"
    libs_dir.mkdir(exist_ok=True)
    
    # Ensure zlibwapi is present; it should be committed into repo (libs/zlibwapi.dll)
    check_zlib_exists(libs_dir)

def _unique_dirs(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        try:
            real = path.resolve()
        except Exception:
            real = path
        if real in seen:
            continue
        seen.add(real)
        unique.append(real)
    return unique


def register_embedded_cuda() -> None:
    """
    Prepend embedded CUDA runtime directories (./libs, ./module/libs) to PATH/LD_LIBRARY_PATH.
    Idempotent: safe to call multiple times.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Ensure libraries are present before adding to path
    if os.name == "nt":
        ensure_libs_exist()

    # Get paths from pip packages + local libs
    pip_lib_paths = get_nvidia_package_paths()
    candidates = _unique_dirs(pip_lib_paths + [PROJECT_ROOT / "libs", MODULE_DIR / "libs"])

    existing_path = os.environ.get("PATH", "")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_path_entries: List[str] = []
    new_ld_entries: List[str] = []

    for lib_dir in candidates:
        if not lib_dir.is_dir():
            continue

        # 리눅스/맥에서는 .dll만 있는 폴더를 PATH에 넣지 않아 CUDA 초기화 충돌을 방지
        if os.name != "nt":
            has_native_lib = any(
                lib_dir.glob("*.so*")
            ) or any(lib_dir.glob("*.dylib*"))
            if not has_native_lib:
                continue

        lib_str = str(lib_dir)
        new_path_entries.append(lib_str)
        new_ld_entries.append(lib_str)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_str)  # type: ignore[attr-defined]
            except Exception:
                pass

    if new_path_entries:
        combined = os.pathsep.join(new_path_entries + [existing_path]) if existing_path else os.pathsep.join(new_path_entries)
        os.environ["PATH"] = combined

    if new_ld_entries:
        combined_ld = os.pathsep.join(new_ld_entries + [existing_ld]) if existing_ld else os.pathsep.join(new_ld_entries)
        os.environ["LD_LIBRARY_PATH"] = combined_ld

    _REGISTERED = True
