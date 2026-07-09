from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


def _default_vendor() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists():
        return root / "vendor" / "python"
    import os as _os
    home = Path(_os.environ.get("CC_BRAIN_HOME", "~/.cc-brain")).expanduser()
    return home / "vendor" / "python"


CUDA_WHEELS = (
    "onnxruntime-gpu",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu12",
)


@dataclass(frozen=True)
class RuntimeStatus:
    device_mode: str
    gpu_detected: bool
    vendor_dir: str
    vendor_present: bool
    providers_requested: list[str]
    dll_dirs: list[str]


def device_mode() -> str:
    if os.environ.get("CC_BRAIN_CPU"):
        return "cpu"
    mode = os.environ.get("CC_BRAIN_DEVICE", "auto").strip().lower()
    return mode if mode in ("auto", "gpu", "cpu") else "auto"


def vendor_dir() -> Path:
    return Path(os.environ.get("CC_BRAIN_VENDOR_DIR", str(_default_vendor()))).expanduser().resolve()


def has_nvidia_gpu() -> bool:
    if os.environ.get("CC_BRAIN_ASSUME_GPU") == "1":
        return True
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except (OSError, subprocess.TimeoutExpired):
        return False


def _candidate_roots() -> list[Path]:
    roots = [vendor_dir()]
    purelib = sysconfig.get_paths().get("purelib", "")
    if purelib:
        roots.append(Path(purelib))
    return roots


def _dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in _candidate_roots():
        if root.exists():
            dirs.extend(Path(p) for p in glob.glob(str(root / "nvidia" / "*" / "bin")))
            dirs.extend(Path(p) for p in glob.glob(str(root / "nvidia" / "*" / "lib")))
            dirs.extend(Path(p) for p in glob.glob(str(root / "onnxruntime" / "capi")))
    return [d for d in dirs if d.exists()]


def add_vendor_to_pythonpath() -> None:
    v = vendor_dir()
    if v.exists() and str(v) not in sys.path:
        sys.path.insert(0, str(v))


def preload_dlls() -> list[str]:
    if os.name != "nt":
        return []
    add_vendor_to_pythonpath()
    loaded_dirs: list[str] = []
    for dll_dir in _dll_dirs():
        try:
            os.add_dll_directory(str(dll_dir))
        except (FileNotFoundError, OSError):
            pass
        loaded_dirs.append(str(dll_dir))
        for file in sorted(dll_dir.iterdir()):
            if file.suffix.lower() == ".dll":
                try:
                    ctypes.WinDLL(str(file))
                except OSError:
                    pass
    return loaded_dirs


def _can_load_cuda_probe() -> bool:
    if os.name != "nt":
        return True
    preload_dlls()
    for dll in ("cublasLt64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return False
    return True


def vendor_gpu_runtime() -> Path:
    """Install CUDA runtime wheels into repo-local vendor/python.

    This vendors redistributable NVIDIA Python wheels locally. It does not commit
    binaries to git; each clone can self-bootstrap on first GPU use.
    """
    v = vendor_dir()
    v.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(v),
        *CUDA_WHEELS,
    ]
    subprocess.run(cmd, check=True)
    add_vendor_to_pythonpath()
    preload_dlls()
    return v


def ensure_gpu_runtime() -> list[str]:
    """Prepare GPU DLLs when GPU mode is requested or auto-detected.

    `CC_BRAIN_AUTO_VENDOR=0` disables network bootstrap. `CC_BRAIN_DEVICE=gpu`
    stays strict and raises if CUDA DLLs cannot be prepared.
    """
    mode = device_mode()
    if mode == "cpu":
        return []
    if mode == "auto" and not has_nvidia_gpu():
        return []
    preload_dlls()
    # On GPU machines, prefer a repo-local GPU runtime so clones are
    # self-contained and do not depend on fragile system PATH state.
    if not vendor_dir().exists() and os.environ.get("CC_BRAIN_AUTO_VENDOR", "1") != "0":
        vendor_gpu_runtime()
    if _can_load_cuda_probe():
        return [str(d) for d in _dll_dirs()]
    if os.environ.get("CC_BRAIN_AUTO_VENDOR", "1") != "0":
        vendor_gpu_runtime()
        if _can_load_cuda_probe():
            return [str(d) for d in _dll_dirs()]
    if mode == "gpu":
        raise RuntimeError(
            "CC_BRAIN_DEVICE=gpu requested, but CUDA/cuDNN/cuBLAS DLLs could not be loaded. "
            "Run `cc-brain bootstrap-gpu` or set CC_BRAIN_DEVICE=cpu."
        )
    return [str(d) for d in _dll_dirs()]


def requested_providers() -> list[str]:
    mode = device_mode()
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    ensure_gpu_runtime()
    if mode == "gpu":
        return ["CUDAExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def status() -> RuntimeStatus:
    dirs = [] if device_mode() == "cpu" else preload_dlls()
    providers = ["CPUExecutionProvider"] if device_mode() == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return RuntimeStatus(
        device_mode=device_mode(),
        gpu_detected=has_nvidia_gpu(),
        vendor_dir=str(vendor_dir()),
        vendor_present=vendor_dir().exists(),
        providers_requested=providers,
        dll_dirs=dirs,
    )
