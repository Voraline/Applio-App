import os
import torch

_ext = None
_load_attempted = False
_cpp_available: bool | None = None
_pv_path_reported = False


def _load():
    global _ext, _load_attempted, _cpp_available
    if _load_attempted:
        return _ext
    _load_attempted = True

    _dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(_dir, "pv_extension.cpp")
    cu_src  = os.path.join(_dir, "pv_kernel.cu")

    sources = [cpp_src]
    if os.name == "nt":
        extra_cflags = ["/O2", "/fp:fast"]
    else:
        extra_cflags = ["-O3", "-ffast-math"]

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-Xptxas",
        "-O3",
        "-Xptxas",
        "-dlcm=ca",
    ]

    if torch.cuda.is_available():
        sources.append(cu_src)

    try:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="pv_ext",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        _cpp_available = True
        print("[Phase Vocoder] C++ extension loaded - using fast C++/CUDA phase vocoder")
    except Exception as e:
        _cpp_available = False
        print(f"[Phase Vocoder] C++ extension build failed - using Python/PyTorch fallback: {e}")
        _ext = None

    return _ext


def is_cpp_available() -> bool:
    _load()
    return bool(_cpp_available)


def phase_vocoder_ext(
    a, b, fade_out_sq, fade_in_sq, window,
    window_over_n, k_grid_delta, t_over_n, inv_2pi, two_pi,
    phia=None, absab=None,
):
    global _pv_path_reported
    ext = _load()

    if not _pv_path_reported:
        _pv_path_reported = True
        if ext is not None:
            print("[Phase Vocoder] Active path: C++/CUDA kernel")
        else:
            print("[Phase Vocoder] Active path: Python/PyTorch fallback")

    if ext is None or phia is None or absab is None:
        return None
    try:
        return ext.phase_vocoder_synth(
            a, b, fade_out_sq, fade_in_sq,
            window_over_n,
            k_grid_delta, t_over_n, phia, absab,
        )
    except Exception:
        return None
