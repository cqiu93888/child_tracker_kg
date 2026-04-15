# 在「同一行程內同時使用 PyTorch（YOLO）與 dlib／numpy」時，降低 Windows 上 OpenMP/MKL 衝突導致無訊息閃退的機率。
# 須在 import ultralytics / torch 之前呼叫 apply()。
import os
import sys


def apply():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if sys.platform != "win32":
        return
    # 多套件各帶一套 OpenMP 時，排擠執行緒常可避開 0xC0000005
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def limit_torch_threads(n: int = 1):
    try:
        import torch

        n = max(1, int(n))
        torch.set_num_threads(n)
    except Exception:
        pass


def prepare_cuda_hidden_for_cpu_only_yolo():
    """在 import torch 之前呼叫：強制看不到 GPU，避免 CPU 模式仍載入 CUDA 驅動而崩潰。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def torch_mitigations_for_mixed_dlib():
    """載入 torch 後呼叫：降低與 dlib/numpy MKL 一齊使用時的不穩定。"""
    try:
        import torch

        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
    except Exception:
        pass
