import os
import torch
import random
import numpy as np


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Add this line
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # MPS lacks deterministic implementations for several ops (e.g.
    # index_put_with_accumulate). warn_only lets training proceed while
    # logging a warning for ops without a deterministic kernel, instead of
    # raising a RuntimeError. CUDA and CPU have full coverage.
    _warn_only = torch.backends.mps.is_available() and not torch.cuda.is_available()
    torch.use_deterministic_algorithms(True, warn_only=_warn_only)
    torch.backends.cudnn.allow_tf32 = False  # Disable TensorFloat32 for exact FP32 math
    torch.backends.cuda.matmul.allow_tf32 = False
