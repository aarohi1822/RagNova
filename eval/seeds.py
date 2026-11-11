"""Fix every RNG the evaluation touches."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 0


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)  # also seeds CUDA/MPS generators
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass
