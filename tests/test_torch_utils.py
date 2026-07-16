"""model_training.torch_utils._to_cpu — detach/clone a tensor onto the CPU."""
import numpy as np
import torch

from model_training.torch_utils import _to_cpu


def test_to_cpu_returns_detached_cpu_tensor():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = _to_cpu(t)
    assert not out.requires_grad
    assert out.device.type == "cpu"
    assert np.allclose(out.numpy(), [1.0, 2.0, 3.0])


def test_to_cpu_clones_by_default():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = _to_cpu(t)          # clone=True -> independent storage
    out[0] = 99.0
    assert t[0].item() == 1.0  # original untouched
