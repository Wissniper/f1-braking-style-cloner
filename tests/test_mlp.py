import torch
import pytest
from model.mlp import CostWeightMLP

def test_output_shape():
    mlp = CostWeightMLP()
    x = torch.tensor([0.8, 0.5, 0.2])
    out = mlp(x)
    assert out.shape == (3,)

def test_output_in_valid_range():
    mlp = CostWeightMLP()
    x = torch.randn(100, 3)
    out = mlp(x)
    assert (out >= 1e-3).all()
    assert (out <= 100.0).all()

def test_output_requires_grad():
    mlp = CostWeightMLP()
    x = torch.tensor([0.8, 0.5, 0.2])
    out = mlp(x)
    assert out.requires_grad

def test_parameter_count():
    mlp = CostWeightMLP()
    n_params = sum(p.numel() for p in mlp.parameters())
    assert n_params < 5000  # should be ~2000