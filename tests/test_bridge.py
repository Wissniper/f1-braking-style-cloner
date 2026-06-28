import numpy as np
import torch
import pytest
from mpc.ocp import build_ocp
from model.mpc_layer import MPCSolve, verify_bridge

@pytest.fixture(scope="module")
def solver():
    return build_ocp()

def test_forward_returns_scalar(solver):
    weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    u = MPCSolve.apply(weights, solver, state, context, True)
    assert u.shape == ()
    assert u.item() >= -50.0 and u.item() <= 10.0

def test_backward_runs(solver):
    weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    u = MPCSolve.apply(weights, solver, state, context, True)
    loss = (u - torch.tensor(-5.0, dtype=torch.float64)) ** 2
    loss.backward()
    assert weights.grad is not None
    assert weights.grad.shape == (3,)

def test_verify_bridge_passes(solver):
    verify_bridge(solver)  # raises AssertionError if gradcheck fails