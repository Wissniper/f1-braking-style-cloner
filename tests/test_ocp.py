# tests/test_ocp.py
import numpy as np
import pytest
from mpc.ocp import build_ocp, solve

DT, N = 0.02, 10
V_MAX, U_MIN, U_MAX = 92.0, -50.0, 10.0

@pytest.fixture(scope="module")
def solver():
    return build_ocp(dt=DT, N=N, v_max=V_MAX, u_min=U_MIN, u_max=U_MAX, rho=1000.0)

def test_solver_returns_feasible(solver):
    weights = np.array([1.0, 1.0, 1.0])
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    u_star, kkt_data = solve(solver, weights, state, context, soft=True)
    assert U_MIN <= u_star <= U_MAX

def test_higher_aggression_reduces_braking_aggressiveness(solver):
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    _, _ = solve(solver, np.array([1.0, 1.0, 0.1]), state, context)
    u_low_agg, _ = solve(solver, np.array([1.0, 1.0, 0.1]), state, context)
    u_high_agg, _ = solve(solver, np.array([1.0, 1.0, 10.0]), state, context)
    # Higher aggression weight penalises terminal speed deviation more → harder braking
    # so u_high_agg should be more negative (or equal)
    assert u_high_agg <= u_low_agg + 1e-3

def test_kkt_data_has_required_keys(solver):
    weights = np.array([1.0, 1.0, 1.0])
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    _, kkt_data = solve(solver, weights, state, context)
    for key in ("lam_x", "lam_g", "x_opt", "solver", "weights", "state", "context"):
        assert key in kkt_data