import numpy as np
import pytest
from mpc.ocp import build_ocp, solve
from mpc.sensitivity import compute_jacobian, fd_check

@pytest.fixture(scope="module")
def solver():
    return build_ocp()

def test_jacobian_shape(solver):
    weights = np.array([1.0, 1.0, 1.0])
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    _, kkt_data = solve(solver, weights, state, context)
    J = compute_jacobian(kkt_data)
    assert J.shape == (1, 3)

def test_kkt_matches_fd(solver):
    weights = np.array([1.0, 1.0, 1.0])
    state = {"s0": 0.0, "v0": 80.0}
    context = {"v_setpoint": 80.0, "v_corner": 55.0}
    _, kkt_data = solve(solver, weights, state, context)
    J_kkt = compute_jacobian(kkt_data)
    J_fd = fd_check(solver, weights, state, context)
    np.testing.assert_allclose(J_kkt, J_fd, atol=1e-3,
                                err_msg="KKT Jacobian diverges from finite-difference")