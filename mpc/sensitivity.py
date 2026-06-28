import numpy as np
import casadi as ca
from mpc.ocp import solve as ocp_solve

def compute_jacobian(kkt_data: dict) -> np.ndarray:
    """
    Compute ∂u_0*/∂[w_j, w_s, w_a] via KKT implicit differentiation.

    At the optimal KKT point the stationarity condition holds:
        ∇_x L(x*, λ*, p) = 0

    Differentiating with respect to p gives the linear system:
        (∂²L/∂x²) · (∂x*/∂p) = -(∂²L/∂x∂p)
        ∂x*/∂p = -H⁻¹ · K

    where H = ∂²L/∂x² and K = ∂²L/∂x∂p are evaluated at (x*, λ*, p).

    Returns shape (1, 3).
    """

    solver = kkt_data["solver"]
    weights = kkt_data["weights"]
    state = kkt_data["state"]
    context = kkt_data["context"]
    x_opt = kkt_data["x_opt"]
    lam_g = kkt_data["lam_g"]
    lam_x = kkt_data["lam_x"]

    p_val = np.array([
        weights[0], weights[1], weights[2],
        state["s0"], state["v0"],
        context["v_setpoint"], context["v_corner"],
    ])

    # Build H and K evaluation functions once and cache on the solver object.
    # The symbolic expressions were stored on the solver by build_ocp().
    if not hasattr(solver, "_H_fn"):
        x_sym   = solver._x_sym
        p_sym   = solver._p_sym
        f_sym   = solver._f_sym
        g_sym   = solver._g_sym

        n_dec = x_sym.shape[0]
        n_g   = g_sym.shape[0]

        lam_g_sym = ca.MX.sym("lam_g", n_g)    # type: ignore[arg-type]
        lam_x_sym = ca.MX.sym("lam_x", n_dec)  # type: ignore[arg-type]

        # Lagrangian: L = f + λ_g' g + λ_x' x
        # (box-constraint duals enter as λ_x' x because lbx ≤ x ≤ ubx)
        L = f_sym + ca.dot(lam_g_sym, g_sym) + ca.dot(lam_x_sym, x_sym)

        grad_L = ca.gradient(L, x_sym)          # ∂L/∂x, shape (n_dec,)
        H = ca.jacobian(grad_L, x_sym)          # ∂²L/∂x²,  shape (n_dec, n_dec)
        K = ca.jacobian(grad_L, p_sym)          # ∂²L/∂x∂p, shape (n_dec, n_p)

        solver._H_fn = ca.Function("H", [x_sym, p_sym, lam_g_sym, lam_x_sym], [H])  # type: ignore[call-arg]
        solver._K_fn = ca.Function("K", [x_sym, p_sym, lam_g_sym, lam_x_sym], [K])  # type: ignore[call-arg]

    # Evaluate H and K at the current KKT point (x*, λ_g*, λ_x*, p)
    H_val = np.array(solver._H_fn(x_opt, p_val, lam_g, lam_x))
    K_val = np.array(solver._K_fn(x_opt, p_val, lam_g, lam_x))

    # Solve H · (∂x*/∂p) = -K.
    # H is positive-definite here: objective is strongly convex in x,
    # and the constraint Jacobians are linear so ∂²g/∂x² = 0.
    dxdp = np.linalg.solve(H_val, -K_val)   # shape (n_dec, n_p)

    return dxdp[0:1, 0:3]  # shape (1, 3): ∂u_0*/∂[w_j, w_s, w_a]


def fd_check(solver,
             weights: np.ndarray,
             state: dict,
             context: dict,
             eps: float = 1e-4) -> np.ndarray:
    
    """
    Finite-difference Jacobian ∂u_0*/∂weights. Shape (1, 3).

    Used to validate compute_jacobian(). Not used in the training loop.
    """

    J = np.zeros((1, 3))
    for i in range(3):
        w_plus = weights.copy()
        w_plus[i] += eps
        u_plus, _ = ocp_solve(solver, w_plus, state, context)

        w_minus = weights.copy()
        w_minus[i] -= eps
        u_minus, _ = ocp_solve(solver, w_minus, state, context)

        J[0, i] = (u_plus - u_minus) / (2 * eps)

    return J

def assert_jacobian_valid(kkt_data: dict,
                          tol: float=1e-3) -> None:
    """
    Assert KKT Jacobian matches FD. Call when DEBUG_GRAD=1.
    """

    J_kkt = compute_jacobian(kkt_data)
    J_fd = fd_check(
        kkt_data["solver"],
        kkt_data["weights"],
        kkt_data["state"],
        kkt_data["context"]
    )

    err = np.max(np.abs(J_kkt - J_fd))
    assert err < tol, f"Jacobian mismatch: max error {err:.2e} > {tol:.2e}"
