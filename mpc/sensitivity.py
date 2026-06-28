import os
import numpy as np
import casadi as ca
from mpc.ocp import solve as ocp_solve

def compute_jacobian(kkt_data: dict) -> np.ndarray:
    """
    Compute ∂u_0*/∂[w_j, w_s, w_a] via CasADi parametric NLP sensitivity.

    Uses the KKT system: at optimality, the sensitivity of primal variables
    w.r.t. parameters is obtained by differentiating the stationarity conditions.
    CasADi's nlpsol exposes this via the 'sens' factory.

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

    # Build the sensitivity function once and cache it on the solver object.
    # solver.factory() derives a new CasADi function from the NLP's internal
    # symbolic graph — the construction is expensive, but the resulting fn is cheap to call.
    if not hasattr(solver, "_sens_fn"):
        sens = solver.factory(
            "sens",                            # arbitrary label for this derived fn
            ["x0", "p", "lam_x0", "lam_g0"],  # inputs: full KKT point (primals + duals)
            ["jac:x:p"],                       # output: ∂x*/∂p via implicit diff of KKT
        )
        # lam_x0 / lam_g0 (dual variables) are required inputs because the sensitivity
        # formula  ∂x*/∂p = -(∇²_xx L)⁻¹ · ∇²_xp L  depends on the Hessian of the
        # Lagrangian ∇²_xx L, which itself depends on λ*. Without them CasADi cannot
        # evaluate the linear system.
        solver._sens_fn = sens

    # Evaluate sensitivity at the current KKT point.
    # "jac:x:p" returns a dense matrix of shape (n_dec, n_params=7).
    res = solver._sens_fn(
        x0=x_opt,
        p=p_val,
        lam_x0=lam_x,
        lam_g0=lam_g,
    )
    
    # We want row 0 (u_0), columns 0:3 (w_j, w_s, w_a)
    jac_full = np.array(res["jac_x_p"])
    return jac_full[0:1, 0:3] # shape (1,3)


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
