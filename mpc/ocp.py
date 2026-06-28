import numpy as np
import casadi as ca

DT = 0.02 # time step
N = 10 # prediction horizon
V_MAX = 92.0 
U_MIN = -50.0 # minimum acceleration (braking)
U_MAX = 10.0 # maximum acceleration (throttle)
RHO = 1000.0 # soft constraint penalty weight

def build_ocp(dt: float = DT,
              N: int = N,
              v_max: float = V_MAX,
              u_min: float = U_MIN,
              u_max: float = U_MAX,
              rho: float = RHO) -> ca.Function:
    """
        Build CasADi parametric NLP. 
        Call once at startup; reuse solver across all training calls
    """

    """
    Define CasADi Decision variables: controls u_0..u_{N-1} and slack variables (2 per step: lo, hi)

    ┌────────────────────┬─────────────┬────────────┬─────────────────────────────┐
    │       Region       │    Lower    │   Upper    │           Meaning           │
    ├────────────────────┼─────────────┼────────────┼─────────────────────────────┤
    │ u_0..u_{N-1}       │ u_min = -50 │ u_max = 10 │ braking/throttle limits     │
    ├────────────────────┼─────────────┼────────────┼─────────────────────────────┤
    │ slack_lo, slack_hi │ 0.0         │ inf        │ slacks must be non-negative │
    └────────────────────┴─────────────┴────────────┴─────────────────────────────┘
    """
    n_u = N # number of control inputs
    n_slack = 2 * (N + 1) # lower and upper slack for v >= 0 and v <= v_max at each of N+1 timesteps
    n_dec = n_u + n_slack

    # Define placeholder for what IPOPT will decide the optimal values are
    x = ca.MX.sym('x', n_dec, 1) # type: ignore[arg-type]  # CasADi stubs mis-declare sym()

    u_vars  = x[:N] # [u_0..u_{N-1}]
    slack_lo = x[N : N + (N + 1)] # v >= 0 slacks, one per velocity state
    slack_hi = x[N + (N + 1):] # v <= v_max slacks, one per velocity state

    # Parameters: [w_jerk, w_speed, w_aggression, s0, v0, v_setpoint, v_corner]
    p = ca.MX.sym("p", 7, 1) # type: ignore[arg-type]
    w_j, w_s, w_a = p[0], p[1], p[2]
    s0, v0, v_setpoint, v_corner = p[3], p[4], p[5], p[6]

    # simulate the system forward in time
    s_traj = [s0]
    v_traj = [v0]
    for k in range(N):
        s_traj.append(s_traj[-1] + v_traj[-1] * dt)
        v_traj.append(v_traj[-1] + u_vars[k] * dt)
    
    # Objective function: weighted sum of jerk, speed error, and aggression
    J = 0
    for k in range(N):
        J += w_j * u_vars[k]**2 # jerk term
        J += w_s * (v_traj[k] - v_setpoint)**2 # speed error term
    J += w_a * (v_traj[N] - v_corner)**2 # aggression, penalizing overshooting the corner speed at the end of the horizon, because we want to be able to brake in time for the corner

    # Soft constraint penalties
    for k in range(N+1):
        J += rho * (slack_lo[k]**2 + slack_hi[k]**2) # penalize slack variables

    # Constraints: velocity limits with slack
    g = []
    for k in range(N+1):
        g.append(-v_traj[k] - slack_lo[k]) # v <= 0
        g.append(v_traj[k] - v_max - slack_hi[k]) # v <= 0

    # Box constraints on control inputs / Bound vectors for IPOPT
    """
    These decide what values x and the constraints g can take. 
    IPOPT will try to find x such that lbg <= g(x) <= ubg and lbx <= x <= ubx.

    First n_u entries of x are the control inputs u_0..u_{N-1}, which are bounded by u_min and u_max.
    [u_min] * N   →  [-50, -50, ..., -50]   # car can't brake harder than 50 m/s²

    Next 2*(N+1) entries of x are the slack variables, which are bounded by 0 and +inf.
    [0.0] * (2*(N+1)) →  [0, 0, ..., 0]   # slack variables must be non-negative

    Soft constraints:
    v_k >= 0           ← hard, can make problem infeasible
    with:
    v_k + slack_lo[k] >= 0    ← always satisfiable
    slack_lo[k] >= 0
    penalty: rho * slack_lo[k]²

    slack is not allowed to go negative, so IPOPT will try to make slack_lo[k] = 0, 
    which means v_k >= 0. If v_k < 0, then slack_lo[k] > 0 and 
    the penalty term rho * slack_lo[k]² is added to the objective function.
    """
    lbx = [u_min] * n_u + [0.0] * n_slack # lower bounds for u and slack
    ubx = [u_max] * n_u + [ca.inf] * n_slack # upper bounds for u and slack
    lbg = [-ca.inf] * len(g)
    ubg = [0.0] * len(g)

    # Create CasADi NLP solver
    nlp  = {
        'x': x,  # decision variables to optimize
        'p': p, # parameters to the problem
        'f': J, # objective function to minimize, built from x and p
        'g': ca.vertcat(*g) # constraints to satisfy, built from x
    }
    
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-8,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Store metadata on the function object for use in solve()
    solver._lbx = lbx
    solver._ubx = ubx
    solver._lbg = lbg
    solver._ubg = ubg
    solver._n_dec = n_dec
    solver._n_u = n_u

    return solver

def solve(solver: ca.Function,
          weights: np.ndarray,
          state: dict,
          context: dict,
          soft: bool = True) -> tuple[float, dict]:
    
    """
    Run one IPOPT solve. Returns (u_star, kkt_data)
    """

    # Unpack state and context
    p_val = np.array([
        weights[0], weights[1], weights[2], # w_j, w_s, w_a
        state["s0"], state["v0"],
        context["v_setpoint"], context["v_corner"],
    ])

    lbx = list(solver._lbx)
    ubx = list(solver._ubx)

    if not soft:
        # If soft constraints are disabled, set slack upper variable bounds to 0 (forces slacks to 0)
        N = solver._n_u
        for i in range(N, solver._n_dec):
            ubx[i] = 0.0

    """
    The inputs you pass (x0, p, lbx, ubx, lbg, ubg) are what IPOPT needs to solve:

    ┌─────────┬─────────────────────────────────────────────────────────────────────────┐
    │  Input  │                                  Role                                   │
    ├─────────┼─────────────────────────────────────────────────────────────────────────┤
    │ x0      │ initial guess for x (all zeros here)                                    │
    ├─────────┼─────────────────────────────────────────────────────────────────────────┤
    │ p       │ concrete parameter values (w_j, w_s, w_a, s0, v0, v_setpoint, v_corner) │
    ├─────────┼─────────────────────────────────────────────────────────────────────────┤
    │ lbx/ubx │ box bounds on decision variables                                        │
    ├─────────┼─────────────────────────────────────────────────────────────────────────┤
    │ lbg/ubg │ bounds on constraints g                                                 │
    └─────────┴─────────────────────────────────────────────────────────────────────────┘
    
    CasADi's nlpsol always returns a fixed set of keys in sol:

    ┌─────────┬───────────────────────────────────────────────────┐
    │   Key   │                      Content                      │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "x"     │ optimal decision variables                        │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "f"     │ optimal objective value                           │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "g"     │ constraint values at the optimum                  │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "lam_x" │ Lagrange multipliers for box bounds on x          │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "lam_g" │ Lagrange multipliers for constraints g            │
    ├─────────┼───────────────────────────────────────────────────┤
    │ "lam_p" │ Lagrange multipliers for parameters (rarely used) │
    └─────────┴───────────────────────────────────────────────────┘

    lam_x and lam_g are always present regardless of whether your problem has active constraints —
    IPOPT computes them as part of solving the KKT system, so they come for free.
    """

    x0 = np.zeros(solver._n_dec)
    sol = solver(x0=x0,
                 p=p_val,
                 lbx=lbx,
                 ubx=ubx,
                 lbg=solver._lbg,
                 ubg=solver._ubg)
    
    x_opt = np.array(sol["x"]).flatten()  # type: ignore[index]
    u_star = float(x_opt[0]) # u_0, the control applied at this time

    """
    lam_x and lam_g are Lagrange multipliers — IPOPT returns them alongside the optimal x.
    They come from the KKT conditions (first-order optimality conditions for the NLP).

    lam_g: one multiplier per constraint in g (length 2*(N+1), two per velocity state)
        It answers: "how much would the optimal cost change if I tightened this constraint slightly?"
        - lam_g[i] = 0    → constraint i is inactive (not tight), has no effect on the cost
        - lam_g[i] != 0   → constraint i is active (tight), relaxing it would change the optimal cost

    lam_x: one multiplier per element of x (length n_dec = N + 2*(N+1))
        Same idea but for the box constraints lbx <= x <= ubx.
        Every element of x has a box constraint: lbx[i] <= x[i] <= ubx[i].
        lam_x[i] tells you whether element i is stuck against one of its bounds at the optimal solution.
        - lam_x[i] = 0    → x[i] landed strictly between its bounds, the bounds aren't restricting it
        - lam_x[i] != 0   → x[i] is pinned to its lower or upper bound, the bound is actively forcing it there

        For controls (u_0..u_{N-1}, bounds [-50, 10]):
            IPOPT wants u_3 = -60 (brake harder) but can't — it gets clamped to -50.
            lam_x[3] != 0 signals the physical limit is active at that step.

        For slacks (slack_lo, slack_hi, bounds [0, inf]):
            If v_k >= 0 (no violation): optimal slack is 0, pinned to its lower bound → lam_x[...] != 0
            If v_k < 0  (violation):    slack lifts off zero to absorb it             → lam_x[...] = 0
            So for slacks it's the opposite of what you might expect — non-zero lam_x means the
            constraint was NOT violated (slack at zero), zero lam_x means it WAS violated (slack lifted).

    sensitivity.py uses both to compute du*/dw — the gradient of the optimal control with
    respect to the MLP weights — via KKT implicit differentiation:
        ∂u*/∂w = -(∂²L/∂x²)⁻¹ · (∂²L/∂x∂w)
    The Lagrangian L = J + lam_g·g + lam_x·x requires these multipliers to be evaluated.
    Without them you cannot differentiate through the solver.
    """
    kkt_data = {
        "lam_x": np.array(sol["lam_x"]).flatten(),  # type: ignore[index]
        "lam_g": np.array(sol["lam_g"]).flatten(),  # type: ignore[index]
        "x_opt": x_opt,
        "solver": solver,
        "weights": weights,
        "state": state,
        "context": context,
    }
    return u_star, kkt_data