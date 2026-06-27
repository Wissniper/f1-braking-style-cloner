# IPOPT

## What it is

IPOPT (Interior Point OPTimizer) is an open-source solver for nonlinear optimization problems. When CasADi formulates your optimization problem, IPOPT is the engine that actually finds the solution.

You almost never interact with IPOPT directly — CasADi calls it for you.

## What problem it solves

```
minimize   f(x)
subject to g_lb ≤ g(x) ≤ g_ub
           x_lb ≤ x ≤ x_ub
```

- **f(x)** — a smooth objective function to minimize
- **g(x)** — constraint functions (equalities or inequalities)
- **x** — the decision variables (what you're optimizing over)

In this project: x = [u_0, ..., u_9, slack variables], f = the MPC cost J, g = dynamics and constraint expressions.

## How it works (the idea)

IPOPT uses the **interior point method**. Instead of enforcing `x ≥ 0` as a hard wall, it adds a penalty to the objective that blows up as `x → 0`. This turns a constrained problem into an unconstrained one that's easier to handle.

It then iteratively improves the solution using Newton steps — computing the gradient and curvature of the objective, stepping toward the optimum, repeating until convergence.

Convergence = the KKT conditions are satisfied within a tolerance (1e-8 in this project).

## What you get back

After a solve:
- `x_opt` — the optimal decision variables (the controls u_0..u_9 and slacks)
- `lam_x` — dual variables for box constraints
- `lam_g` — dual variables for constraint functions
- A status (0 = converged)

`lam_x` and `lam_g` are the KKT multipliers — needed to compute the sensitivity Jacobian in `mpc/sensitivity.py`.

## Key settings in this project

```python
opts = {
    "ipopt.print_level": 0,  # suppress all output
    "print_time": 0,
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-8,       # tight tolerance for accurate KKT sensitivity
}
```

Tight tolerance matters here: the KKT sensitivity formula assumes you're at a true optimum. A loose tolerance means the sensitivity Jacobian is inaccurate, giving wrong gradients to the MLP.

## Why people use IPOPT

**Advantages**
- Free and open-source
- Handles general nonlinear, non-convex problems
- Robust — works on problems where simpler solvers fail
- Returns KKT multipliers needed for sensitivity analysis

**Disadvantages**
- Slower than solvers specialized for convex problems (OSQP, qpOASES)
- Can fail to converge on badly conditioned problems
- Error messages are cryptic ("Restoration phase failed" tells you something is wrong, not what)

## In this project

The OCP has ~30 decision variables (10 controls + slack variables). IPOPT solves it in under 1ms. We call it once per training timestep with no warm-starting between calls, to keep the KKT point consistent and the sensitivity Jacobian accurate.
