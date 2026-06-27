# KKT Conditions and Implicit Differentiation

## The problem we're solving

We want to train a neural network by backpropagating through an optimization solver. The solver takes cost weights as input and returns an optimal control `u*`. We need: how does `u*` change if we slightly change the weights? — i.e. `∂u*/∂weights`.

You can't just backprop through IPOPT — it's an iterative black box, not a differentiable computation graph. The solution: **implicit differentiation via KKT conditions**.

## KKT conditions — what they are

At the optimal solution of a constrained optimization, these four conditions hold:

**1. Stationarity** — the gradient of the Lagrangian is zero:
```
∇_x f(x) + λ · ∇_x g(x) = 0
```
In plain English: at the optimum, the objective's pull and the constraints' pull perfectly cancel. If you could still improve f without violating constraints, you'd have moved already.

**2. Primal feasibility** — constraints are satisfied:
```
g(x) ≤ 0
```

**3. Dual feasibility** — multipliers are non-negative:
```
λ ≥ 0
```

**4. Complementary slackness** — if a constraint isn't active, its multiplier is zero:
```
λ_i · g_i(x) = 0
```
In plain English: a loose constraint has no effect on the optimum, so its "shadow price" is zero.

`λ` (lambda) = the **KKT multipliers**. In CasADi/IPOPT these are `lam_g` and `lam_x`. They come out of every IPOPT solve for free.

## Implicit differentiation — the key insight

The stationarity condition is a system of equations that `x*` satisfies:

```
F(x*, p) = 0          where p = [w_jerk, w_speed, w_aggression]
```

If we change `p` slightly, how does `x*` change? Differentiate both sides with respect to p:

```
∂F/∂x · ∂x*/∂p  +  ∂F/∂p  =  0
```

Rearrange:

```
∂x*/∂p  =  - (∂F/∂x)⁻¹ · ∂F/∂p
```

- `∂F/∂x` — how sensitive the stationarity condition is to the solution. CasADi computes this from the problem structure.
- `∂F/∂p` — how sensitive the stationarity condition is to the parameters. Also symbolic, computed by CasADi.
- The result gives us `∂u*/∂weights` — **without running the solver again**.

## Why soft constraints?

The implicit differentiation formula requires LICQ (Linear Independence Constraint Qualification) — roughly: no two active constraints point in the same direction.

Hard inequality constraints that are exactly active can violate LICQ subtly. Soft constraints (slack variables + penalty) remove this risk: the slacks are always strictly positive interior to their bounds, so LICQ holds by construction.

## How CasADi does this for you

```python
sens = solver.factory('sens', ['x0', 'p', 'lam_x0', 'lam_g0'], ['jac:x:p'])
result = sens(x0=x_opt, p=p_val, lam_x0=lam_x, lam_g0=lam_g)
J = result['jac_x_p']   # shape (n_dec, n_params)
# Extract row 0 (u_0), columns 0:3 (the three cost weights)
du_dw = J[0, 0:3]       # shape (1, 3)
```

You pass in the solved quantities from IPOPT, and CasADi applies the formula above internally.

## The finite-difference check

```
J_fd[0, i] = ( u*(w + ε·eᵢ)  -  u*(w - ε·eᵢ) ) / (2ε)
```

- Perturb weight i by +ε and -ε
- Re-solve the OCP both times
- Divide the difference in u* by 2ε

Slow (3 extra solves per check) but guaranteed correct. If `J_kkt ≈ J_fd`, the sensitivity is right. If not, something is broken in the KKT path — fix it before training.
