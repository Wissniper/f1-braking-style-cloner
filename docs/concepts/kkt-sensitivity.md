# KKT Conditions and Implicit Differentiation

## The problem we're solving

We want to train a neural network by backpropagating through an optimization solver. The solver takes cost weights as input and returns an optimal control `u*`. We need: how does `u*` change if we slightly change the weights?

That quantity is written `∂u*/∂weights` and means: **"if I nudge weight W by a tiny amount, how much does the braking profile move?"** It's a matrix — one row per decision variable, one column per weight. Each entry says "nudge this weight → this much change in that control."

That's exactly what's needed to train the neural network. The NN outputs weights → IPOPT solves → you compare the braking profile to the expert's → `∂u*/∂weights` tells you which direction to adjust the weights.

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

## What F(x\*, p) = 0 actually means

The stationarity condition says the gradient of the cost is zero at the optimum. Consider a simplified 1D version of your problem:

```
f(u) = w_jerk · u²
```

To find the minimum, take the derivative and set it to zero:

```
df/du = 2 · w_jerk · u = 0
```

That equation `df/du = 0` is `F(x*, p) = 0`. That's all it is:

- `x*` = `u` (the optimal braking force IPOPT solved for)
- `p` = `w_jerk` (the cost weight, the parameter)
- `F` = `2 · w_jerk · u` (the gradient of the cost)

At the optimum, this is always zero by definition — if the gradient weren't zero, you could still improve, so the optimizer would have kept going.

In your actual OCP, `F` is a big vector (one equation per decision variable: all `u_k`, all slacks). CasADi builds it symbolically.

## Implicit differentiation — the key insight

`F(x*, p) = 0` stays zero for **any** value of `p` — the optimizer always finds the minimum. So when you change `p`, `x*` adjusts automatically to keep F at zero.

That constraint relationship lets you differentiate through the solver without touching it.

Differentiate `F(x*, p) = 0` with respect to `p`:

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

### Concrete 1D example

With `F = 2 · w_jerk · u*`, differentiate with respect to `w_jerk`:

```
2 · u*  +  2 · w_jerk · (du*/dw_jerk)  =  0
```

Solve:

```
du*/dw_jerk  =  -u* / w_jerk
```

You get the gradient using numbers already computed by the solve — no re-running IPOPT.

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

You pass in the solved quantities from IPOPT (`x_opt`, `lam_x`, `lam_g`), and CasADi applies the implicit differentiation formula internally.

## Verifying correctness

The KKT sensitivity is validated against finite differences before training. See [finite-difference-check.md](finite-difference-check.md) for how that works and why it's needed.
