# Finite Difference Check

## What it is

A numerical method to approximate a derivative by running the solver twice with slightly different inputs and measuring how much the output changes.

For a single weight `w_i`:

```
∂u*/∂w_i  ≈  ( u*(w + ε·eᵢ)  -  u*(w - ε·eᵢ) ) / (2ε)
```

- `ε` = a small perturbation (e.g. 1e-4)
- `eᵢ` = a vector of zeros with a 1 at position i (nudges only weight i)
- `u*(w + ε·eᵢ)` = braking profile when weight i is slightly higher
- `u*(w - ε·eᵢ)` = braking profile when weight i is slightly lower

In plain English: nudge the weight up a tiny bit, nudge it down a tiny bit, see how much the output moved, divide by how far you nudged. That ratio approximates the true derivative.

## Why use both +ε and -ε (central differences)?

Using only `+ε` (forward difference) introduces first-order error. Using both `+ε` and `-ε` (central difference) cancels that error, giving a much more accurate approximation for the same step size.

## Why it's needed

The KKT sensitivity (from [kkt-sensitivity.md](kkt-sensitivity.md)) is computed analytically via CasADi — fast, but it can silently return wrong values if the problem is set up incorrectly (wrong constraint structure, LICQ violated, CasADi factory misconfigured).

Finite differences are slow but **guaranteed correct** — they're just re-running the solver. If the two methods agree, the KKT path is trustworthy. If they disagree, something is broken in the analytical path and it must be fixed before training.

## The check in code

```python
def finite_difference_check(solver, w, eps=1e-4):
    J_fd = np.zeros((n_dec, n_params))
    for i in range(n_params):
        w_plus  = w.copy(); w_plus[i]  += eps
        w_minus = w.copy(); w_minus[i] -= eps
        u_plus  = solve_ocp(solver, w_plus)['u']
        u_minus = solve_ocp(solver, w_minus)['u']
        J_fd[:, i] = (u_plus - u_minus) / (2 * eps)
    return J_fd
```

Then compare:

```python
J_kkt = compute_kkt_sensitivity(solver, w)
assert np.allclose(J_kkt, J_fd, atol=1e-4), "KKT sensitivity mismatch"
```

## Cost

For `n_params` weights, the check requires `2 · n_params` extra solver calls. With 3 weights that's 6 extra IPOPT solves — slow, but only run once as a diagnostic before training begins.

## When to run it

- After any change to the OCP formulation
- After changing the CasADi `factory` call
- If training loss behaves unexpectedly (gradients may be wrong)
- Never in the training loop — it's a one-off sanity check

## Choosing ε

Too large: the approximation is inaccurate (nonlinearity dominates).  
Too small: floating point cancellation errors dominate.  
`ε = 1e-4` to `1e-5` works for most OCP problems. If the check fails, try a different ε before assuming the KKT path is broken.
