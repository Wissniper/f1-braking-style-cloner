# CasADi

## What it is

CasADi is a Python (and MATLAB) library for writing and solving optimization problems — specifically the kind that show up in optimal control. You describe your problem symbolically (like writing a math formula), and CasADi handles the differentiation and hands it off to a solver like IPOPT.

Think of it as the middleman between "here's my math" and "here's the solution."

## The key idea: symbolic computation

Normal Python computes values immediately: `x = 3 + 5` gives `8`.

CasADi computes symbolically: `x = ca.MX.sym('x')` creates a placeholder. `y = x**2 + 3*x` creates a symbolic expression. Nothing is evaluated yet — CasADi just builds a graph of operations.

This matters because CasADi can then:
- Automatically differentiate that expression (compute `dy/dx` exactly, no approximations)
- Pass the expression and its derivatives to a numerical solver

## How you use it in practice

```python
import casadi as ca

# 1. Declare symbolic variables
x = ca.MX.sym('x', 2)   # [position, speed]
u = ca.MX.sym('u')       # acceleration

# 2. Write the dynamics symbolically
dt = 0.02
x_next = ca.vertcat(x[0] + x[1]*dt, x[1] + u*dt)

# 3. Define a cost
cost = u**2

# 4. Package as an NLP and hand to IPOPT
nlp = {'x': u, 'f': cost, 'g': x_next - x}
solver = ca.nlpsol('solver', 'ipopt', nlp)
sol = solver(x0=0.0, lbg=0, ubg=0)
```

## Parametric NLPs — why this matters for this project

CasADi supports **parameters** — values that are fixed during one solve but can change between solves. The cost weights are declared as parameters:

```python
p = ca.MX.sym('p', 3)   # [w_jerk, w_speed, w_aggression]
cost = p[0]*u**2 + p[1]*(v - v_ref)**2 + ...
nlp = {'x': decision_vars, 'p': p, 'f': cost, ...}
solver = ca.nlpsol('solver', 'ipopt', nlp)
```

You build the solver **once**. Then each training call just passes different `p` values:

```python
sol = solver(x0=x0, p=[1.0, 2.0, 0.5], ...)
```

This is much faster than rebuilding the solver from scratch each time — critical when you call it thousands of times during training.

## Why people use CasADi

**Advantages**
- Exact automatic differentiation — no numerical approximations
- Built-in connection to solvers (IPOPT, OSQP, etc.)
- Supports sensitivity analysis (∂solution/∂parameters) natively — essential for this project
- Efficient: the symbolic graph is compiled once, not re-interpreted on every call

**Disadvantages**
- Unfamiliar syntax — `ca.MX.sym`, `ca.vertcat`, `ca.nlpsol` take getting used to
- Debugging is harder — errors often surface as cryptic IPOPT failure codes, not Python tracebacks
- Only useful for structured optimization problems, not general Python code

## In this project

We build the longitudinal MPC once at startup with cost weights as parameters. During training we call `solver(p=weights, ...)` at every timestep. CasADi's `solver.factory(...)` then provides the sensitivity function that gives ∂u*/∂weights — the key gradient for backpropagating through the MPC into the MLP.
