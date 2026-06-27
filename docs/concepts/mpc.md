# Model Predictive Control (MPC)

## What it is

MPC is a control strategy where, at every timestep, you solve a small optimization problem to decide what action to take. You look ahead a fixed number of steps, find the best sequence of actions, apply only the first one, then repeat at the next timestep.

Think of it like planning your route at every intersection rather than following a fixed map.

## How it works

1. You know your current state (e.g. speed, position)
2. Solve: "what sequence of actions over the next N steps minimizes my cost?"
3. Apply only the first action
4. Move to the next timestep and repeat

This "receding horizon" trick means you always re-plan with fresh information, so even if your plan was slightly wrong, you correct it constantly.

## The optimization problem

```
minimize   J = Σ stage_costs + terminal_cost
subject to  x_{k+1} = f(x_k, u_k)    ← dynamics (physics)
            x_k ∈ X                   ← state constraints (e.g. v ≥ 0)
            u_k ∈ U                   ← input constraints (e.g. |brake| ≤ limit)
```

- **J** — a number measuring how bad the trajectory is. You design it to encode what you want.
- **Σ stage_costs** — cost accumulated at each step k of the horizon.
- **terminal_cost** — cost at the very last step N. Often encodes a goal you must reach.
- **x_{k+1} = f(x_k, u_k)** — the physics. Forces the optimizer to only consider real trajectories.
- **X, U** — hard limits the optimizer can never violate.

## In this project

```
J = w_jerk        * Σ u_k²                  ← penalise harsh inputs
  + w_speed       * Σ (v_k - v_setpoint)²   ← stay near target speed
  + w_aggression  * (v_N - v_corner)²        ← nail the corner entry speed
```

- **w_jerk**: high → smooth braking. Low → controller doesn't care about smoothness.
- **w_speed**: high → controller tries to maintain a reference speed on the straight.
- **w_aggression**: high → controller really wants to hit `v_corner` at step N, so it brakes harder and later (more aggressive). Low → it arrives at the corner with margin to spare.

The MLP outputs these three weights based on context. Different weights = different driving style. That's the whole point.

## Why people use MPC

**Advantages**
- Handles constraints natively — you can't command something physically impossible
- Looks ahead — avoids situations that seem fine now but are bad in 5 steps
- Interpretable — you can read the cost function and understand why the controller acts the way it does

**Disadvantages**
- Computationally expensive — solving an optimization at every timestep adds up
- Needs a model — if the physics model is wrong, the controller is wrong
- Cost function tuning is non-trivial — which is exactly what this project automates with the MLP
