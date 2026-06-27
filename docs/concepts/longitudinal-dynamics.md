# Longitudinal Vehicle Dynamics

## What it is

Longitudinal dynamics models how a vehicle moves along a straight line — speeding up and slowing down. "Longitudinal" means along the direction of travel (as opposed to lateral = sideways).

It ignores everything sideways: no steering, no cornering forces, no yaw. Just position and speed.

## The state

The car is described by two numbers at each timestep:

```
x = [s, v]
```

- **s** — position along the straight (meters). Where is the car?
- **v** — speed (m/s). How fast is it going?

Together these fully describe the car's situation for this problem. You don't need mass, engine torque, or tyre models — we work with acceleration directly as the control input.

## The control input

```
u = longitudinal acceleration (m/s²)
```

Positive = accelerating forward. Negative = braking (decelerating).

## The dynamics equations

```
s_{k+1} = s_k + v_k · dt
v_{k+1} = v_k + u_k · dt
```

**Breaking it down:**

`s_{k+1} = s_k + v_k · dt`
New position = old position + (speed × time elapsed).
If you're going 80 m/s and 0.02 seconds pass, you move 1.6 meters forward.

`v_{k+1} = v_k + u_k · dt`
New speed = old speed + (acceleration × time).
If you brake at -50 m/s² for 0.02 seconds, you lose 1 m/s.

This is **Euler integration** — the simplest way to step forward in time. It assumes acceleration is constant within each timestep, which is accurate enough when dt is small.

## Why Euler and not something more accurate

More accurate methods (Runge-Kutta 4th order) exist but add complexity without meaningful benefit here. At dt = 0.02 s, Euler error over the 10-step horizon (0.2 s) is negligible. We also re-solve the MPC at every timestep, so any small model error is corrected immediately.

## Why this simplified model is enough

A real F1 car involves aerodynamic drag, downforce, tyre slip, engine torque curves, and more. But we're not simulating a car from scratch — we're **matching a real driver's recorded acceleration commands**. The physics model only needs to be accurate enough that the MPC produces realistic speed profiles over a short horizon. The behavioral cloning loss corrects for model mismatch implicitly by training on real data.

## Constraints in this project

```
0 ≤ v ≤ 92 m/s     ← can't go backwards; can't exceed Monza top speed
-50 ≤ u ≤ 10 m/s²  ← approximate F1 braking and acceleration limits
```

- **-50 m/s² ≈ 5g braking** — F1 cars can peak at ~6g but briefly; -50 is a safe sustained limit.
- **+10 m/s² ≈ 1g acceleration** — conservative for an F1 car on a straight, but appropriate for the coasting/braking zone we're modelling (the car isn't accelerating hard here).
