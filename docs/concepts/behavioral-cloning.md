# Behavioral Cloning (Imitation Learning)

## What it is

Behavioral cloning (BC) is the simplest form of imitation learning: you watch an expert do something, record their (state → action) pairs, then train a model to reproduce that mapping using supervised learning.

No reward function, no simulation, no trial-and-error. Just: "given the same situation the expert was in, do what they did."

## How it works

1. Collect expert demonstrations: a dataset of `(state, action)` pairs
2. Train a model to predict the expert's action given a state
3. Loss = MSE(predicted_action, expert_action)

That's it. It's basically image classification but for actions.

## Why it's useful

- Simple to implement and understand
- No reward engineering — the expert defines what "good" looks like implicitly
- Works well when you have enough clean demonstrations

## The big limitation: distribution shift

The expert always acts correctly, so the training data never shows the kinds of states you get after making a mistake. At test time, if your model makes a small error, it enters a state the expert never visited — and it has no idea what to do, making things worse.

This is less of a problem here because we're in a low-dimensional, well-constrained setting (a straight with known physics).

## In this project

The expert is Hamilton's telemetry. The "action" is the longitudinal acceleration command `u_expert` at each timestep.

But we don't clone the action directly — we clone it **through** the MPC. The model outputs MPC cost weights, the MPC solves for `u*`, and the loss penalises the gap between `u*` and `u_expert`:

```
loss = MSE(u*, u_expert) + λ * |w_t - w_{t-1}|²
```

- **MSE(u*, u_expert)** — behavioral cloning loss. Forces `u*` to match what Hamilton actually did.
- **λ * |w_t - w_{t-1}|²** — temporal smoothness. Stops the weights jumping wildly between timesteps. λ = 0.01 keeps it mild.

The advantage over cloning `u` directly: the MPC enforces hard physical constraints automatically — the model can't accidentally output a braking force that exceeds tyre limits.
