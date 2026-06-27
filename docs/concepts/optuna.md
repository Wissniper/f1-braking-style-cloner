# Optuna (Bayesian Hyperparameter Optimization)

## What it is

Optuna is a Python library for automated hyperparameter search. You define a function that takes parameter values and returns a score (lower = better), and Optuna finds the parameter values that minimize that score — smarter and faster than trying combinations by hand.

In this project it finds the best fixed MPC cost weights `[w_jerk, w_speed, w_aggression]` to use as a baseline.

## The problem with grid search

If you want to try 10 values for each of 3 parameters, that's 10³ = 1000 combinations. Most of them will be terrible, and you're spending equal time on all of them.

**Random search** is already better — randomly sampling from the space tends to find good regions faster than a grid.

**Bayesian optimization** is smarter still: it learns from previous trials to guess where the good regions are.

## How Bayesian optimization works

After a few random trials, Optuna builds a model of "which parameter regions tend to give low scores." It then suggests new combinations that are likely to be good — balancing exploitation (go deeper where it's already good) and exploration (try unknown regions).

Optuna's default sampler is **TPE (Tree-structured Parzen Estimator)**:
- Model `P(params | score is good)` and `P(params | score is bad)` separately
- Suggest parameters where the first is high relative to the second

In practice: significantly more efficient than grid or random search. 200 trials with TPE typically finds a solution competitive with thousands of random trials.

## How you use it

```python
import optuna

def objective(trial):
    w_jerk        = trial.suggest_float("w_jerk", 0.01, 100.0, log=True)
    w_speed       = trial.suggest_float("w_speed", 0.01, 100.0, log=True)
    w_aggression  = trial.suggest_float("w_aggression", 0.01, 100.0, log=True)
    return compute_loss([w_jerk, w_speed, w_aggression])

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

print(study.best_params)
```

`log=True` samples on a logarithmic scale — appropriate here because the weights span several orders of magnitude (0.01 to 100) and relative differences matter more than absolute ones.

## In this project

The baseline answers: "what's the best a static, context-free controller can possibly do?" Same MPC, same loss, but weights are fixed numbers found by Optuna rather than context-dependent outputs from an MLP.

If the learned MPC doesn't beat this baseline, the MLP isn't learning anything useful — either the context features don't help, or the training loop is broken.

## Why people use Optuna

**Advantages**
- Simple API — minimal boilerplate
- Much smarter than random search with almost no extra code
- Handles float, integer, categorical, and log-scale parameters
- Parallel trials out of the box

**Disadvantages**
- Still expensive if each trial is slow (200 trials × ~1500 MPC solves each in this project)
- Not suitable when gradients are available — gradient-based methods will always be faster
- Struggles in very high-dimensional spaces (>20 parameters)
