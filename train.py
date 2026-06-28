import os
import numpy as np
import torch
import torch.nn as nn

from data.fetch_telemetry import load_straight_data
from model.mlp import CostWeightMLP
from model.mpc_layer import MPCSolve, verify_bridge
from mpc.ocp import build_ocp

# --- Hyperparameters ---
LR = 1e-3           # learning rate: how big a step Adam takes each update
EPOCHS = 100        # number of full passes over all training laps
SMOOTH_LAMBDA = 0.01  # weight of the smoothing penalty (keeps cost weights from jumping between segments)
TRAIN_FRAC = 0.8    # fraction of laps used for training; remainder held out for validation
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")  # absolute path to data/ folder


def compute_context(seg):
    """
    Convert a telemetry segment into the MLP input tensor.

    A segment is one snapshot of the car's state. The three values are
    already normalised by load_straight_data() so they are in [0, 1]:
        v_norm   — current speed, scaled 0-1
        d_norm   — distance to corner, scaled 0-1
        a_recent — recent acceleration (negative = braking)
    """
    return torch.tensor(
        [seg.v_norm, seg.d_norm, seg.a_recent],
        dtype=torch.float32 
    )


def val_rmse(mlp, solver, val_laps, v_setpoint, v_corner):
    """
    Compute Root Mean Squared Error on the held-out validation laps.

    RMSE tells you the average prediction error in m/s² — smaller is better.
    Validation laps were never seen during training, so this measures how well
    the model generalises to new data rather than just memorising training laps.
    """
    mlp.eval()  # switch to eval mode: disables training-only behaviour (e.g. dropout)
    sq_errors = []
    context_dict = {"v_setpoint": v_setpoint, "v_corner": v_corner}

    with torch.no_grad():
        # torch.no_grad() tells PyTorch not to track operations for gradients.
        # During validation we only measure performance, never update weights,
        # so gradient tracking would waste memory and time.
        for lap in val_laps:
            for seg in lap:
                ctx = compute_context(seg)
                weights = mlp(ctx)                  # MLP predicts cost weights for this context
                state = {"s0": seg.s, "v0": seg.v}
                u_star = MPCSolve.apply(            # OCP solver predicts the braking action
                    weights.float(), solver, state, context_dict,
                    False                           # soft=False: hard constraints during eval (no slack)
                )
                sq_errors.append((u_star.item() - seg.u_expert) ** 2)
                # .item() converts a scalar tensor to a plain Python float

    mlp.train()  # switch back to training mode before returning
    return float(np.sqrt(np.mean(sq_errors)))  # RMSE: sqrt of average squared error


def main():
    # --- 1. Load telemetry data ---
    print("Loading telemetry...")
    all_laps, v_setpoint, v_corner = load_straight_data()
    # all_laps: list of laps, each lap a list of segments
    # v_setpoint: target speed on the straight
    # v_corner:   speed the car must reach at corner entry
    print(f"Loaded {len(all_laps)} laps. v_setpoint={v_setpoint:.1f}, v_corner={v_corner:.1f}")

    # --- 2. Train / validation split ---
    n_train = max(1, int(len(all_laps) * TRAIN_FRAC))  # at least 1 training lap
    train_laps = all_laps[:n_train]
    val_laps = all_laps[n_train:] or all_laps[-1:]     # fallback: reuse last lap if only 1 lap total

    # --- 3. Build OCP solver (expensive to construct, cheap to call) ---
    solver = build_ocp()

    # --- 4. Build MLP and optimizer ---
    mlp = CostWeightMLP()  # randomly initialised weights
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    # Adam: adaptive gradient descent. Better than plain SGD because it adjusts
    # the step size per parameter and uses momentum. mlp.parameters() returns
    # every weight and bias in the network so Adam knows what to update.
    context_dict = {"v_setpoint": v_setpoint, "v_corner": v_corner}

    # --- 5. Verify bridge gradients before training ---
    print("Verifying bridge gradients...")
    verify_bridge(solver)
    # Runs gradcheck once. If the KKT Jacobian is wrong, training would
    # silently produce bad gradients for all 100 epochs — catch it early.

    train_losses, val_rmses = [], []  # history for plotting after training

    # --- 6. Training loop ---
    for epoch in range(EPOCHS):
        mlp.train()         # ensure training mode at the start of each epoch
        optimizer.zero_grad()   # clear gradients from the previous epoch;
                                # PyTorch accumulates gradients by default so you must reset manually
        total_loss = torch.tensor(0.0)  # accumulate all segment losses before backprop
        w_prev = None                   # previous segment's weight vector for smoothing penalty

        for lap in train_laps:
            w_prev = None   # reset smoothing at each lap boundary (different laps aren't continuous)
            for seg in lap:

                # --- Forward pass ---
                ctx = compute_context(seg)
                weights = mlp(ctx)                      # MLP: context → [w_jerk, w_speed, w_aggression]
                state = {"s0": seg.s, "v0": seg.v}
                u_star = MPCSolve.apply(                # OCP solver: weights → u* (predicted braking)
                    weights, solver, state, context_dict,
                    True    # soft=True during training: slack constraints prevent infeasibility
                )

                # --- Loss function ---
                # Behavioural cloning loss: how far is the prediction from what the expert driver did?
                bc_loss = (u_star - seg.u_expert) ** 2

                # Smoothing penalty: penalise cost weights changing too rapidly between segments.
                # Without this the MLP could output very different weights for similar situations,
                # producing jerky behaviour. SMOOTH_LAMBDA = 0.01 keeps this term small.
                smooth = (SMOOTH_LAMBDA * ((weights - w_prev) ** 2).sum()
                          if w_prev is not None else torch.tensor(0.0))

                total_loss = total_loss + bc_loss + smooth

                # Save weights for next segment's smoothing penalty.
                # .detach() removes gradient tracking — we don't want gradients
                # flowing backwards through the previous step's weights.
                w_prev = weights.detach()

        # --- Backward pass ---
        total_loss.backward()
        # Traverses the computation graph backwards from total_loss through:
        #   bc_loss → u_star → MPCSolve.backward() (KKT Jacobian) → weights → MLP layers
        # Every MLP parameter gets a .grad value: "nudge me this much to reduce the loss".

        optimizer.step()
        # Adam reads all .grad values and updates every MLP parameter.
        # The network shifts slightly to make future predictions closer to u_expert.

        # --- Logging ---
        rmse = val_rmse(mlp, solver, val_laps, v_setpoint, v_corner)
        train_losses.append(total_loss.item())
        val_rmses.append(rmse)
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_loss={total_loss.item():.4f}  val_rmse={rmse:.4f}")
        # Watch for: both numbers decreasing = good.
        # If train_loss falls but val_rmse rises = overfitting (memorising, not generalising).

    # --- 7. Save outputs ---
    torch.save(mlp.state_dict(), os.path.join(DATA_DIR, "mlp_weights.pt"))
    # state_dict(): dictionary of every weight and bias tensor — all you need to restore the model.

    np.savez(os.path.join(DATA_DIR, "train_history.npz"),
             train_loss=np.array(train_losses),
             val_rmse=np.array(val_rmses))
    # Saves the loss curves so you can plot training progress afterwards.

    print(f"\nFinal val RMSE: {val_rmses[-1]:.4f} m/s")


if __name__ == "__main__":
    main()
