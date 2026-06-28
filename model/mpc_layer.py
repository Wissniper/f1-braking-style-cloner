import os
import numpy as np
import torch
from mpc.ocp import build_ocp, solve as ocp_solve
from mpc.sensitivity import compute_jacobian, assert_jacobian_valid

DEBUG_GRAD = os.environ.get("DEBUG_GRAD", "0") == "1"


class MPCSolve(torch.autograd.Function):
    """
    Bridge between PyTorch MLP output and CasADi MPC solver.

    Usage:
        u_star = MPCSolve.apply(weights, solver, state, context, soft)

    weights: torch.Tensor shape (3,) — [w_jerk, w_speed, w_aggression]
    solver:  casadi.Function from mpc.ocp.build_ocp()
    state:   dict with keys s0, v0
    context: dict with keys v_setpoint, v_corner
    soft:    bool — True during training (soft constraints), False during eval
    Returns: scalar torch.Tensor — u_star (m/s²)
    """

    @staticmethod
    def forward(ctx, weights, solver, state, context, soft):

        """
        .detach() — cuts the tensor out of the computation graph. 
            The OCP solver doesn't understand autograd, so you don't want 
            PyTorch trying to track operations inside it.
        .double() — CasADi expects 64-bit floats; PyTorch defaults to 32-bit (float32).
        .numpy() — converts to a NumPy array, which CasADi can actually consume.
        """
        w = weights.detach().double().numpy()
        u_star, kkt_data = ocp_solve(solver, w, state, context, soft=soft)

        if DEBUG_GRAD:
            assert_jacobian_valid(kkt_data)

        J = compute_jacobian(kkt_data) # shape (1, 3)
        ctx.J = J
        ctx.weights_dtype = weights.dtype
        return torch.tensor(u_star, dtype=weights.dtype)
    
    @staticmethod
    def backward(ctx, grad_u):  # type: ignore[override]
        # grad_u: scalar tensor ∂loss/∂u*
        # ctx.J:  (1, 3) numpy array ∂u*/∂weights

        g = grad_u.item() # Tensor -> Numpy

        """
         ∂loss/∂weights = ∂loss/∂u*  ·  ∂u*/∂weights
                 =     g       ·     ctx.J
                 =  (scalar)   ·    (1, 3)
                 =             (1, 3)
        """
        grad_w_np = g * ctx.J

        #  [[0.4, -0.1, 0.7]]   # shape (1, 3) — can't be returned as grad for a (3,) tensor
        #  [0.4, -0.1, 0.7]     # shape (3,)   — matches weights, PyTorch accepts this
        #  => flatten
        grad_w = torch.tensor(grad_w_np.flatten(), dtype=ctx.weights_dtype)

        return grad_w, None, None, None, None # None for solver, state, context, soft
    
def verify_bridge(solver,
                    state: dict | None = None,
                    context: dict | None = None) -> None:
    
    """
    Run torch.autograd.gradcheck on MPCSolve.
    Raises if gradients are wrong
    """

    if state is None:
        state = {"s0": 0.0, "v0": 80.0}
    if context is None:
        context = {"v_setpoint": 80.0, "v_corner": 55.0}

    weights = torch.tensor(
        [1.0, 1.0, 1.0],
        dtype = torch.float64,
        requires_grad=True
    )

    def fn(w):
        """
        A wrapper to isolate the differentiable input.

        gradcheck can only probe inputs that are tensors with requires_grad=True.
        But MPCSolve.apply takes 5 arguments — only w (the weights) is differentiable.
        Wrapping it in fn hides the non-differentiable arguments (solver, state, context, True)
        so gradcheck only sees w.

        .unsqueeze(0) promotes the scalar output from shape () to (1,).
        gradcheck requires at least 1D output.
        """
        return MPCSolve.apply(w, solver, state, context, True).unsqueeze(0)

    """
    gradcheck validates your hand-written backward by comparing it against numerical finite differences:

        ∂u*/∂w_i  ≈  (fn(w + ε·eᵢ) - fn(w - ε·eᵢ)) / 2ε

    It does this for each weight w_i in turn, then checks the result matches what backward returns.

    ┌────────────┬────────────────────┬───────────────────────────────────────────────────────────────────────┐
    │  Argument  │       Value        │                                Meaning                                │
    ├────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────┤
    │ fn         │ the wrapper        │ function to test                                                      │
    ├────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────┤
    │ (weights,) │ tuple              │ inputs to probe — must match fn's arguments                           │
    ├────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────┤
    │ eps=1e-4   │ perturbation size  │ how much to nudge each weight for finite diff                         │
    ├────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────┤
    │ atol=1e-2  │ absolute tolerance │ max allowed absolute difference between analytical and numerical grad │
    ├────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────┤
    │ rtol=1e-2  │ relative tolerance │ max allowed relative difference                                       │
    └────────────┴────────────────────┴───────────────────────────────────────────────────────────────────────┘

    The tolerances are looser than gradcheck's defaults (1e-5) because the OCP solver
    introduces small numerical noise — tight tolerances would give false failures.

    Your backward is hand-written (it uses the KKT Jacobian, not PyTorch's autograd).
    gradcheck is the standard way to verify it's correct before training.
    If it passes, you can trust the gradients flowing back into the MLP are accurate.
    """
    passed = torch.autograd.gradcheck(fn, (weights,), eps=1e-4, atol=1e-2, rtol=1e-2)
    assert passed, "gradcheck failed, bridge gradients are incorrect"
    print("Bridge gradcheck PASSED.")

if __name__ == "__main__":
    solver = build_ocp()
    verify_bridge(solver)


