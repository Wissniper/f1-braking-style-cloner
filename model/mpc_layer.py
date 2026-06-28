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