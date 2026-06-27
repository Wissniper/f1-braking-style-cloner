# PyTorch Autograd and Custom Functions

## What autograd is

Autograd is PyTorch's automatic differentiation engine. Every time you do math on a tensor with `requires_grad=True`, PyTorch records the operation in a **computation graph**. When you call `.backward()`, it walks the graph backwards and computes gradients using the chain rule.

This is how neural networks are trained: define a loss, call `loss.backward()`, and every parameter's `.grad` is populated automatically.

## The chain rule in one line

If `loss = f(g(x))`, then:
```
∂loss/∂x = (∂f/∂g) · (∂g/∂x)
```
Autograd applies this repeatedly through every operation in the graph — no matter how many layers deep.

## The problem: non-PyTorch operations

When your computation includes something outside PyTorch — like an IPOPT solve — autograd has no idea how to differentiate through it. The graph stops there.

Solution: **`torch.autograd.Function`** — a way to teach PyTorch how to differentiate through custom operations by manually defining the `forward` and `backward` passes.

## Custom Function structure

```python
class MyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = some_external_call(input.numpy())
        ctx.J = compute_jacobian(...)  # save what backward will need
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = ∂loss/∂output  (comes from PyTorch)
        # return      = ∂loss/∂input   (chain rule: grad_output * ∂output/∂input)
        return torch.tensor(grad_output.item() * ctx.J)
```

`ctx` is a context object — use it to pass data from `forward` to `backward`. Anything you need for the gradient (like the KKT Jacobian) goes here.

## In this project: MPCSolve

```
forward(ctx, weights, solver, state, context, soft):
    w = weights.numpy()
    u_star, kkt_data = ocp.solve(w, state, context)    ← IPOPT solve
    ctx.J = sensitivity.compute_jacobian(kkt_data)     ← ∂u*/∂weights, shape (1,3)
    return torch.tensor(u_star)

backward(ctx, grad_u):
    # grad_u = ∂loss/∂u*      scalar, comes from PyTorch
    # ctx.J  = ∂u*/∂weights   shape (1,3), saved in forward
    # chain rule: ∂loss/∂weights = ∂loss/∂u* · ∂u*/∂weights
    grad_w = grad_u.item() * ctx.J    # shape (1,3)
    return torch.tensor(grad_w), None, None, None, None
```

The `None` returns correspond to `solver`, `state`, `context`, `soft` — not tensors, no gradients needed.

## gradcheck — verifying your backward

`torch.autograd.gradcheck` compares your custom backward against a numerical finite-difference Jacobian. If they match within tolerance, your backward is correct.

```python
weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (weights,), eps=1e-4, atol=1e-2)
```

`float64` is required — the finite differences are too noisy at float32 precision. Run this before training. If it fails, the bridge is broken and training gradients will be wrong.

## Why custom autograd Functions

**Use when:** bridging PyTorch to external code (solvers, simulators, compiled libraries) that has known, computable gradients but isn't written in PyTorch.

**Don't use when:** everything can be expressed in native PyTorch ops — just let autograd handle it natively.
