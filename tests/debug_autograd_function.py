"""Test if custom autograd.Function backward works in this PyTorch version."""

import torch
import sys


class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print(f"  MyFunction.forward called, x.shape={x.shape}", file=sys.stderr, flush=True)
        ctx.save_for_backward(x)
        # Simple operation: pad with zeros
        zeros = torch.zeros(1, 1, 1, x.shape[-1], device=x.device)
        result = torch.cat([zeros, x, zeros], dim=2)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f"  MyFunction.backward called, grad_output.shape={grad_output.shape}", file=sys.stderr, flush=True)
        x, = ctx.saved_tensors
        # Return gradient for the middle part only
        grad_x = grad_output[:, :, 1:-1, :]
        return grad_x


# Test
x = torch.randn(1, 1, 4, 4, requires_grad=True, device='cuda:0')
print(f"x.requires_grad = {x.requires_grad}")

y = MyFunction.apply(x)
print(f"y.requires_grad = {y.requires_grad}")
print(f"y.grad_fn = {y.grad_fn}")

loss = y.sum()
loss.backward()
print(f"x.grad shape = {x.grad.shape}")
print(f"x.grad[0,0,:,0] = {x.grad[0,0,:,0].tolist()}")
print(f"Expected: all 1s (since each x element appears once in the output)")
