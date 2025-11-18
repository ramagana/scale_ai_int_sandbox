"""
⸻

1. Tensors: Creation, Dtypes, Shapes, Device Transfers

Concepts
	•	torch.tensor([...]) creates a tensor from data.
	•	torch.zeros, torch.ones, torch.randn create initialized tensors.
	•	Dtypes must match model expectations:
	•	torch.float32 → model inputs
	•	torch.long → classification labels (CrossEntropyLoss)
	•	Shapes:
	•	Use .shape to inspect.
	•	Use .unsqueeze and .reshape to fix shape mismatches.
	•	Device:
	•	CPU default.
	•	Move to GPU with .to("cuda") or .cuda().
"""
# Demonstration Code

import torch

# Creation
a = torch.tensor([1.0, 2.0, 3.0])          # float tensor
b = torch.tensor([1, 2, 3], dtype=torch.long)

# Random initialization
x = torch.randn(2, 3)

print("x:", x)
print("dtype:", x.dtype)
print("shape:", x.shape)

# Device example
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
print("device:", x.device)

# Reshaping
y = torch.randn(6)
y2 = y.reshape(2, 3)
print("reshaped:", y2.shape)

"""
What to do in debugger:
Inspect x.dtype, x.device, x.shape.
Hover over variables to confirm device movement.
⸻"""

"""
2. Autograd: requires_grad, backward(), gradient inspection, zero_grad()

Concepts
	•	Autograd tracks ops on tensors with requires_grad=True.
	•	After a forward pass:
	•	loss.backward() computes gradients.
	•	Gradients accumulate unless reset with optimizer.zero_grad().
	•	.grad attributes appear only after backward.

Demonstration Code"""

import torch
from torch import nn

# Simple linear model
w = torch.randn(3, 1, requires_grad=True)     # parameters
x = torch.randn(4, 3)                         # batch of 4 samples
y = torch.randn(4, 1)                         # targets

# Forward pass
pred = x @ w
loss = ((pred - y)**2).mean()

print("loss:", loss.item())

# Backward pass
loss.backward()

print("gradients for w:")
print(w.grad)          # inspect gradients

# Accumulation demo: DO NOT run backward twice without zeroing
w.grad.zero_()
print("grad after zeroing:", w.grad)
"""
What to examine:
	•	Before backward: w.grad is None.
	•	After backward: tensor of same shape as w.
	•	After zeroing: tensor of zeros.

⸻"""

"""3. No Grad Context: with torch.no_grad()

Concepts

Use no_grad when you:
	•	evaluate a model,
	•	generate predictions,
	•	want to avoid building computation graphs,
	•	want to avoid memory growth.

Inside this context:
	•	Gradients are not tracked.
	•	.requires_grad is ignored.

Demonstration Code"""

import torch
from torch import nn

model = nn.Linear(3, 1)
x = torch.randn(1, 3)

# With gradient tracking
pred1 = model(x)
print("requires_grad (pred1):", pred1.requires_grad)

# Disable autograd
with torch.no_grad():
    pred2 = model(x)
    print("requires_grad (pred2):", pred2.requires_grad)

"""Expected behavior:
	•	pred1.requires_grad=True
	•	pred2.requires_grad=False

⸻

Suggested Debugger Tasks

Task 1 — Set a breakpoint in the autograd demo

Inspect:
	•	w.grad before and after backward.
	•	loss graph in call stack.
	•	How the debugger shows grad_fn on tensors.

Task 2 — Force a shape error

Add this in the code:

bad = torch.randn(5, 5)
pred = x @ bad  # shape mismatch

Use the debugger to inspect the stack trace.

Task 3 — Device mismatch

Add:

x = x.to("cuda")
w = w.to("cpu")  # mismatch

pred = x @ w     # runtime error

Debug by stepping into the failing line.

⸻"""

