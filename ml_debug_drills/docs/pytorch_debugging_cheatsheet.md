PyTorch Debugging Cheat Sheet
=============================

1. Core Invariants
------------------

Shapes:
- X: (batch, features)
- y: (batch,)
- logits: (batch, num_classes)

Dtypes:
- X: float32
- y: int64 (torch.long)

Devices:
- Model, inputs, and labels must be on the same device

Training Order:
- zero_grad()
- forward
- loss
- backward()
- step()

Gradient Sanity:
- Check isnan() or isinf() on logits and on param.grad


2. Debugging Pattern (Step-by-Step)
------------------------------------

1. Inspect data (shape, dtype, device)
2. Inspect model and confirm final layer matches num_classes
3. Run a single batch manually
4. Verify training loop order
5. Validate accuracy calculation using argmax(dim=1)
6. Check for NaNs or Infs in logits and gradients
7. Ensure shuffle=True for training
8. Fix one bug at a time and re-run


3. Verbal Flow (for interviews)
-------------------------------

- "First I'll validate the core invariants: shape, dtype, and device."
- "I'll examine one batch to confirm X is float32 and y is long."
- "I'll verify that the model’s final layer outputs num_classes logits."
- "I'll check training order: zero_grad -> forward -> loss -> backward -> step."
- "I'll ensure xb, yb, and the model are all on the same device."
- "For accuracy, I'll use logits.argmax(dim=1)."
- "If loss becomes NaN, I check learning rate, logits, and gradients."


4. Quick Checklist
------------------

[ ] X: correct shape, dtype, device  
[ ] y: correct shape, dtype, device  
[ ] Model final layer outputs num_classes  
[ ] One batch sanity forward pass  
[ ] zero_grad before backward  
[ ] xb and yb moved to device  
[ ] logits shape: (N, C)  
[ ] y shape: (N,)  
[ ] y dtype: long  
[ ] Accuracy uses argmax  
[ ] Check logits isnan/isinf  
[ ] Check param.grad isnan/isinf  


5. Useful Code Snippets
-----------------------

Checking for NaN or Inf:
    logits.isnan().any()
    logits.isinf().any()

    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.grad.isnan().any(), p.grad.isinf().any())

One-batch sanity test:
    xb, yb = next(iter(dataloader))
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
    loss = criterion(logits, yb)

    print("xb:", xb.shape, xb.dtype, xb.device)
    print("yb:", yb.shape, yb.dtype, yb.device)
    print("logits:", logits.shape)
    print("loss:", loss.item())

Correct training loop:
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()


Notes:
------
Add any of your own debugging reminders here.