"""
ANIL (Almost No Inner Loop) meta-learning for value prediction.

Only the value head is adapted in the inner loop. The trunk + bottleneck
are frozen during inner-loop adaptation and only updated by the outer
(meta) optimizer.

This is FOMAML with a restricted adaptation surface — first-order only,
no second derivatives through the inner loop.
"""

import torch
import torch.nn as nn
from torch.func import functional_call


def value_loss(v_pred, z):
    """MSE loss for value prediction."""
    return torch.nn.functional.mse_loss(v_pred, z)


def inner_adapt_anil(
    model,
    base_params,
    head_param_names,
    Xs,
    ys_val,
    inner_lr,
    inner_steps,
    verbose=False,
):
    """
    ANIL inner-loop: adapt only value head parameters on the support set.

    Parameters
    ----------
    model : nn.Module
        The full model (trunk + value head).
    base_params : dict
        Full named parameter dict for the model.
    head_param_names : set
        Names of parameters to adapt (value head only).
    Xs : Tensor
        Support set inputs [k_support, C, H, W].
    ys_val : Tensor
        Support set value targets [k_support].
    inner_lr : float
        Inner-loop learning rate.
    inner_steps : int
        Number of inner-loop gradient steps.
    verbose : bool
        Print per-step loss.

    Returns
    -------
    adapted_params : dict
        Full parameter dict with adapted value head params.
    """
    # Start from a copy of all params
    params = {k: v.clone() for k, v in base_params.items()}

    for step in range(1, inner_steps + 1):
        # Only require gradients on the head params
        params_for_grad = {}
        for k, v in params.items():
            if k in head_param_names:
                params_for_grad[k] = v.detach().clone().requires_grad_(True)
            else:
                params_for_grad[k] = v.detach()  # trunk frozen

        v_pred = functional_call(model, params_for_grad, (Xs,))
        loss = value_loss(v_pred, ys_val)

        # Compute gradients only for head params
        head_tensors = [params_for_grad[k] for k in sorted(head_param_names)]
        grads = torch.autograd.grad(
            loss,
            head_tensors,
            create_graph=False,
        )

        # SGD step on head params only
        grad_map = dict(zip(sorted(head_param_names), grads))
        new_params = {}
        for k, v in params_for_grad.items():
            if k in head_param_names:
                new_params[k] = (v - inner_lr * grad_map[k]).detach()
            else:
                new_params[k] = v.detach()

        params = new_params

        if verbose:
            print(f"    inner step {step}: value_MSE={loss.item():.4f}")

    return params


def meta_step_anil(
    model,
    optimizer,
    task_batch,
    device,
    inner_lr,
    inner_steps,
    head_param_names,
    max_grad_norm=5.0,
    verbose=False,
):
    """
    One ANIL meta-step across a batch of tasks.

    Each task: adapt value head on support, evaluate on query, accumulate
    meta-gradients. Trunk gradients come from the query loss backpropagated
    through the (frozen-in-inner-loop) trunk features.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    task_batch : list of (sX, sy_val, qX, qy_val, task_id) tuples
    device : torch.device
    inner_lr : float
    inner_steps : int
    head_param_names : set
    max_grad_norm : float
        Gradient clipping norm. Set to 0 to disable.
    verbose : bool

    Returns
    -------
    meta_loss : float
    grad_norm : float
    """
    optimizer.zero_grad(set_to_none=True)

    all_params = list(model.named_parameters())
    accum_grads = {n: torch.zeros_like(p, device=device) for n, p in all_params}

    meta_total = 0.0

    for task_idx, task in enumerate(task_batch, start=1):
        sX, sy_val, qX, qy_val, task_id = task

        sX = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_val = torch.tensor(sy_val, dtype=torch.float32, device=device)
        qX = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_val = torch.tensor(qy_val, dtype=torch.float32, device=device)

        if verbose:
            print(f"\n  Task {task_idx}/{len(task_batch)} | id={task_id}")

        base_params = {n: p for n, p in model.named_parameters()}

        # Inner loop: adapt value head only
        adapted = inner_adapt_anil(
            model=model,
            base_params=base_params,
            head_param_names=head_param_names,
            Xs=sX,
            ys_val=sy_val,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            verbose=verbose,
        )

        # Query loss with adapted params — need grads for ALL params here
        # so the trunk gets meta-gradients from the query loss
        adapted_req = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in adapted.items()
        }

        q_v = functional_call(model, adapted_req, (qX,))
        q_loss = value_loss(q_v, qy_val)

        if verbose:
            print(f"    query value_MSE={q_loss.item():.4f}")

        # Gradients w.r.t. all adapted params (trunk gets signal from query)
        q_grads = torch.autograd.grad(
            q_loss,
            tuple(adapted_req.values()),
            create_graph=False,
            allow_unused=True,
        )

        for (name, _), g in zip(adapted_req.items(), q_grads):
            if g is not None:
                accum_grads[name] += g.detach()

        meta_total += q_loss.detach().item()

    # Average across tasks
    num_tasks = len(task_batch)
    for name in accum_grads:
        accum_grads[name] /= num_tasks

    # Assign to model params
    grad_norm_sq = 0.0
    for name, p in model.named_parameters():
        p.grad = accum_grads[name]
        grad_norm_sq += float((p.grad.detach() ** 2).sum().item())
    grad_norm = grad_norm_sq ** 0.5

    # Gradient clipping
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    return meta_total / num_tasks, grad_norm
