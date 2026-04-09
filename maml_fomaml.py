import torch
from torch.func import functional_call
from losses import combined_loss


def inner_adapt_fomaml(
    model,
    base_params,
    Xs,
    ys_pol,
    ys_val,
    inner_lr,
    inner_steps,
    lambda_value,
    legal_mask=None,
    verbose=False,
):
    """
    First-order inner-loop adaptation on the support set.
    """
    params = {k: v for k, v in base_params.items()}

    for step in range(1, inner_steps + 1):
        # Make differentiable copies for this step
        p_req = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in params.items()
        }

        logits, v_pred = functional_call(model, p_req, (Xs,))
        loss, lp, lv = combined_loss(
            logits, v_pred, ys_pol, ys_val,
            lambda_value=lambda_value,
            legal_mask=legal_mask,
        )

        grads = torch.autograd.grad(
            loss,
            tuple(p_req.values()),
            create_graph=False,
            allow_unused=True,
        )

        # Replace unused grads with zeros
        new_params = {}
        for (k, p), g in zip(p_req.items(), grads):
            if g is None:
                g = torch.zeros_like(p)
            new_params[k] = (p - inner_lr * g).detach()

        params = new_params

        if verbose:
            print(
                f"    inner step {step}: "
                f"support_total={loss.item():.4f} "
                f"policy_CE={lp.item():.4f} "
                f"value_MSE={lv.item():.4f}"
            )

    return params


def meta_step_fomaml(
    model,
    optimizer,
    task_batch,
    device,
    inner_lr,
    inner_steps,
    lambda_value,
    grad_clip=1.0,
    verbose=False,
):
    """
    One FOMAML meta-step across a batch of tasks.
    """
    optimizer.zero_grad(set_to_none=True)

    named_params = list(model.named_parameters())

    accum_grads = {
        n: torch.zeros_like(p, device=device)
        for n, p in named_params
    }

    meta_total = 0.0
    pol_total = 0.0
    val_total = 0.0
    task_gids = []

    for task_idx, task in enumerate(task_batch, start=1):
        sX, sy_pol, sy_val, s_legal, qX, qy_pol, qy_val, q_legal, gid = task
        task_gids.append(int(gid))

        sX = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_pol = torch.tensor(sy_pol, dtype=torch.long, device=device)
        sy_val = torch.tensor(sy_val, dtype=torch.float32, device=device)

        qX = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_pol = torch.tensor(qy_pol, dtype=torch.long, device=device)
        qy_val = torch.tensor(qy_val, dtype=torch.float32, device=device)

        s_legal_t = (
            torch.tensor(s_legal, dtype=torch.bool, device=device)
            if s_legal is not None else None
        )
        q_legal_t = (
            torch.tensor(q_legal, dtype=torch.bool, device=device)
            if q_legal is not None else None
        )

        if verbose:
            print(f"\n  Task {task_idx}/{len(task_batch)} | gid={gid}")
            print(
                f"    support: X={tuple(sX.shape)} "
                f"y_pol={tuple(sy_pol.shape)} y_val={tuple(sy_val.shape)}"
            )
            print(
                f"    query:   X={tuple(qX.shape)} "
                f"y_pol={tuple(qy_pol.shape)} y_val={tuple(qy_val.shape)}"
            )

        base_params = {n: p for n, p in model.named_parameters()}

        adapted = inner_adapt_fomaml(
            model=model,
            base_params=base_params,
            Xs=sX,
            ys_pol=sy_pol,
            ys_val=sy_val,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            lambda_value=lambda_value,
            legal_mask=s_legal_t,
            verbose=verbose,
        )

        adapted_req = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in adapted.items()
        }

        q_logits, q_v = functional_call(model, adapted_req, (qX,))
        q_loss, q_lp, q_lv = combined_loss(
            q_logits, q_v, qy_pol, qy_val,
            lambda_value=lambda_value,
            legal_mask=q_legal_t,
        )

        if verbose:
            print(
                f"    query loss: total={q_loss.item():.4f} "
                f"policy_CE={q_lp.item():.4f} "
                f"value_MSE={q_lv.item():.4f}"
            )

        q_grads = torch.autograd.grad(
            q_loss,
            tuple(adapted_req.values()),
            create_graph=False,
            allow_unused=True,
        )

        for (name, p), g in zip(adapted_req.items(), q_grads):
            if g is None:
                g = torch.zeros_like(p)
            accum_grads[name] += g.detach()

        meta_total += float(q_loss.detach().item())
        pol_total += float(q_lp.item())
        val_total += float(q_lv.item())

    num_tasks = len(task_batch)

    for name in accum_grads:
        accum_grads[name] /= num_tasks

    for name, p in model.named_parameters():
        p.grad = accum_grads[name]

    # Gradient clipping on the meta-gradient for stability.
    if grad_clip is not None and grad_clip > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    else:
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += float((p.grad.detach() ** 2).sum().item())
        grad_norm = grad_norm_sq ** 0.5

    optimizer.step()

    if verbose:
        print(f"\n  Meta-batch gids: {task_gids}")
        print(f"  Meta-step grad L2 norm (post-clip): {grad_norm:.6f}")

    return (
        meta_total / num_tasks,
        pol_total / num_tasks,
        val_total / num_tasks,
        grad_norm,
    )
