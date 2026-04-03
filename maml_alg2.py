# maml_fomaml.py
import torch
from torch.nn.utils.stateless import functional_call
from losses import combined_loss


def inner_adapt_fomaml(
    model,
    base_params: dict,
    Xs: torch.Tensor,
    ys_pol: torch.Tensor,
    ys_val: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    lambda_value: float,
    verbose: bool = False,
):
    """
    First-order inner adaptation (no second derivatives).
    """
    params = {k: v for k, v in base_params.items()}

    for step in range(1, inner_steps + 1):
        # require grads for this step
        p_req = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}

        logits, v_pred = functional_call(model, p_req, (Xs,))  # forward with these params
        loss, lp, lv = combined_loss(logits, v_pred, ys_pol, ys_val, lambda_value=lambda_value)

        grads = torch.autograd.grad(loss, tuple(p_req.values()), create_graph=False)

        # SGD step
        params = {k: (p_req[k] - inner_lr * g).detach() for (k, _), g in zip(p_req.items(), grads)}

        if verbose:
            print(f"    inner step {step}: support_total={loss.item():.4f} "
                  f"policy_CE={lp.item():.4f} value_MSE={lv.item():.4f}")

    return params


def meta_step(
    model,
    optimizer,
    task_batch,
    device: torch.device,
    inner_lr: float,
    inner_steps: int,
    lambda_value: float,
    verbose: bool = False,
):
    """
    One meta-iteration across a meta-batch of tasks.
    Prints detailed trace if verbose=True.
    """
    optimizer.zero_grad(set_to_none=True)

    named_params = list(model.named_parameters())
    accum_grads = {n: torch.zeros_like(p, device=device) for n, p in named_params}

    meta_total = 0.0
    pol_total = 0.0
    val_total = 0.0

    gids = []

    for t_i, task in enumerate(task_batch, start=1):
        sX, sy_pol, sy_val, qX, qy_pol, qy_val, gid = task
        gids.append(int(gid))

        # to torch
        sX = torch.from_numpy(sX).to(device=device, dtype=torch.float32)
        sy_pol = torch.from_numpy(sy_pol).to(device=device, dtype=torch.long)
        sy_val = torch.from_numpy(sy_val).to(device=device, dtype=torch.float32)

        qX = torch.from_numpy(qX).to(device=device, dtype=torch.float32)
        qy_pol = torch.from_numpy(qy_pol).to(device=device, dtype=torch.long)
        qy_val = torch.from_numpy(qy_val).to(device=device, dtype=torch.float32)

        base_params = {n: p for n, p in model.named_parameters()}

        if verbose:
            print(f"\n  Task {t_i}/{len(task_batch)} (gid={gid})")
            print(f"    support batch: X={tuple(sX.shape)} y_pol={tuple(sy_pol.shape)} y_val={tuple(sy_val.shape)}")
            print(f"    query   batch: X={tuple(qX.shape)} y_pol={tuple(qy_pol.shape)} y_val={tuple(qy_val.shape)}")

        # inner adaptation
        adapted = inner_adapt_fomaml(
            model=model,
            base_params=base_params,
            Xs=sX,
            ys_pol=sy_pol,
            ys_val=sy_val,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            lambda_value=lambda_value,
            verbose=verbose,
        )

        # query loss on adapted params
        adapted_req = {k: v.detach().clone().requires_grad_(True) for k, v in adapted.items()}
        q_logits, q_v = functional_call(model, adapted_req, (qX,))
        q_loss, q_lp, q_lv = combined_loss(q_logits, q_v, qy_pol, qy_val, lambda_value=lambda_value)

        if verbose:
            print(f"    query loss: total={q_loss.item():.4f} policy_CE={q_lp.item():.4f} value_MSE={q_lv.item():.4f}")

        q_grads = torch.autograd.grad(q_loss, tuple(adapted_req.values()), create_graph=False)

        for (name, _), g in zip(adapted_req.items(), q_grads):
            accum_grads[name] += g.detach()

        meta_total += float(q_loss.detach().item())
        pol_total += float(q_lp.item())
        val_total += float(q_lv.item())

    # average grads across tasks
    m = len(task_batch)
    for name in accum_grads:
        accum_grads[name] /= m

    # assign grads to real model params
    grad_norm_sq = 0.0
    for name, p in model.named_parameters():
        p.grad = accum_grads[name]
        grad_norm_sq += float((p.grad.detach() ** 2).sum().item())
    grad_norm = grad_norm_sq ** 0.5

    optimizer.step()

    if verbose:
        print("\n  Meta-batch gids:", gids)
        print(f"  Meta-step grad L2 norm: {grad_norm:.6f}")

    return meta_total / m, pol_total / m, val_total / m, grad_norm