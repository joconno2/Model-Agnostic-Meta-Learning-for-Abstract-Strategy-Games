import torch
import matplotlib.pyplot as plt
from torch.func import functional_call

from spec import UnifiedSpec, num_channels
from model import ChessPolicyValueNet
from task_sampler import ChessTaskSampler
from maml_fomaml import meta_step_fomaml, inner_adapt_fomaml
from losses import combined_loss

def evaluate_meta_loss(
    model,
    sampler,
    device,
    k_support,
    k_query,
    lambda_value,
    inner_lr,
    inner_steps,
    num_tasks=8,
):
    """
    Validation meta-loss:
    - sample validation tasks
    - adapt on support
    - evaluate on query
    - DO NOT update model
    """
    model.eval()

    task_batch = sampler.sample_meta_batch(
        meta_batch_size=num_tasks,
        k_support=k_support,
        k_query=k_query,
        split="val",
    )

    total_loss = 0.0

    for task in task_batch:
        sX, sy_pol, sy_val, qX, qy_pol, qy_val, _ = task

        sX = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_pol = torch.tensor(sy_pol, dtype=torch.long, device=device)
        sy_val = torch.tensor(sy_val, dtype=torch.float32, device=device)

        qX = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_pol = torch.tensor(qy_pol, dtype=torch.long, device=device)
        qy_val = torch.tensor(qy_val, dtype=torch.float32, device=device)

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
            verbose=False,
        )

        logits, v = functional_call(model, adapted, (qX,))
        loss, _, _ = combined_loss(
            logits,
            v,
            qy_pol,
            qy_val,
            lambda_value=lambda_value,
        )

        total_loss += loss.item()

    model.train()
    return total_loss / num_tasks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # First official run params
    data_dir = "./processed_chess_flat"
    train_frac = 0.8
    seed = 42
    min_positions_per_game = 16

    meta_iters = 3000
    meta_batch_size = 8
    k_support = 8
    k_query = 8

    inner_lr = 0.1
    inner_steps = 1
    outer_lr = 1e-3
    lambda_value = 0.5

    val_every = 20
    val_num_tasks = 16

    # Data
    sampler = ChessTaskSampler(
        data_dir=data_dir,
        train_frac=train_frac,
        seed=seed,
        min_positions_per_game=min_positions_per_game,
    )

    # Model
    spec = UnifiedSpec()
    C = num_channels(spec)

    model = ChessPolicyValueNet(
        in_channels=C,
        n_actions=20480,
        trunk_hidden=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    # History
    train_meta_history = []
    val_meta_history = []
    val_meta_x = []

   
    # Training loop
    for it in range(1, meta_iters + 1):
        task_batch = sampler.sample_meta_batch(
            meta_batch_size=meta_batch_size,
            k_support=k_support,
            k_query=k_query,
            split="train",
        )

        meta_loss, pol_loss, val_loss, grad_norm = meta_step_fomaml(
            model=model,
            optimizer=optimizer,
            task_batch=task_batch,
            device=device,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            lambda_value=lambda_value,
            verbose=False,
        )

        train_meta_history.append(meta_loss)

        # Print only one line per iteration
        first_gid = task_batch[0][-1]
        print(f"[it {it}] gid={first_gid} meta={meta_loss:.4f}")

        # Periodic validation
        if it % val_every == 0:
            val_meta = evaluate_meta_loss(
                model=model,
                sampler=sampler,
                device=device,
                k_support=k_support,
                k_query=k_query,
                lambda_value=lambda_value,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                num_tasks=val_num_tasks,
            )
            val_meta_history.append(val_meta)
            val_meta_x.append(it)
            print(f"[it {it}] VAL meta={val_meta:.4f}")

    # Save graph only

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_meta_history) + 1), train_meta_history, label="Train Meta Loss")
    plt.plot(val_meta_x, val_meta_history, label="Validation Meta Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Meta Loss")
    plt.title("MAML Training vs Validation Meta Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("maml_train_val_loss.png", dpi=200)
    print("Saved: maml_train_val_loss.png")


if __name__ == "__main__":
    main()