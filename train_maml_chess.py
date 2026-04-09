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
    num_tasks=32,
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

        with torch.no_grad():
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


def save_checkpoint(
    filepath,
    model,
    optimizer,
    iteration,
    best_val_meta,
    meta_iters,
    meta_batch_size,
    k_support,
    k_query,
    inner_lr,
    inner_steps,
    outer_lr,
    lambda_value,
    min_positions_per_game,
    train_meta_history,
    val_meta_history,
    val_meta_x,
):
    checkpoint = {
        "iteration": iteration,
        "best_val_meta": best_val_meta,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta_iters": meta_iters,
        "meta_batch_size": meta_batch_size,
        "k_support": k_support,
        "k_query": k_query,
        "inner_lr": inner_lr,
        "inner_steps": inner_steps,
        "outer_lr": outer_lr,
        "lambda_value": lambda_value,
        "min_positions_per_game": min_positions_per_game,
        "train_meta_history": train_meta_history,
        "val_meta_history": val_meta_history,
        "val_meta_x": val_meta_x,
    }
    torch.save(checkpoint, filepath)
    print(f"Saved: {filepath}")


def save_loss_plot(train_meta_history, val_meta_history, val_meta_x, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_meta_history) + 1), train_meta_history, label="Train Meta Loss")
    plt.plot(val_meta_x, val_meta_history, label="Validation Meta Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Meta Loss")
    plt.title("MAML Training vs Validation Meta Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Final recommended settings
    data_dir = "./processed_chess_flat"
    train_frac = 0.8
    seed = 42
    min_positions_per_game = 16  # because 8 support + 8 query

    meta_iters = 2600
    meta_batch_size = 8
    k_support = 8
    k_query = 8

    inner_lr = 0.1
    inner_steps = 1
    outer_lr = 1e-3
    lambda_value = 0.5

    val_every = 50
    val_num_tasks = 32
    ckpt_every = 500

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

    best_val_meta = float("inf")

    # Save config
    with open("run_config.txt", "w") as f:
        f.write(f"data_dir={data_dir}\n")
        f.write(f"train_frac={train_frac}\n")
        f.write(f"seed={seed}\n")
        f.write(f"min_positions_per_game={min_positions_per_game}\n")
        f.write(f"meta_iters={meta_iters}\n")
        f.write(f"meta_batch_size={meta_batch_size}\n")
        f.write(f"k_support={k_support}\n")
        f.write(f"k_query={k_query}\n")
        f.write(f"inner_lr={inner_lr}\n")
        f.write(f"inner_steps={inner_steps}\n")
        f.write(f"outer_lr={outer_lr}\n")
        f.write(f"lambda_value={lambda_value}\n")
        f.write(f"val_every={val_every}\n")
        f.write(f"val_num_tasks={val_num_tasks}\n")
        f.write(f"ckpt_every={ckpt_every}\n")
    print("Saved: run_config.txt")

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

        first_gid = task_batch[0][-1]
        print(f"[TRAIN it {it}] gid={first_gid} meta={meta_loss:.4f}")

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
            print(f"[VAL   it {it}] meta={val_meta:.4f}")

            if val_meta < best_val_meta:
                best_val_meta = val_meta
                save_checkpoint(
                    filepath="chess_maml_checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    iteration=it,
                    best_val_meta=best_val_meta,
                    meta_iters=meta_iters,
                    meta_batch_size=meta_batch_size,
                    k_support=k_support,
                    k_query=k_query,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                    outer_lr=outer_lr,
                    lambda_value=lambda_value,
                    min_positions_per_game=min_positions_per_game,
                    train_meta_history=train_meta_history,
                    val_meta_history=val_meta_history,
                    val_meta_x=val_meta_x,
                )

            save_checkpoint(
                filepath="chess_maml_checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                iteration=it,
                best_val_meta=best_val_meta,
                meta_iters=meta_iters,
                meta_batch_size=meta_batch_size,
                k_support=k_support,
                k_query=k_query,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                outer_lr=outer_lr,
                lambda_value=lambda_value,
                min_positions_per_game=min_positions_per_game,
                train_meta_history=train_meta_history,
                val_meta_history=val_meta_history,
                val_meta_x=val_meta_x,
            )

        if it % ckpt_every == 0:
            save_checkpoint(
                filepath=f"chess_maml_checkpoint_it{it}.pt",
                model=model,
                optimizer=optimizer,
                iteration=it,
                best_val_meta=best_val_meta,
                meta_iters=meta_iters,
                meta_batch_size=meta_batch_size,
                k_support=k_support,
                k_query=k_query,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                outer_lr=outer_lr,
                lambda_value=lambda_value,
                min_positions_per_game=min_positions_per_game,
                train_meta_history=train_meta_history,
                val_meta_history=val_meta_history,
                val_meta_x=val_meta_x,
            )

    # Save graph
    save_loss_plot(
        train_meta_history=train_meta_history,
        val_meta_history=val_meta_history,
        val_meta_x=val_meta_x,
        outpath="maml_train_val_loss.png",
    )

    # Save final checkpoint
    save_checkpoint(
        filepath="chess_maml_checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        iteration=meta_iters,
        best_val_meta=best_val_meta,
        meta_iters=meta_iters,
        meta_batch_size=meta_batch_size,
        k_support=k_support,
        k_query=k_query,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        outer_lr=outer_lr,
        lambda_value=lambda_value,
        min_positions_per_game=min_positions_per_game,
        train_meta_history=train_meta_history,
        val_meta_history=val_meta_history,
        val_meta_x=val_meta_x,
    )


if __name__ == "__main__":
    main()