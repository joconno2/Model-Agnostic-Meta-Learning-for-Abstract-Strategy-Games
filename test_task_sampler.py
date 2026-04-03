import numpy as np
from task_sampler import ChessTaskSampler


def main():
    sampler = ChessTaskSampler(
        data_dir="./processed_chess_flat",   # change if needed
        train_frac=0.8,
        seed=42,
        min_positions_per_game=16,
    )

    print("\n--- Single task test ---")
    task = sampler.sample_task(
        k_support=8,
        k_query=8,
        split="train",
    )

    sX, sy_pol, sy_val, qX, qy_pol, qy_val, gid = task

    print("Sampled game_id:", gid)
    print("Support X shape:", sX.shape)
    print("Support y_policy shape:", sy_pol.shape)
    print("Support y_value shape:", sy_val.shape)
    print("Query X shape:", qX.shape)
    print("Query y_policy shape:", qy_pol.shape)
    print("Query y_value shape:", qy_val.shape)

    assert sX.shape[0] == 8
    assert qX.shape[0] == 8
    assert sy_pol.shape[0] == 8
    assert sy_val.shape[0] == 8
    assert qy_pol.shape[0] == 8
    assert qy_val.shape[0] == 8

    assert len(sX.shape) == 4, f"Expected support X to be 4D, got {sX.shape}"
    assert len(qX.shape) == 4, f"Expected query X to be 4D, got {qX.shape}"

    print("\n--- Meta-batch test ---")
    tasks = sampler.sample_meta_batch(
        meta_batch_size=4,
        k_support=8,
        k_query=8,
        split="train",
    )

    print("Meta-batch size:", len(tasks))
    assert len(tasks) == 4

    gids = []
    for i, t in enumerate(tasks, start=1):
        sX, sy_pol, sy_val, qX, qy_pol, qy_val, gid = t
        gids.append(gid)

        print(f"Task {i}: gid={gid}, support={sX.shape}, query={qX.shape}")

        assert sX.shape[0] == 8
        assert qX.shape[0] == 8

    print("Sampled game_ids in meta-batch:", gids)

    print("\n--- Validation split test ---")
    val_task = sampler.sample_task(
        k_support=8,
        k_query=8,
        split="val",
    )
    print("Validation task game_id:", val_task[-1])

    print("\nAll sampler tests passed.")


if __name__ == "__main__":
    main()