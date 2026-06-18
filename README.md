# MAML for Cross-Game Position Evaluation

ANIL/MAML applied to board game position evaluation across Chess and Shogi. Opening-as-task formulation: each meta-learning task is an opening family (ECO code for chess, joseki for shogi). A shared convolutional trunk learns spatial features; a small value head (4,225 of 472,833 params) adapts per opening in 5 gradient steps.

Authors: Melanie Fernandez, Jim O'Connor, Gary Parker (Connecticut College).

## Status

SMC 2026 submission #1998 rejected Jun 16. Reviewer feedback valid: missing baselines, no per-game split, single seed. Revision in progress targeting AIIDE 2026 (abstract Jun 26, full Jul 3).

**Critical finding (Jun 17):** Original training used same-game support/query positions within each opening task. Proper game-disjoint evaluation (support from games A, query from games B) reveals the original results were inflated by within-game memorization. Retraining with game-disjoint sampling is underway.

## Architecture

- Trunk: 4x Conv3x3 (64 filters) + ReLU, flatten, Linear(5184, 64) bottleneck
- Value head: Linear(64, 64) + ReLU + Linear(64, 1) + tanh
- 472,833 total params, 4,225 adapted in inner loop (ANIL)
- Unified 45-channel 9x9 board encoding (chess zero-padded, shogi hand pieces in channels 26-45)

## Data

- Chess: 2.82M positions, 38,608 Lichess games (Elo >= 1800), 446 ECO codes
- Shogi: ~1.03M positions, ~9,105 Lishogi games, 77 opening codes
- Combined: 3.85M positions, 523 codes (302 with >= 10 games after filtering)

## Scripts

### Training

- `train_disjoint.py` - **Current.** ANIL or full MAML with game-disjoint support/query sampling. Filters openings to min 10 games. Single-machine, CPU-only.
- `train_value_anil_ray.py` - Original Ray-distributed ANIL training. Uses old task sampler (same-game support/query). **Produces flawed checkpoints.**
- `train_supervised.py` - Supervised baseline (same architecture, no meta-learning).
- `train_full_maml.py` - Full MAML (all params adapted). Uses old sampler. Superseded by `train_disjoint.py --mode maml`.

### Evaluation

- `eval_reviewer_suite.py` - Comprehensive eval: adaptation curve, per-game split, cross-transfer, random trunk baseline, CCA analysis. Uses game-disjoint sampling and 95% CIs.
- `eval_gameplay.py` - Alpha-beta minimax with ANIL value function. Requires `pip install chess`.
- `ablation_steps_support.py` - Inner-step x support-size grid ablation.

### Data preprocessing

- `encode_chess.py`, `encode_shogi.py` - Position encoding to 45-channel tensors
- `db_preprocess_chess.py`, `db_preprocess_shogi.py` - SQLite DB + .npz shard generation
- `spec.py` - Channel layout specification

## Runs in progress (DGX Spark, Jun 17)

| Run | Script | Data | Mode | Seed | Status |
|-----|--------|------|------|------|--------|
| disjoint_anil_combined_s42 | train_disjoint.py | combined | anil | 42 | Running, ~22s/iter |
| disjoint_anil_chess_s42 | train_disjoint.py | chess | anil | 42 | Running, ~22s/iter |

tmux session `maml` on informatics-spark2 (136.244.224.24). `CUDA_VISIBLE_DEVICES=` to avoid GPU interference with evo-distill.

## Key results

### Original checkpoints (flawed: same-game support/query)

Training MSE looked good (chess-only 0.0366, cross-game 0.0516) but proper game-disjoint evaluation shows MSE ~1.2, worse than random baseline (0.93). The trunk learned within-game correlation features, not generalizable position evaluation.

### What game-disjoint evaluation revealed

| Condition | 0 steps | 5 steps | Notes |
|-----------|---------|---------|-------|
| Meta-learned (old ckpt) | 1.48 | 1.23 | Worse than random |
| Random trunk | 0.93 | 0.94 | Predicting ~0 for everything |
| Random baseline (MSE on {-1,+1}) | 1.00 | - | Theoretical floor |

The new training runs with game-disjoint sampling should produce representations that actually generalize.
