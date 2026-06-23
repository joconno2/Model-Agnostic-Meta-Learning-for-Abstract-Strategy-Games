# MAML for Cross-Game Position Evaluation (Chess and Shogi)

Meta-learning of board-game position value across chess and shogi, with an
**opening-as-task** formulation: each meta-learning task is an opening (ECO code
for chess, board fingerprint for shogi), and a value head adapts to a new opening
in a few gradient steps (ANIL).

Authors: Melanie Fernandez, Jim O'Connor, Gary Parker (Connecticut College).
Target: AIIDE 2026 (resubmission of IEEE SMC 2026 #1998).

## What this paper shows

The original submission was rejected for within-game memorization, no baselines,
and a single seed. The redesign fixes the methodology and reframes around the
on-title question: **does position-value knowledge transfer across games?**

1. **Labels are engine centipawns, not game outcome.** Win/loss labels only
   supported within-game memorization (post-adaptation MSE worse than random).
   We relabel every position with an engine evaluation, `tanh(cp/scale)`:
   Stockfish (chess) and Fairy-Stockfish `UCI_Variant=shogi` (shogi). One
   comparable, dense target across both games.

2. **Raw-board representations do NOT transfer across games.** A value net
   trained on one game, applied to the other, is *worse than random* (chess→shogi
   0.75 vs random 0.52); a jointly-trained chess+shogi net is *no better than a
   single-game net* (shogi-only 0.057 < joint 0.079, matched split, paired
   p=0.0055). The 45-channel encoding has game-specific planes, so a single-game
   trunk has untrained channels for the other game. These negatives motivate (3).

3. **A shared game-general abstraction DOES transfer.** Eight game-invariant ratio
   features (material fraction, phase, mobility-per-piece, in-check, king safety,
   hand share, promoted share, center control) put both games on one value-relevant
   scale. A head trained on one game transfers cold to the other (chess→shogi 0.26,
   shogi→chess 0.17, vs raw-board 0.75), and meta-learning over the abstraction
   adapts few-shot to the within-game ceiling (shogi→chess 0.52 → 0.17 in 5 steps).

4. **The value net plays real shogi.** In alpha-beta search it beats a
   material-counting opponent 81% and random 99.8% over 200 games. (Adaptation
   improves the value estimate but rarely changes the chosen move — an honest
   limitation we report.)

**Contribution:** cross-game position-value transfer requires a shared abstraction;
raw representations cannot do it, a compact game-general representation can — both
zero-shot and few-shot.

## Results (10 seeds, 600 tasks/cell, bootstrap 95% CIs; gameplay n=200, Wilson CI)

| Experiment | Result |
|---|---|
| Chess within-game adaptation | 0-step 0.199 → 5-step **0.108** [0.106, 0.110] |
| Joint per-game (chess / shogi), 5-step | chess 0.107 / shogi 0.079 |
| Raw-board transfer probe (chess-only→shogi, 5-step) | **0.751** vs random 0.518 — fails |
| Joint vs shogi-only (matched shogi split, 5-step) | shogi-only **0.057** < joint 0.079, paired **p=0.0055** — joint no help |
| Shared-feature transfer (zero-shot) | chess→shogi **0.255** [0.248, 0.263]; shogi→chess **0.173** [0.166, 0.180] |
| Few-shot over abstraction (shogi→chess) | 0-step 0.518 → 5-step **0.169** [0.157, 0.182]; random-body 0.273 |
| Few-shot over abstraction (chess→shogi) | ~0.20 (weak; shogi harder to squeeze); random-body 0.40 |
| Shogi gameplay vs Material / Random | **81.0%** [75.0, 85.8] / 99.8% (n=200) |
| Shogi gameplay NN-adapted vs NN-base | 53.2% [46.3, 60.0] — NS (adaptation doesn't change the move) |

Variance baselines (predict-mean MSE): chess 0.30, shogi 0.32.
Full table + per-seed JSONs: run `stats_summary.py`; data package archived in the lab
vault at `Research/Active/maml-dasg-results/`.

## Architecture

- **Deep value net** (`model_v2.py`): 4× Conv3×3(64)+ReLU → flatten → Linear(5184,64)
  bottleneck → value head Linear(64,64)+ReLU+Linear(64,1)+tanh. ANIL adapts the value
  head only (4,225 of 472,833 params). Unified 45-channel 9×9 encoding (chess
  zero-padded; shogi hand/lance/gold/promoted planes).
- **Shared-feature head** (`train_transfer*.py`): 8 game-invariant ratio features →
  MLP 64-64-1+tanh; ANIL adapts the final layer. This is where cross-game transfer works.

## Data (engine-labeled)

- **Chess:** Lichess games (Elo ≥ 1800) → positions → Stockfish depth-15 centipawns.
  ~2.85M positions for the deep net (`processed_sf_chess`), 400K for the feature track.
- **Shogi:** floodgate (wdoor) CSA games, parsed with python-shogi → 454,715 positions
  → Fairy-Stockfish depth-12 centipawns (4,939 games).
- **Combined meta-tasks:** 383 openings with ≥10 games (325 chess + 58 shogi).
  Opening = ECO (chess) or ply-2 board fingerprint (shogi).

## Pipeline

```
# Chess labels                              # Shogi labels (Fairy-Stockfish)
parser.py         lichess pgn.zst -> sqlite  parse_shogi_floodgate.py  floodgate CSA -> sqlite
stockfish_eval.py   cp labels                shogi_eval.py             cp labels (UCI_Variant=shogi)
db_preprocess_chess_sf.py  -> npz shards     db_preprocess_shogi_sf.py -> npz shards
extract_openings.py  -> openings sqlite
combine_sf_datasets.py   chess+shogi -> processed_sf_combined + sf_combined_openings.sqlite
feat_extract.py          shared game-invariant features (chess/shogi)
```

## Scripts

**Training**
- `train_disjoint.py` — ANIL/MAML, game-disjoint support/query, min-10-games openings. The deep models.
- `train_transfer.py` — plain shared-feature head; cross-game zero-shot transfer test (multi-seed, JSON out).
- `train_transfer_maml.py` — ANIL over shared features; few-shot cross-game adaptation (multi-seed). Thread-capped for many-wide runs.
- `train_supervised.py`, `train_full_maml.py` — baselines.

**Evaluation / analysis**
- `eval_sf_suite.py` — opening-disjoint adaptation curve + baselines + CCA, bootstrap CIs (`--out` to avoid clobbering).
- `eval_joint_suite.py` — joint per-game curves, chess→shogi transfer probe, matched joint-vs-shogi-only comparison.
- `eval_gameplay_shogi.py` — alpha-beta gameplay: NN-adapted / NN-base / material / random (needs `python-shogi`).
- `stats_summary.py` — assembles all result JSONs into one table (merges multi-batch seeds, bootstrap CIs, paired tests, Wilson intervals).

**Encoding / spec:** `encode_chess.py`, `encode_shogi.py`, `spec.py` (45-channel 9×9 layout).

**Engine note:** chess uses Stockfish; shogi uses Fairy-Stockfish
(`fairy-stockfish-largeboard`, `setoption name UCI_Variant value shogi`) — there is
no Stockfish for shogi.

## Reproduce the analysis

```
python stats_summary.py --dir runs_all     # all results: CIs, paired tests, Wilson intervals
```

## Superseded (kept for provenance)

- `train_value_anil_ray.py`, original same-game-support/query sampler — produced the
  within-game-memorization checkpoints the SMC reviewers flagged.
- `eval_reviewer_suite.py` — W/L-target game-disjoint eval that exposed the memorization.
- W/L labels were dropped (post-adaptation MSE ~1.2, worse than random 0.93) in favor of
  engine centipawns.

## Next directions (specced, not in this submission)

- **A2:** learn small per-game encoders into the shared latent (vs handcrafted features).
- **B:** game-as-task over many Fairy-Stockfish variants (xiangqi, crazyhouse, …) — few-shot
  adaptation to a held-out game. See `Research/Active/maml-crossgame-spec{A,B}` in the vault.
