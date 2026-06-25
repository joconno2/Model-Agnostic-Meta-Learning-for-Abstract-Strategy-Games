# MAML-DASG — Data Brief (for writeup)

Dense fact package for the AIIDE 2026 resubmission. All MSE = value MSE on held-out
openings; targets are engine centipawns squashed `tanh(cp/scale)` in [-1,1].
Protocol: 10 seeds, 600 tasks/cell, bootstrap 95% CI; gameplay n=200, Wilson CI.
Regenerate: `python stats_summary.py --dir results`. Figures: `python make_figures.py`.

## The arc (3 claims)

**Claim 0 (setup).** W/L labels only support within-game memorization: game-disjoint
eval of the original checkpoints gave post-adaptation MSE ~1.2 vs random 0.93. Fix:
relabel with engine centipawns (Stockfish for chess, Fairy-Stockfish `UCI_Variant=shogi`).

**Claim 1 — within-game adaptation works (the floor).**
- Chess: 0-step 0.199 [0.197, 0.201] → 5-step 0.108 [0.106, 0.110]. Gain +0.090, paired p 1e-140..1e-170 (per-episode). Baselines (random-trunk, meta-trunk/random-head) flat ~0.26 → the gain needs the meta-learned init, not generic SGD.
- 8 handcrafted shared features alone hit within-game MSE 0.119 (chess) / 0.130 (shogi) vs variance 0.30/0.32 — they carry most of the value signal.

**Claim 2 — raw-board cross-game FAILS (the motivation, two negatives).**
- Transfer: chess-only model on shogi = **0.751 [0.730, 0.772]**, vs random 0.518. Worse than random — negative transfer (disjoint piece channels).
- Joint no help: matched-split shogi, shogi-only **0.057 [0.047, 0.066]** < joint **0.079 [0.069, 0.090]**, paired **p=0.0055**. Training on both games does not beat single-game. (An earlier "joint helps +0.046" was an unmatched-split artifact; reversed under matched eval.)

**Claim 3 — shared abstraction WORKS (the contribution).**
8 game-invariant ratio features (material fraction, phase, mobility/piece, in-check,
king safety, hand share, promoted share, center) on engine labels.
- Zero-shot transfer (head trained one game, tested cold on other):
  chess→shogi **0.255 [0.248, 0.263]**, shogi→chess **0.173 [0.166, 0.180]**.
  (vs raw-board 0.75, variance 0.30/0.32.)
- Few-shot (ANIL over the abstraction):
  shogi→chess 0-step 0.518 → 5-step **0.169 [0.157, 0.182]**, random-body 0.273 — reaches chess within-game ceiling.
  chess→shogi ~0.20 (weak; shogi label harder to squeeze), random-body 0.40.
  Note: shogi→chess zero-shot CI is wide [0.36, 0.68] across seeds (the MAML init is tuned for adaptability, not zero-shot); post-adaptation is tight.

**Claim 4 — the value net plays real shogi (practical test).**
Joint net in alpha-beta (depth 1), n=200/matchup:
- NN-base vs Material **81.0% [75.0, 85.8]** (162-0-38)
- NN-base vs Random 99.8% [97.7, 100.0]
- Material vs Random 99.0% (sanity; material is a real baseline, net still beats it 4:1)
- NN-adapted vs NN-base 53.2% [46.3, 60.0] — **NS**. Adaptation lowers value MSE but rarely changes the chosen move (99.2% greedy move agreement). Honest limitation.

## Headline
Cross-game position-value transfer requires a shared abstraction. Raw representations
cannot do it (transfer worse than random; joint training no better than single-game);
a compact game-general representation can — zero-shot and few-shot — and the
engine-distilled value net plays real shogi above baseline. Answers both SMC #1998
hits (within-game memorization + no practical test) and is on-title for
"Cross-Game Position Evaluation in Chess and Shogi."

## Limitations to state
- Few-shot chess→shogi adaptation is weak (shogi already well-predicted zero-shot).
- Adaptation improves the estimate, not move selection, in shogi gameplay.
- Shogi openings fragment hard: 4,939 floodgate games → 58 openings with ≥10 games (ply-2 fingerprint) vs 325 chess; shogi CIs are wider.
- Shared features are handcrafted (Spec A1). Learned per-game encoders into a shared latent (Spec A2) and game-as-task over many Fairy-SF variants (Spec B) are the next steps.

## Data / models
- Chess: Lichess (Elo≥1800), Stockfish depth-15 cp. Shogi: floodgate CSA, Fairy-SF depth-12 cp (454,715 positions, 4,939 games).
- Models (backed up `nas:/aall/maml-dasg/checkpoints/`): `disjoint_anil_sf_k64_s42` (chess), `joint_sf_k64_s42` (chess+shogi, it2500 val 0.1036), `shogi_sf_k64_s42`.
- Result JSONs: `results/`. Figures: `figures/` (fig1 within-game, fig2 transfer bar, fig3 few-shot curves, fig4 gameplay).
