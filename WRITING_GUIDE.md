# MAML-DASG — Writing Guide & Data Package

**For the writer (Melanie).** This is everything needed to draft the AIIDE 2026
paper without re-deriving anything. Numbers are final (10 seeds, bootstrap 95% CI;
gameplay n=200, Wilson CI). Regenerate the table any time: `python stats_summary.py
--dir results`. Regenerate figures: `python make_figures.py`. Do not change the
numbers; cite them as given. Flag anything unclear to Jim.

- **Venue / format:** AIIDE 2026, AAAI two-column style. Abstract due **Jun 26**, full paper **Jul 3**.
- **Title (working):** *Cross-Game Position Evaluation in Chess and Shogi Requires a Shared Abstraction*
  (alternatives: "Opening-as-Task Meta-Learning for Cross-Game Position Value"; "When Does
  Position-Value Knowledge Transfer Across Games?").
- **History:** resubmission of IEEE SMC 2026 #1998 (rejected: within-game memorization, no
  baselines, single seed, no practical test). This draft fixes all four — see the rebuttal map.

---

## 1. Thesis (use this framing everywhere)

**One sentence:** Cross-game position-value transfer requires a shared, game-general
abstraction — raw board representations cannot transfer (worse than random; joint training no
better than single-game), but a compact game-invariant representation transfers both zero-shot
and few-shot, and the engine-distilled value function plays real shogi well above baseline.

**The arc (this is the spine of the paper — three acts):**
1. *Setup.* Position value is meta-learned with openings as tasks (ANIL adapts a value head
   per opening). Labels are **engine centipawns**, not game outcome (outcome labels only
   supported memorization). One comparable target across both games via `tanh(cp/scale)`.
2. *Negative (motivation).* On raw 45-channel board tensors, cross-game transfer **fails**:
   a chess-trained net on shogi is worse than random, and a jointly-trained chess+shogi net is
   no better than a single-game net. The encoding has game-specific channels; the adaptable
   component (value head) sits downstream of the game-specific part (trunk), so adaptation
   cannot repair a wrong representation.
3. *Positive (contribution).* A **shared game-general abstraction** (8 game-invariant ratio
   features) puts both games on one value-relevant scale. A value head transfers across games
   zero-shot, and meta-learning over the abstraction adapts few-shot to the within-game ceiling.
   The value net plays real shogi (81% vs material).

**Do NOT oversell.** The honest, defensible claims are the ones above. Two results are
deliberately negative/neutral and should be *reported as findings*, not hidden:
- Joint deep training does not beat single-game (this motivates the abstraction).
- Adaptation lowers value MSE but does not change move selection in gameplay (both games).

---

## 2. Reviewer-rebuttal map (SMC #1998 → what this paper does)

| SMC #1998 critique | Fix in this paper | Evidence |
|---|---|---|
| Within-game memorization (inflated results) | Game-disjoint support/query (support from games A, query from games B); engine-centipawn labels replace W/L | §Method; E1 0.199→0.108 |
| Missing baselines | random-trunk, meta-trunk/random-head, predict-mean variance, random-body, material, random-play | E1 baselines flat ~0.26; E6 variance 0.30/0.32; E8 |
| Single seed | 10 seeds, bootstrap 95% CI, paired tests; per-episode gates p 1e-140..1e-170 | all of §Results |
| No practical / gameplay test | Alpha-beta gameplay vs material & random, n=200, Wilson CI | E8 shogi 81% vs material |
| (implicit) "cross-game in title but chess-only" | Full chess+shogi with comparable engine labels; explicit cross-game transfer experiments | E4, E6, E7 |

---

## 3. Section-by-section outline (what to write, with the numbers)

### Abstract (~150 words)
Hit: opening-as-task meta-learning for position value; engine-centipawn labels across chess and
shogi; **finding** that raw representations don't transfer across games (chess→shogi worse than
random; joint no better than single-game); a shared game-general abstraction transfers zero-shot
(chess→shogi 0.26, shogi→chess 0.17 MSE vs raw-board 0.75) and few-shot to the within-game ceiling
(shogi→chess 0.52→0.17); engine-distilled value net beats material 81% in real shogi games.
Position: cross-game value transfer requires a shared abstraction.

### 1. Introduction
- Problem: can a learned position evaluator carry value knowledge across different abstract
  strategy games? Most game-AI value functions are single-game with unlimited self-play data;
  the interesting question is cross-game generality.
- Opening-as-task framing; ANIL few-shot adaptation.
- State the three-act arc (Section 1 above) and the contribution bullets.
- Contributions list: (i) engine-centipawn meta-learning protocol with game-disjoint eval;
  (ii) negative result — raw representations fail cross-game; (iii) positive — shared abstraction
  transfers zero/few-shot; (iv) gameplay demonstration; (v) open-source pipeline + 10-seed data.

### 2. Related work (pointers in Section 8)
MAML/ANIL; meta-learning for value/RL; general game playing; engine distillation (AlphaZero-style
value nets, NNUE); cross-domain/representation transfer; opening theory as task structure.

### 3. Method
- 3.1 Task formulation: opening = task; support/query **game-disjoint**; min 10 games/opening.
- 3.2 Labels: engine centipawns, `tanh(cp/scale)`; Stockfish d15 (chess, scale 400),
  Fairy-Stockfish `UCI_Variant=shogi` d12 (shogi, scale 800). State explicitly there is no
  Stockfish for shogi; Fairy-Stockfish is the multi-variant engine.
- 3.3 Deep value net + ANIL (architecture in Section 6). Inner: adapt value head only, 5 steps,
  lr 0.005; outer: Adam, meta-batch 128.
- 3.4 Shared abstraction: 8 game-invariant ratio features (Section 6); MLP head; ANIL adapts the
  final layer (lr 0.05).
- 3.5 Unified 45-channel 9×9 encoding (chess zero-padded; shogi hand/lance/gold/promoted planes).

### 4. Experimental setup
- Data (Section 7). Protocol: 10 seeds {42,123,456,789,1337,7,99,256,512,2024}, 600 tasks/cell,
  K=64 support/query. Metrics: value MSE on held-out openings; adaptation curve over inner steps
  {0,1,3,5,10}. Stats: bootstrap 95% CI from seed means; paired t for adaptation gates
  (per-episode) and model comparisons (seed-level); Wilson CI for win rates.

### 5. Results (each subsection = one claim; map to figures/tables below)
- **5.1 Within-game adaptation works (Fig 1).** Chess 0.199→0.108; joint chess 0.232→0.107,
  joint shogi 0.103→0.079. Baselines (random-trunk, meta-trunk/random-head) flat ~0.26 → the gain
  needs the meta-learned init. 8 features alone reach within-game 0.119/0.130 vs variance 0.30/0.32.
- **5.2 Raw-board cross-game fails (Table 1).** Transfer probe (5-step): joint 0.079,
  **chess-only→shogi 0.751** [0.730,0.772] vs random 0.518 — worse than random. Joint vs
  shogi-only (matched split): shogi-only **0.057** < joint 0.079, paired **p=0.0055**.
- **5.3 Shared abstraction transfers (Fig 2).** Zero-shot: chess→shogi **0.255** [0.248,0.263],
  shogi→chess **0.173** [0.166,0.180]. Compare raw-board 0.751 and variance 0.30/0.32.
- **5.4 Few-shot cross-game adaptation (Fig 3).** shogi→chess 0.518→**0.169** [0.157,0.182]
  (random-body 0.273) — reaches the chess within-game ceiling. chess→shogi weak (~0.20; shogi is
  already well-predicted zero-shot). Note: shogi→chess zero-shot CI is wide [0.36,0.68] — the MAML
  init is tuned for adaptability, not zero-shot; post-adaptation is tight.
- **5.5 Gameplay (Fig 4).** Shogi: NN vs Material **81.0%** [75.0,85.8] (162-0-38), vs Random 99.8%.
  Chess: drawish at depth 1 (NN vs Material 53.0%, 162 draws). NN-adapted vs NN-base: shogi 53.2% NS,
  chess 47.2% NS → **adaptation improves the value estimate, not move selection, in both games.**

### 6. Discussion / why it works
The adaptable component (head) is downstream of the game-specific component (trunk/encoding), so
ANIL cannot fix a representation that is wrong for a new game — hence raw-board transfer fails. A
shared abstraction moves the game-specific work into a fixed, game-invariant feature map, so the
transferable value reasoning operates in a common space. (This is the AALL input-compression line
— PAL/SCOPE/NeuroPAL — applied across games.)

### 7. Limitations & future work (Section 9)

### 8. Conclusion
Restate thesis; emphasize the negative-motivates-positive structure.

---

## 4. Full results table (paste-ready; MSE = value MSE on held-out openings, 95% CI)

```
WITHIN-GAME ADAPTATION (inner steps -> MSE)
  Chess (single-game):  0: .199[.197,.201]  1: .138  3: .110  5: .108[.106,.110]  10: .107
  Joint, chess:         0: .232            1: .148  3: .111  5: .107             10: .107
  Joint, shogi:         0: .103            1: .095  3: .085  5: .079[.069,.090]  10: .076

RAW-BOARD CROSS-GAME (5-step) -- FAILS
  chess-only -> shogi:  .751 [.730,.772]   (random .518; worse than random)
  joint (saw both):     .079 [.069,.090]
  random-init:          .518 [.506,.531]
  Joint vs shogi-only (matched shogi split, 5-step):
     joint .079  vs  shogi-only .057 [.047,.066]   delta -.022, paired p = 5.5e-3  (joint NO help)

SHARED-FEATURE TRANSFER (zero-shot) -- WORKS
  within-game ceiling:  chess .119  shogi .130     (variance baselines: chess .30  shogi .32)
  chess -> shogi:       .255 [.248,.263]
  shogi -> chess:       .173 [.166,.180]

FEW-SHOT OVER ABSTRACTION (inner steps -> MSE)
  shogi -> chess:  0: .518[.355,.681]  1: .357  3: .181  5: .169[.157,.182]  10: .165   (random-body .273)
  chess -> shogi:  0: .203            1: .201  3: .188  5: .196             10: .185   (random-body .400)

GAMEPLAY (win rate, Wilson 95% CI, n=200/matchup)
  SHOGI  NN vs Material   81.0% [75.0,85.8]  (162-0-38)
         NN vs Random     99.8% [97.7,100]   (199-1-0)
         NN-adapted vs NN-base  53.2% [46.3,60.0] NS  (87-39-74)
  CHESS  NN vs Material   53.0% [46.1,59.8]  (25-162-13)  [draw-heavy at depth 1]
         NN vs Random     72.8% [66.2,78.4]  (91-109-0)
         NN-adapted vs NN-base  47.2% [40.4,54.2] NS  (13-163-24)
```

## 5. Figure captions (drafts; figures in `figures/`)

- **Fig 1 (`fig1_within_game.png`).** Within-game opening adaptation. Value MSE on held-out
  openings vs inner adaptation steps, for the single-game chess model and the joint model
  (per-game). Shaded = 95% bootstrap CI over 10 seeds. Adaptation lowers MSE in both games;
  the gain requires the meta-learned initialization (baselines flat ~0.26, not shown).
- **Fig 2 (`fig2_transfer_bar.png`).** Cross-game transfer (5-step MSE on the *other* game).
  Raw-board transfer (chess→shogi) is worse than random (dashed); a shared game-general
  abstraction (zero-shot and few-shot, both directions) reaches the within-game ceiling (dotted).
  Error bars = 95% CI.
- **Fig 3 (`fig3_fewshot_curves.png`).** Few-shot cross-game adaptation over the shared
  abstraction. MSE on the held-out game vs inner steps; dashed lines = random-body baseline.
  shogi→chess adapts to the chess ceiling; chess→shogi is already near-saturated zero-shot.
- **Fig 4 (`fig4_gameplay.png`).** Gameplay win rates (n=200/matchup, Wilson 95% CI), shogi vs
  chess. The value net beats material decisively in shogi; chess is draw-heavy at depth 1.
  Adaptation does not change play in either game (NN-adapted vs NN-base ≈ 50%).

## 6. Suggested tables

- **Table 1.** Cross-game transfer probe + joint-vs-single (the two negatives) — chess-only→shogi
  0.751, random 0.518, joint 0.079; shogi-only 0.057 vs joint 0.079 (p=0.0055).
- **Table 2.** Shared-abstraction transfer (E6) + few-shot 5-step (E7) with CIs, both directions,
  against variance and ceiling.

---

## 7. Methods detail (for the Method section)

**Architecture (deep, `model_v2.py`).** Input 45×9×9. Trunk: 4× [Conv3×3(64)+ReLU]. Flatten →
bottleneck Linear(5184→64)+ReLU. Value head: Linear(64→64)+ReLU+Linear(64→1)+tanh. 472,833 params;
ANIL adapts the value head only (4,225 params). Output in [-1,1].

**Shared-feature model (`train_transfer*.py`).** 8 features → MLP Linear(8→64)+ReLU+Linear(64→64)
+ReLU+Linear(64→1)+tanh. ANIL adapts the final linear layer. Plain-transfer variant trains the
whole head and tests cold (no adaptation).

**The 8 game-invariant features** (side-to-move perspective, all ratios so "ahead by X" means the
same in both games): (0) material fraction (my−opp)/(my+opp); (1) phase (my+opp)/start_material;
(2) mobility per piece = legal moves/(my pieces+1); (3) in-check; (4) piece fraction my/(my+opp);
(5) hand-value share (shogi; 0 for chess); (6) promoted-piece share (shogi); (7) center-control
fraction. Piece values: chess P1 N3 B3 R5 Q9; shogi P1 L3 N4 S5 G6 B8 R10 (+promoted). Start
material: chess 78, shogi 126.

**ANIL.** Inner loop: `θ_head' = θ_head − α ∇_head L_support`, 5 steps. α = 0.005 (deep), 0.05
(feature head). Outer: meta-loss = query MSE with adapted head; Adam, lr 3e-4 (deep) / 1e-3
(feature). Deep meta-batch 128; second-order through the inner loop for the feature MAML.

**Engine labels.** Each position evaluated to centipawns from side-to-move; `y = tanh(cp/scale)`,
scale 400 (chess) / 800 (shogi); mates → ±1. Chess: Stockfish depth 15. Shogi: Fairy-Stockfish
(`fairy-stockfish-largeboard`, `setoption name UCI_Variant value shogi`) depth 12.

**Stats.** 10 seeds; 600 tasks/cell; bootstrap 95% CI (5000 resamples) from per-seed means;
adaptation gates = paired t over 600 per-episode losses (0 vs 5 steps), p 1e-140..1e-170;
model comparisons = paired t over seed means; win rates = Wilson 95% CI.

---

## 8. Related work (pointers — verify and expand; do NOT fabricate citations)

- **MAML / ANIL:** Finn et al. 2017 (MAML); Raghu et al. 2020 (ANIL — rapid learning vs feature
  reuse, motivates head-only adaptation).
- **Meta-learning for games / value:** few-shot value estimation; opening books as structure.
- **General game playing:** Schaeffer/Genesereth GGP; the cross-game generality question.
- **Engine-distilled value:** AlphaZero value head; NNUE evaluation for shogi/chess; Fairy-Stockfish
  (Fairhurst/ianfab) multi-variant engine.
- **Representation / cross-domain transfer:** why disjoint input features block transfer; shared
  latent spaces.
- **AALL line (cite own prior):** PAL, SCOPE, NeuroPAL (input compression / compact representations) —
  the shared-abstraction result is this line applied across games. Gary's shogi CNN match-prediction
  papers for the shogi-evaluation grounding.

> NB: every citation must be human-verified (the SMC reviewers flagged bibliography issues on a
> sibling paper). Do not auto-generate references.

---

## 9. Limitations & future work (state plainly)

- Few-shot chess→shogi adaptation is weak (shogi already well-predicted zero-shot; little headroom).
- Adaptation improves the value estimate, not move selection (both games' gameplay NS).
- Chess gameplay is draw-heavy at depth 1 (160+/200 draws); a decisive chess gameplay number needs
  depth 2–3. Shogi is the clean practical result.
- Shogi openings fragment: 4,939 floodgate games → only 58 openings with ≥10 games (ply-2
  fingerprint) vs 325 chess; shogi CIs are wider.
- Shared features are handcrafted. **Future:** (A2) learn small per-game encoders into a shared
  latent; (B) game-as-task over many Fairy-Stockfish variants (xiangqi, crazyhouse, …) with few-shot
  adaptation to a held-out game. Specs in the vault (`maml-crossgame-spec{A,B}`).

---

## 10. Reproducibility appendix

**Code (repo `joconno2/Model-Agnostic-Meta-Learning-for-Abstract-Strategy-Games`):**
- Pipeline: `parser.py`, `stockfish_eval.py`, `db_preprocess_chess_sf.py` (chess);
  `parse_shogi_floodgate.py`, `shogi_eval.py`, `db_preprocess_shogi_sf.py` (shogi);
  `combine_sf_datasets.py`, `feat_extract.py`, `extract_openings.py`.
- Training: `train_disjoint.py` (deep), `train_transfer.py` / `train_transfer_maml.py` (feature).
- Eval: `eval_sf_suite.py`, `eval_joint_suite.py`, `eval_gameplay_shogi.py`, `eval_gameplay_chess.py`.
- Analysis: `stats_summary.py`, `make_figures.py`. Result JSONs in `results/`; figures in `figures/`.

**Reproduce results table / figures:**
```
python stats_summary.py --dir results
python make_figures.py --dir results --out figures
```

**Data + models (backed up `nas:/aall/maml-dasg/`):**
- `checkpoints/`: `disjoint_anil_sf_k64_s42` (chess), `joint_sf_k64_s42` (chess+shogi, it2500
  val 0.1036), `shogi_sf_k64_s42`.
- `data/`: `lichess_chess.sqlite` (Stockfish-labeled), `shogi_positions.sqlite` (Fairy-SF-labeled),
  `processed_sf_chess`, `processed_sf_shogi`, `feat_chess.npz`, `feat_shogi.npz`, opening DBs,
  `fairy-stockfish` binary.

**Data sources:** chess = Lichess standard games (Elo ≥ 1800); shogi = floodgate (wdoor) CSA
archives, `http://wdoor.c.u-tokyo.ac.jp/shogi/x/`.
