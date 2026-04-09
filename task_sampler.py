import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class ChessTaskSampler:
    """
    Memory-safe MAML task sampler.

    Task = one chess game (game_id)

    Strategy:
    1. Scan all shards once and load ONLY metadata (game_id arrays).
    2. Build mapping:
         game_id -> [(shard_idx, local_indices_array), ...]
    3. At sample time, load only the rows needed for one sampled task.
    """

    def __init__(
        self,
        data_dir: str,
        train_frac: float = 0.8,
        seed: int = 0,
        min_positions_per_game: int = 16,
        max_positions_per_game: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.min_positions_per_game = min_positions_per_game
        self.max_positions_per_game = max_positions_per_game

        self.shard_files = sorted(
            [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        )
        if not self.shard_files:
            raise RuntimeError(f"No .npz shards found in {self.data_dir}")

        print(f"Indexing {len(self.shard_files)} shards (metadata only)...")
        self._build_index()
        self._split_train_val(train_frac)

    def _build_index(self):
        """
        Build:
          game_to_locs[gid] = [(shard_idx, np.array(local_rows)), ...]
        """
        self.game_to_locs: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.game_counts: Dict[int, int] = defaultdict(int)
        total_samples = 0

        # Probe the first shard to see if legal_mask is present.
        with np.load(self.shard_files[0]) as d:
            self.has_legal_mask = "legal_mask" in d.files
        if self.has_legal_mask:
            print("Shards contain legal_mask — will mask illegal moves in loss.")
        else:
            print("WARNING: shards do not contain legal_mask. "
                  "Re-run db_preprocess_chess.py to enable legal move masking "
                  "(policy loss will plateau high without it).")

        for shard_idx, path in enumerate(self.shard_files):
            with np.load(path) as d:
                gids = d["game_id"].astype(np.int64)

            total_samples += len(gids)

            uniq = np.unique(gids)
            for gid in uniq:
                local_idx = np.where(gids == gid)[0].astype(np.int32)
                self.game_to_locs[int(gid)].append((shard_idx, local_idx))
                self.game_counts[int(gid)] += len(local_idx)

        # filter games
        filtered = {}
        filtered_counts = {}
        for gid, locs in self.game_to_locs.items():
            n = self.game_counts[gid]
            if n < self.min_positions_per_game:
                continue
            if self.max_positions_per_game is not None and n > self.max_positions_per_game:
                # keep the game; we can still subsample within it
                pass
            filtered[gid] = locs
            filtered_counts[gid] = n

        self.game_to_locs = filtered
        self.game_counts = filtered_counts
        self.all_game_ids = sorted(self.game_to_locs.keys())

        print(f"Total samples indexed: {total_samples}")
        print(f"Total games (after filtering): {len(self.all_game_ids)}")

        if len(self.all_game_ids) == 0:
            raise RuntimeError("No games left after filtering. Lower min_positions_per_game.")

    def _split_train_val(self, train_frac: float):
        gids = self.all_game_ids[:]
        self.rng.shuffle(gids)

        split = int(len(gids) * train_frac)
        self.train_game_ids = gids[:split]
        self.val_game_ids = gids[split:]

        print(f"Train games: {len(self.train_game_ids)} | Val games: {len(self.val_game_ids)}")

    def _sample_indices_for_game(self, gid: int, k_support: int, k_query: int):
        """
        Sample k_support + k_query position references from one game.

        Returns:
          selected_refs: list of (shard_idx, local_idx)
        """
        total_needed = k_support + k_query
        total_available = self.game_counts[gid]

        if total_available < total_needed:
            return None

        # flatten references for this game
        refs = []
        for shard_idx, local_rows in self.game_to_locs[gid]:
            for r in local_rows:
                refs.append((shard_idx, int(r)))

        chosen = self.rng.sample(refs, total_needed)
        support_refs = chosen[:k_support]
        query_refs = chosen[k_support:]
        return support_refs, query_refs

    def _load_rows(self, refs: List[Tuple[int, int]]):
        """
        Load only the requested rows from the relevant shards.

        Returns:
          X, y_policy, y_value, legal_mask  (legal_mask is None if shards lack it)
        """
        by_shard = defaultdict(list)
        for shard_idx, row_idx in refs:
            by_shard[shard_idx].append(row_idx)

        X_parts, yp_parts, yv_parts, lm_parts = [], [], [], []

        for shard_idx, rows in by_shard.items():
            rows = np.array(rows, dtype=np.int64)
            path = self.shard_files[shard_idx]

            with np.load(path) as d:
                X_parts.append(d["X"][rows])
                yp_parts.append(d["y_policy"][rows])
                yv_parts.append(d["y_value"][rows])
                if self.has_legal_mask:
                    lm_parts.append(d["legal_mask"][rows])

        X = np.concatenate(X_parts, axis=0)
        y_policy = np.concatenate(yp_parts, axis=0)
        y_value = np.concatenate(yv_parts, axis=0)
        legal_mask = np.concatenate(lm_parts, axis=0) if self.has_legal_mask else None

        return X, y_policy, y_value, legal_mask

    def sample_task(
        self,
        k_support: int,
        k_query: int,
        split: str = "train",
    ):
        game_pool = self.train_game_ids if split == "train" else self.val_game_ids
        if not game_pool:
            raise RuntimeError(f"No games in split={split}. Check train_frac.")

        while True:
            gid = self.rng.choice(game_pool)
            out = self._sample_indices_for_game(gid, k_support, k_query)
            if out is None:
                continue
            support_refs, query_refs = out
            break

        sX, sy_pol, sy_val, s_legal = self._load_rows(support_refs)
        qX, qy_pol, qy_val, q_legal = self._load_rows(query_refs)

        return (
            sX,
            sy_pol,
            sy_val,
            s_legal,
            qX,
            qy_pol,
            qy_val,
            q_legal,
            gid,
        )

    def sample_meta_batch(
        self,
        meta_batch_size: int,
        k_support: int,
        k_query: int,
        split: str = "train",
    ):
        tasks = []
        for _ in range(meta_batch_size):
            tasks.append(
                self.sample_task(
                    k_support=k_support,
                    k_query=k_query,
                    split=split,
                )
            )
        return tasks


if __name__ == "__main__":
    sampler = ChessTaskSampler(
        data_dir="./processed_chess_flat",
        train_frac=0.8,
        seed=42,
        min_positions_per_game=16,
    )

    task = sampler.sample_task(k_support=8, k_query=8, split="train")
    sX, sy_pol, sy_val, s_legal, qX, qy_pol, qy_val, q_legal, gid = task

    print("Sampled gid:", gid)
    print("Support X:", sX.shape)
    print("Query X:", qX.shape)
    if s_legal is not None:
        print("Support legal_mask:", s_legal.shape,
              "mean legal moves/pos:", float(s_legal.sum(axis=1).mean()))