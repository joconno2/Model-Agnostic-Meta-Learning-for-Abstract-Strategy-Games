"""
Value-only MAML task sampler with multiple task definitions.

Task definitions:
  - "game"    : one game = one task (original). z alternates +1/-1 each ply
                (side-to-move perspective). Positions within a game are highly
                correlated, so inner loop tends to memorize rather than learn
                generalizable features.
  - "opening" : one ECO opening code = one task. Support/query drawn from
                different games with the same opening. Inner loop learns
                "what does a good position look like in the Sicilian?"
  - "player"  : one player = one task. Support/query drawn from games by
                the same player (as either color). Inner loop learns
                "what does this player's winning positions look like?"

All modes use the same sharded .npz format from db_preprocess_chess.py,
but "opening" and "player" modes require an additional metadata index
built from the original SQLite database.
"""

import os
import random
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class ValueTaskSampler:
    """
    Memory-safe MAML task sampler for value-only training.

    Returns (sX, sy_val, qX, qy_val, task_id) tuples -- no policy labels.
    """

    # Max positions to keep in memory per grouped task (opening/player).
    # 50 positions * 446 tasks * 14.6 KB = ~325 MB per actor.
    MAX_POS_PER_TASK = 50

    def __init__(
        self,
        data_dir: str,
        db_path: Optional[str] = None,
        task_mode: str = "game",
        train_frac: float = 0.8,
        seed: int = 0,
        min_positions_per_task: int = 16,
    ):
        self.data_dir = data_dir
        self.task_mode = task_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.min_positions_per_task = min_positions_per_task

        self.shard_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        )
        if not self.shard_files:
            raise RuntimeError(f"No .npz shards found in {data_dir}")

        print(f"[TaskSampler] mode={task_mode} | {len(self.shard_files)} shards")

        # Build game_id -> shard locations index (always needed)
        self._build_game_index()

        if task_mode == "game":
            self._build_game_tasks()
        elif task_mode in ("opening", "player"):
            if db_path is None:
                raise ValueError(f"task_mode={task_mode!r} requires db_path to the SQLite DB")
            self._build_grouped_tasks(db_path, task_mode)
            self._preload_grouped_data()
        else:
            raise ValueError(f"Unknown task_mode={task_mode!r}. Use 'game', 'opening', or 'player'.")

        self._split_train_val(train_frac)

    def _build_game_index(self):
        """Map game_id -> [(shard_idx, local_row_indices)]."""
        self.game_to_locs: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.game_counts: Dict[int, int] = defaultdict(int)
        total = 0

        for shard_idx, path in enumerate(self.shard_files):
            with np.load(path) as d:
                gids = d["game_id"].astype(np.int64)
            total += len(gids)
            for gid in np.unique(gids):
                local_idx = np.where(gids == gid)[0].astype(np.int32)
                self.game_to_locs[int(gid)].append((shard_idx, local_idx))
                self.game_counts[int(gid)] += len(local_idx)

        print(f"[TaskSampler] indexed {total} positions across {len(self.game_counts)} games")

    def _build_game_tasks(self):
        """Task = one game. Filter by min positions."""
        self.task_to_games: Dict[str, List[int]] = {}
        for gid, count in self.game_counts.items():
            if count >= self.min_positions_per_task:
                self.task_to_games[str(gid)] = [gid]

        self.all_task_ids = sorted(self.task_to_games.keys())
        print(f"[TaskSampler] {len(self.all_task_ids)} game-tasks after filtering")

    def _build_grouped_tasks(self, db_path: str, mode: str):
        """
        Task = one opening (ECO code) or one player.
        Group games by opening/player, then each group is a task.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        if mode == "opening":
            rows = conn.execute(
                "SELECT id, eco FROM games WHERE eco IS NOT NULL AND eco != ''"
            ).fetchall()
            group_key = lambda r: r["eco"]
        else:  # player
            rows_w = conn.execute("SELECT id, white AS player FROM games WHERE white IS NOT NULL").fetchall()
            rows_b = conn.execute("SELECT id, black AS player FROM games WHERE black IS NOT NULL").fetchall()
            rows = rows_w + rows_b
            group_key = lambda r: r["player"]

        conn.close()

        # Group game_ids by the key
        groups: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            key = group_key(r)
            gid = int(r["id"])
            if gid in self.game_counts:
                groups[key].append(gid)

        # Deduplicate game lists and filter by total positions
        self.task_to_games: Dict[str, List[int]] = {}
        for key, gids in groups.items():
            gids = list(set(gids))
            total_positions = sum(self.game_counts.get(g, 0) for g in gids)
            if total_positions >= self.min_positions_per_task:
                self.task_to_games[key] = gids

        self.all_task_ids = sorted(self.task_to_games.keys())
        print(f"[TaskSampler] {len(self.all_task_ids)} {mode}-tasks after filtering")

    def _preload_grouped_data(self):
        """
        Pre-load positions for each grouped task into memory.

        Streams through each shard once, distributing positions to their
        task's in-memory buffer. Caps at MAX_POS_PER_TASK per task to
        keep total memory around 1-2 GB.
        """
        # Invert: game_id -> task_id
        game_to_task: Dict[int, str] = {}
        for tid, gids in self.task_to_games.items():
            for gid in gids:
                game_to_task[gid] = tid

        # Pre-allocate per-task collectors
        task_X: Dict[str, List[np.ndarray]] = defaultdict(list)
        task_yv: Dict[str, List[np.ndarray]] = defaultdict(list)
        task_counts: Dict[str, int] = defaultdict(int)

        cap = self.MAX_POS_PER_TASK

        for shard_idx, path in enumerate(self.shard_files):
            with np.load(path) as d:
                X = d["X"]
                yv = d["y_value"]
                gids = d["game_id"].astype(np.int64)

            # For each game in this shard, add positions to its task
            for gid in np.unique(gids):
                tid = game_to_task.get(int(gid))
                if tid is None:
                    continue
                if task_counts[tid] >= cap:
                    continue

                mask = gids == gid
                remaining = cap - task_counts[tid]
                rows = np.where(mask)[0][:remaining]

                task_X[tid].append(X[rows])
                task_yv[tid].append(yv[rows])
                task_counts[tid] += len(rows)

        # Concatenate into final arrays
        self._task_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tid in self.all_task_ids:
            if tid in task_X and task_X[tid]:
                self._task_data[tid] = (
                    np.concatenate(task_X[tid], axis=0),
                    np.concatenate(task_yv[tid], axis=0),
                )

        total_pos = sum(len(v[1]) for v in self._task_data.values())
        mem_mb = sum(v[0].nbytes + v[1].nbytes for v in self._task_data.values()) / 1e6
        print(f"[TaskSampler] preloaded {total_pos} positions for {len(self._task_data)} tasks ({mem_mb:.0f} MB)")

        # Free the shard index -- no longer needed for grouped modes
        del self.game_to_locs
        del self.game_counts

    def _split_train_val(self, train_frac: float):
        ids = self.all_task_ids[:]
        self.rng.shuffle(ids)
        split = int(len(ids) * train_frac)
        self.train_task_ids = ids[:split]
        self.val_task_ids = ids[split:]
        print(f"[TaskSampler] train: {len(self.train_task_ids)} | val: {len(self.val_task_ids)}")

    def _load_rows(self, refs: List[Tuple[int, int]]):
        """Load specific rows from shards (game mode only). Returns X [N,C,H,W] and y_value [N]."""
        by_shard = defaultdict(list)
        for shard_idx, row_idx in refs:
            by_shard[shard_idx].append(row_idx)

        X_parts, yv_parts = [], []
        for shard_idx, rows in by_shard.items():
            rows = np.array(rows, dtype=np.int64)
            with np.load(self.shard_files[shard_idx]) as d:
                X_parts.append(d["X"][rows])
                yv_parts.append(d["y_value"][rows])

        return np.concatenate(X_parts, axis=0), np.concatenate(yv_parts, axis=0)

    def sample_task(self, k_support: int, k_query: int, split: str = "train"):
        """
        Sample one task: k_support + k_query positions.

        For game-tasks: loads from disk via shard index.
        For opening/player-tasks: samples from preloaded in-memory data.

        Returns (sX, sy_val, qX, qy_val, task_id).
        """
        pool = self.train_task_ids if split == "train" else self.val_task_ids
        total_needed = k_support + k_query

        if hasattr(self, "_task_data"):
            # Grouped mode: sample from preloaded data
            for _ in range(100):
                task_id = self.rng.choice(pool)
                if task_id not in self._task_data:
                    continue
                X_all, yv_all = self._task_data[task_id]
                if len(yv_all) < total_needed:
                    continue
                idx = self.rng.sample(range(len(yv_all)), total_needed)
                idx = np.array(idx)
                sX = X_all[idx[:k_support]]
                sy_val = yv_all[idx[:k_support]]
                qX = X_all[idx[k_support:]]
                qy_val = yv_all[idx[k_support:]]
                return sX, sy_val, qX, qy_val, task_id
            raise RuntimeError(f"Could not find a task with >= {total_needed} positions after 100 tries")

        # Game mode: load from disk
        for _ in range(100):
            task_id = self.rng.choice(pool)
            game_ids = self.task_to_games[task_id]

            all_refs = []
            for gid in game_ids:
                for shard_idx, local_rows in self.game_to_locs.get(gid, []):
                    for r in local_rows:
                        all_refs.append((shard_idx, int(r)))

            if len(all_refs) < total_needed:
                continue

            chosen = self.rng.sample(all_refs, total_needed)
            support_refs = chosen[:k_support]
            query_refs = chosen[k_support:]

            sX, sy_val = self._load_rows(support_refs)
            qX, qy_val = self._load_rows(query_refs)

            return sX, sy_val, qX, qy_val, task_id

        raise RuntimeError(f"Could not find a task with >= {total_needed} positions after 100 tries")

    def sample_meta_batch(self, meta_batch_size: int, k_support: int, k_query: int, split: str = "train"):
        return [
            self.sample_task(k_support, k_query, split)
            for _ in range(meta_batch_size)
        ]
