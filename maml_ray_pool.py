"""
Distributed MAML task execution via Ray.

Each actor holds a local copy of the task sampler (data shards on local disk)
and model architecture. Per meta-iteration the driver broadcasts serialized
model weights, each actor runs a slice of the meta-batch (inner adapt on
support → query loss → gradients), and returns accumulated gradients as a
flat numpy array. The driver averages and applies them.

Data must be pre-distributed to each worker at a known path (e.g.
/tmp/maml-chess/processed_chess_flat/). Use cluster_sync or rsync.
"""

from __future__ import annotations

import hashlib
import io
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TaskResult:
    """Accumulated result from one actor's task slice."""
    grad_flat: np.ndarray       # flattened gradient array
    meta_loss_sum: float        # sum of per-task meta losses (not averaged)
    num_tasks: int              # how many tasks this actor processed
    host: str = ""


def _make_maml_actor():
    import ray

    @ray.remote
    class MAMLActor:
        """
        Stateful Ray actor for distributed MAML inner-loop execution.

        Each actor:
        - Loads data shards from local disk
        - Keeps a model architecture for functional_call
        - Receives weights each iteration, runs tasks, returns gradients
        """

        def __init__(
            self,
            data_dir: str,
            db_path: Optional[str],
            task_mode: str,
            in_channels: int,
            trunk_hidden: int,
            bottleneck_dim: int,
            value_hidden: int,
            train_frac: float,
            seed: int,
            min_positions_per_task: int,
            slot_name: str,
        ):
            import torch
            from model_v2 import ValueNet
            from task_sampler_v2 import ValueTaskSampler

            self.host = socket.gethostname()
            self.slot_name = slot_name
            self.device = torch.device("cpu")  # inner loop is cheap, CPU is fine

            # Build model (architecture only — weights will be sent each iteration)
            self.model = ValueNet(
                in_channels=in_channels,
                trunk_hidden=trunk_hidden,
                bottleneck_dim=bottleneck_dim,
                value_hidden=value_hidden,
            ).to(self.device)

            self.head_param_names = set(n for n, _ in self.model.head_params())
            self.param_names = [n for n, _ in self.model.named_parameters()]
            self.param_shapes = [p.shape for _, p in self.model.named_parameters()]

            # Load task sampler (indexes data shards on local disk)
            self.sampler = ValueTaskSampler(
                data_dir=data_dir,
                db_path=db_path,
                task_mode=task_mode,
                train_frac=train_frac,
                seed=seed + hash(slot_name) % 10000,  # different seed per actor
                min_positions_per_task=min_positions_per_task,
            )

        def ready(self) -> dict:
            return {
                "host": self.host,
                "slot": self.slot_name,
                "n_train_tasks": len(self.sampler.train_task_ids),
                "n_val_tasks": len(self.sampler.val_task_ids),
            }

        def run_tasks(
            self,
            weights_flat: np.ndarray,
            num_tasks: int,
            k_support: int,
            k_query: int,
            inner_lr: float,
            inner_steps: int,
            split: str = "train",
        ) -> dict:
            """
            Run num_tasks ANIL tasks and return accumulated gradients.

            Parameters
            ----------
            weights_flat : np.ndarray
                Flattened model parameters (float32).
            num_tasks : int
                Number of tasks to sample and process.
            k_support, k_query : int
                Positions per support/query set.
            inner_lr : float
            inner_steps : int
            split : str
                "train" or "val".

            Returns
            -------
            dict with keys: grad_flat, meta_loss_sum, num_tasks, host
            """
            import torch
            from torch.func import functional_call
            from maml_anil import inner_adapt_anil, value_loss

            # Load weights into model
            offset = 0
            state = {}
            for name, shape in zip(self.param_names, self.param_shapes):
                numel = 1
                for s in shape:
                    numel *= s
                state[name] = torch.from_numpy(
                    weights_flat[offset:offset + numel].reshape(shape).copy()
                ).float()
                offset += numel

            # Accumulate gradients
            accum_grads = {n: torch.zeros(s) for n, s in zip(self.param_names, self.param_shapes)}
            meta_loss_sum = 0.0
            tasks_done = 0

            for _ in range(num_tasks):
                try:
                    sX, sy_val, qX, qy_val, task_id = self.sampler.sample_task(
                        k_support, k_query, split=split
                    )
                except RuntimeError:
                    continue

                sX = torch.tensor(sX, dtype=torch.float32, device=self.device)
                sy_val = torch.tensor(sy_val, dtype=torch.float32, device=self.device)
                qX = torch.tensor(qX, dtype=torch.float32, device=self.device)
                qy_val = torch.tensor(qy_val, dtype=torch.float32, device=self.device)

                # Inner adapt (ANIL: only head params)
                adapted = inner_adapt_anil(
                    model=self.model,
                    base_params=state,
                    head_param_names=self.head_param_names,
                    Xs=sX,
                    ys_val=sy_val,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                )

                # Query loss with adapted params
                adapted_req = {
                    k: v.detach().clone().requires_grad_(True)
                    for k, v in adapted.items()
                }
                q_v = functional_call(self.model, adapted_req, (qX,))
                q_loss = value_loss(q_v, qy_val)

                # Gradients w.r.t. all params
                q_grads = torch.autograd.grad(
                    q_loss,
                    tuple(adapted_req[n] for n in self.param_names),
                    create_graph=False,
                    allow_unused=True,
                )

                for name, g in zip(self.param_names, q_grads):
                    if g is not None:
                        accum_grads[name] += g.detach()

                meta_loss_sum += q_loss.detach().item()
                tasks_done += 1

            # Flatten gradients for transfer
            grad_flat = np.concatenate([
                accum_grads[n].numpy().ravel() for n in self.param_names
            ]).astype(np.float32)

            return {
                "grad_flat": grad_flat,
                "meta_loss_sum": meta_loss_sum,
                "num_tasks": tasks_done,
                "host": self.host,
            }

        def run_val_tasks(
            self,
            weights_flat: np.ndarray,
            num_tasks: int,
            k_support: int,
            k_query: int,
            inner_lr: float,
            inner_steps: int,
        ) -> dict:
            """Same as run_tasks but on val split, returns loss only (no grads)."""
            import torch
            from torch.func import functional_call
            from maml_anil import inner_adapt_anil, value_loss

            offset = 0
            state = {}
            for name, shape in zip(self.param_names, self.param_shapes):
                numel = 1
                for s in shape:
                    numel *= s
                state[name] = torch.from_numpy(
                    weights_flat[offset:offset + numel].reshape(shape).copy()
                ).float()
                offset += numel

            val_loss_sum = 0.0
            tasks_done = 0

            for _ in range(num_tasks):
                try:
                    sX, sy_val, qX, qy_val, task_id = self.sampler.sample_task(
                        k_support, k_query, split="val"
                    )
                except RuntimeError:
                    continue

                sX = torch.tensor(sX, dtype=torch.float32, device=self.device)
                sy_val = torch.tensor(sy_val, dtype=torch.float32, device=self.device)
                qX = torch.tensor(qX, dtype=torch.float32, device=self.device)
                qy_val = torch.tensor(qy_val, dtype=torch.float32, device=self.device)

                adapted = inner_adapt_anil(
                    model=self.model,
                    base_params=state,
                    head_param_names=self.head_param_names,
                    Xs=sX,
                    ys_val=sy_val,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                )

                with torch.no_grad():
                    adapted_detached = {k: v.detach() for k, v in adapted.items()}
                    q_v = functional_call(self.model, adapted_detached, (qX,))
                    loss = value_loss(q_v, qy_val)

                val_loss_sum += loss.item()
                tasks_done += 1

            return {
                "val_loss_sum": val_loss_sum,
                "num_tasks": tasks_done,
                "host": self.host,
            }

    return MAMLActor


class MAMLRayPool:
    """
    Pool of Ray actors for distributed MAML training.

    Usage:
        pool = MAMLRayPool(data_dir="/tmp/maml-chess/processed_chess_flat", ...)
        pool.start()
        for iteration in range(N):
            weights = model_to_flat_numpy(model)
            grads, meta_loss = pool.meta_step(weights, tasks_per_actor=4, ...)
            # apply grads to model
        pool.shutdown()
    """

    def __init__(
        self,
        data_dir: str,
        db_path: Optional[str] = None,
        task_mode: str = "game",
        in_channels: int = 45,
        trunk_hidden: int = 64,
        bottleneck_dim: int = 64,
        value_hidden: int = 64,
        train_frac: float = 0.8,
        seed: int = 42,
        min_positions_per_task: int = 32,
        max_actors: Optional[int] = None,
    ):
        self.config = dict(
            data_dir=data_dir,
            db_path=db_path,
            task_mode=task_mode,
            in_channels=in_channels,
            trunk_hidden=trunk_hidden,
            bottleneck_dim=bottleneck_dim,
            value_hidden=value_hidden,
            train_frac=train_frac,
            seed=seed,
            min_positions_per_task=min_positions_per_task,
        )
        self.max_actors = max_actors
        self.actors = []
        self._hosts: dict[str, int] = {}

    def start(self):
        import ray

        if self.actors:
            return

        cluster_cpus = int(float(ray.cluster_resources().get("CPU", 0)))
        # MAML actors are heavier than selfplay actors (PyTorch inner loop),
        # so use fewer actors than CPUs. 1 actor per 4 CPUs is a good start.
        target = max(1, cluster_cpus // 4)
        if self.max_actors is not None:
            target = min(target, self.max_actors)

        actor_cls = _make_maml_actor()

        for i in range(target):
            slot = f"maml-{i + 1:04d}"
            actor = actor_cls.options(num_cpus=1).remote(
                slot_name=slot,
                **self.config,
            )
            self.actors.append(actor)

        # Block until all actors are ready
        ready_info = ray.get([a.ready.remote() for a in self.actors])
        for info in ready_info:
            host = info.get("host", "?")
            self._hosts[host] = self._hosts.get(host, 0) + 1

    @property
    def actor_count(self):
        return len(self.actors)

    def describe_capacity(self):
        return dict(self._hosts)

    def shutdown(self):
        try:
            import ray
        except ImportError:
            return
        for a in self.actors:
            try:
                ray.kill(a, no_restart=True)
            except Exception:
                pass
        self.actors = []
        self._hosts = {}

    def meta_step(
        self,
        weights_flat: np.ndarray,
        total_tasks: int,
        k_support: int,
        k_query: int,
        inner_lr: float,
        inner_steps: int,
    ) -> tuple[np.ndarray, float]:
        """
        Distributed meta-step: fan out tasks across actors, collect and
        average gradients.

        Returns (averaged_grad_flat, averaged_meta_loss).
        """
        import ray

        weights_ref = ray.put(weights_flat)

        # Divide tasks across actors
        n_actors = len(self.actors)
        base = total_tasks // n_actors
        remainder = total_tasks % n_actors
        tasks_per = [base + (1 if i < remainder else 0) for i in range(n_actors)]

        futures = [
            actor.run_tasks.remote(
                weights_ref, n, k_support, k_query, inner_lr, inner_steps
            )
            for actor, n in zip(self.actors, tasks_per)
            if n > 0
        ]

        results = ray.get(futures)

        # Aggregate
        total_tasks_done = sum(r["num_tasks"] for r in results)
        if total_tasks_done == 0:
            return np.zeros_like(weights_flat), 0.0

        grad_sum = np.zeros_like(weights_flat)
        loss_sum = 0.0
        for r in results:
            grad_sum += r["grad_flat"]
            loss_sum += r["meta_loss_sum"]

        return grad_sum / total_tasks_done, loss_sum / total_tasks_done

    def val_loss(
        self,
        weights_flat: np.ndarray,
        total_tasks: int,
        k_support: int,
        k_query: int,
        inner_lr: float,
        inner_steps: int,
    ) -> float:
        """Distributed validation: fan out val tasks, return averaged loss."""
        import ray

        weights_ref = ray.put(weights_flat)

        n_actors = len(self.actors)
        base = total_tasks // n_actors
        remainder = total_tasks % n_actors
        tasks_per = [base + (1 if i < remainder else 0) for i in range(n_actors)]

        futures = [
            actor.run_val_tasks.remote(
                weights_ref, n, k_support, k_query, inner_lr, inner_steps
            )
            for actor, n in zip(self.actors, tasks_per)
            if n > 0
        ]

        results = ray.get(futures)

        total_done = sum(r["num_tasks"] for r in results)
        if total_done == 0:
            return float("inf")

        return sum(r["val_loss_sum"] for r in results) / total_done
