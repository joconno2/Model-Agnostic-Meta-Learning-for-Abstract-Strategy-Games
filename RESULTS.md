# MAML Value-Only ANIL — Results

> Auto-generated. Updated each validation step.

## Config
```
bottleneck_dim=64
ckpt_every=500
data_dir=/tmp/maml-chess/processed_chess_flat
db_path=None
head_params=4225
inner_lr=0.005
inner_steps=5
k_query=16
k_support=16
max_actors=25
max_grad_norm=5.0
max_hours=24.0
meta_batch_size=128
meta_iters=5000
min_positions=32
n_actors=25
n_channels=45
out_dir=./runs/game_task_v1
outer_lr=0.0003
ray_address=ray://127.0.0.1:10001
seed=42
task_mode=game
total_params=472833
train_frac=0.8
trunk_hidden=64
val_every=50
val_tasks=128
value_hidden=64
```

## Console Log (last 30 of 127 lines)
```
[36m(MAMLActor pid=2678605, ip=136.244.224.200)[0m [TaskSampler] train: 28442 | val: 7111
[36m(MAMLActor pid=2678608, ip=136.244.224.200)[0m [TaskSampler] indexed 2820000 positions across 38532 games
[36m(MAMLActor pid=2678608, ip=136.244.224.200)[0m [TaskSampler] 35553 game-tasks after filtering
[36m(MAMLActor pid=2678608, ip=136.244.224.200)[0m [TaskSampler] train: 28442 | val: 7111
Pool ready: 25 actors across 6 hosts
  {'NL214-Lin11166': 5, 'NL214-Lin11177': 4, 'NL214-Lin11168': 6, 'dsr4': 4, 'NL214-Lin11176': 2, 'NL214-Lin11170': 4}

Training: 5000 iters, 128 tasks/iter (distributed across 25 actors)
  inner: 5 steps @ lr=0.005
  outer: Adam lr=0.0003, grad_clip=5.0

[36m(MAMLActor pid=2678607, ip=136.244.224.200)[0m [TaskSampler] indexed 2820000 positions across 38532 games
[36m(MAMLActor pid=2678607, ip=136.244.224.200)[0m [TaskSampler] 35553 game-tasks after filtering
[36m(MAMLActor pid=2678607, ip=136.244.224.200)[0m [TaskSampler] train: 28442 | val: 7111
[it     1] meta_loss=0.9402 | 28151ms | 28s
[it     2] meta_loss=0.9531 | 28379ms | 57s
[it     3] meta_loss=0.9580 | 28311ms | 85s
[it     4] meta_loss=0.9544 | 28346ms | 113s
[it     5] meta_loss=0.9331 | 28464ms | 142s
[it     6] meta_loss=0.9394 | 28476ms | 170s
[it     7] meta_loss=0.9477 | 27911ms | 198s
[it     8] meta_loss=0.9304 | 28470ms | 227s
[it     9] meta_loss=0.9319 | 28392ms | 255s
[it    10] meta_loss=0.9944 | 28235ms | 283s
[it    11] meta_loss=0.9090 | 28338ms | 311s
[it    12] meta_loss=0.9105 | 27852ms | 339s
[it    13] meta_loss=0.9395 | 28765ms | 368s
[it    14] meta_loss=0.9399 | 28261ms | 396s
[it    15] meta_loss=0.9632 | 28474ms | 425s
[it    16] meta_loss=0.9388 | 28484ms | 453s
```
