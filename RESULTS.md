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

## latest.pt
- Iteration: 200
- Best val meta-loss: 0.4949
- Latest train meta-loss: 2.6953
- Train loss range (last 50): 1.1217 – 2.7062
- Latest val meta-loss: 2.7436
- Val loss range (last 10): 0.4949 – 2.7436

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 328 lines)
```
[it   185] meta_loss=2.0413 | 28171ms | 5336s
[it   186] meta_loss=1.9035 | 28315ms | 5364s
[it   187] meta_loss=2.0298 | 28484ms | 5392s
[it   188] meta_loss=2.1173 | 30748ms | 5423s
[it   189] meta_loss=2.2166 | 28429ms | 5452s
[it   190] meta_loss=2.4233 | 28416ms | 5480s
[it   191] meta_loss=2.1523 | 28225ms | 5508s
[it   192] meta_loss=2.4865 | 28131ms | 5536s
[it   193] meta_loss=2.3756 | 28343ms | 5565s
[it   194] meta_loss=2.5575 | 28133ms | 5593s
[it   195] meta_loss=2.5368 | 28253ms | 5621s
[it   196] meta_loss=2.4769 | 28386ms | 5649s
[it   197] meta_loss=2.4474 | 27509ms | 5677s
[it   198] meta_loss=2.7062 | 28336ms | 5705s
[it   199] meta_loss=2.4605 | 28081ms | 5733s
[it   200] meta_loss=2.6953 | 28358ms | 5762s
  [VAL] meta_loss=2.7436 (best=0.4949)
[it   201] meta_loss=2.6561 | 28258ms | 5817s
[it   202] meta_loss=2.7963 | 28272ms | 5845s
[it   203] meta_loss=2.7468 | 28462ms | 5874s
[it   204] meta_loss=2.9558 | 28221ms | 5902s
[it   205] meta_loss=2.9502 | 28660ms | 5931s
[it   206] meta_loss=2.9023 | 27349ms | 5958s
[it   207] meta_loss=2.9419 | 27550ms | 5986s
[it   208] meta_loss=2.7411 | 28489ms | 6014s
[it   209] meta_loss=2.8913 | 28501ms | 6043s
[it   210] meta_loss=3.0097 | 28516ms | 6071s
[it   211] meta_loss=3.3419 | 28339ms | 6099s
[it   212] meta_loss=2.9977 | 28333ms | 6128s
[it   213] meta_loss=3.1329 | 28300ms | 6156s
```
