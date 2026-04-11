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
- Iteration: 450
- Best val meta-loss: 0.4949
- Latest train meta-loss: 1.0389
- Train loss range (last 50): 0.5761 – 1.2264
- Latest val meta-loss: 1.0506
- Val loss range (last 10): 0.4949 – 3.3700

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 582 lines)
```
[it   434] meta_loss=0.6329 | 28421ms | 12538s
[it   435] meta_loss=0.5794 | 28356ms | 12566s
[it   436] meta_loss=0.5792 | 28306ms | 12595s
[it   437] meta_loss=0.5761 | 28459ms | 12623s
[it   438] meta_loss=0.6122 | 28425ms | 12651s
[it   439] meta_loss=0.6634 | 28017ms | 12679s
[it   440] meta_loss=0.6800 | 28405ms | 12708s
[it   441] meta_loss=0.6835 | 28315ms | 12736s
[it   442] meta_loss=0.7409 | 28492ms | 12765s
[it   443] meta_loss=0.7383 | 28334ms | 12793s
[it   444] meta_loss=0.7662 | 28412ms | 12821s
[it   445] meta_loss=0.8156 | 28511ms | 12850s
[it   446] meta_loss=0.7880 | 28496ms | 12878s
[it   447] meta_loss=0.8297 | 28478ms | 12907s
[it   448] meta_loss=0.8161 | 28239ms | 12935s
[it   449] meta_loss=0.8575 | 28322ms | 12963s
[it   450] meta_loss=1.0389 | 28302ms | 12992s
  [VAL] meta_loss=1.0506 (best=0.4949)
[it   451] meta_loss=1.1032 | 28239ms | 13047s
[it   452] meta_loss=1.0574 | 28248ms | 13075s
[it   453] meta_loss=1.0600 | 28005ms | 13103s
[it   454] meta_loss=1.1752 | 28539ms | 13132s
[it   455] meta_loss=1.2134 | 27635ms | 13160s
[it   456] meta_loss=1.2307 | 28302ms | 13188s
[it   457] meta_loss=1.2565 | 28715ms | 13217s
[it   458] meta_loss=1.2721 | 28514ms | 13245s
[it   459] meta_loss=1.2967 | 28619ms | 13274s
[it   460] meta_loss=1.3422 | 28373ms | 13302s
[it   461] meta_loss=1.4167 | 28013ms | 13330s
[it   462] meta_loss=1.4229 | 28389ms | 13359s
```
