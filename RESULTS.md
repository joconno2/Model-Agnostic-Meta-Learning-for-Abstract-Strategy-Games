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
- Iteration: 1050
- Best val meta-loss: 0.4949
- Latest train meta-loss: 1.1165
- Train loss range (last 50): 0.5728 – 1.9719
- Latest val meta-loss: 1.0745
- Val loss range (last 10): 0.6860 – 1.0745

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 1216 lines)
```
[it  1055] meta_loss=0.9080 | 28512ms | 30542s
[it  1056] meta_loss=0.9129 | 28444ms | 30570s
[it  1057] meta_loss=0.8763 | 28645ms | 30599s
[it  1058] meta_loss=0.8923 | 28212ms | 30627s
[it  1059] meta_loss=0.8480 | 27900ms | 30655s
[it  1060] meta_loss=0.7706 | 28538ms | 30684s
[it  1061] meta_loss=0.7572 | 28449ms | 30712s
[it  1062] meta_loss=0.8426 | 28452ms | 30740s
[it  1063] meta_loss=0.8110 | 28736ms | 30769s
[it  1064] meta_loss=0.8562 | 28464ms | 30798s
[it  1065] meta_loss=0.8824 | 28492ms | 30826s
[it  1066] meta_loss=0.8543 | 28404ms | 30855s
[it  1067] meta_loss=0.8885 | 28190ms | 30883s
[it  1068] meta_loss=0.9130 | 27995ms | 30911s
[it  1069] meta_loss=0.8075 | 28613ms | 30939s
[it  1070] meta_loss=0.8464 | 28787ms | 30968s
[it  1071] meta_loss=0.7835 | 28325ms | 30996s
[it  1072] meta_loss=0.8427 | 27857ms | 31024s
[it  1073] meta_loss=0.8080 | 28383ms | 31053s
[it  1074] meta_loss=0.8501 | 28424ms | 31081s
[it  1075] meta_loss=0.8296 | 28538ms | 31110s
[it  1076] meta_loss=0.8062 | 28543ms | 31138s
[it  1077] meta_loss=0.8383 | 28605ms | 31167s
[it  1078] meta_loss=0.8300 | 28395ms | 31195s
[it  1079] meta_loss=0.8235 | 28426ms | 31224s
[it  1080] meta_loss=0.8140 | 28426ms | 31252s
[it  1081] meta_loss=0.8636 | 28461ms | 31280s
[it  1082] meta_loss=0.8461 | 28439ms | 31309s
[it  1083] meta_loss=0.8698 | 28340ms | 31337s
[it  1084] meta_loss=0.8604 | 28510ms | 31366s
```
