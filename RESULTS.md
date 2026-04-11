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
- Iteration: 700
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.9139
- Train loss range (last 50): 0.8473 – 0.9972
- Latest val meta-loss: 0.9438
- Val loss range (last 10): 0.9438 – 3.3700

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 835 lines)
```
[it   682] meta_loss=0.8905 | 28530ms | 19723s
[it   683] meta_loss=0.8857 | 28488ms | 19752s
[it   684] meta_loss=0.8770 | 28616ms | 19780s
[it   685] meta_loss=0.8810 | 28668ms | 19809s
[it   686] meta_loss=0.8950 | 28255ms | 19837s
[it   687] meta_loss=0.8609 | 28454ms | 19866s
[it   688] meta_loss=0.8889 | 28569ms | 19894s
[it   689] meta_loss=0.9059 | 28548ms | 19923s
[it   690] meta_loss=0.8473 | 28508ms | 19951s
[it   691] meta_loss=0.9150 | 28294ms | 19979s
[it   692] meta_loss=0.8897 | 27977ms | 20007s
[it   693] meta_loss=0.9157 | 28746ms | 20036s
[it   694] meta_loss=0.8914 | 28451ms | 20065s
[it   695] meta_loss=0.9468 | 28268ms | 20093s
[it   696] meta_loss=0.9230 | 28517ms | 20121s
[it   697] meta_loss=0.9003 | 28644ms | 20150s
[it   698] meta_loss=0.9257 | 28075ms | 20178s
[it   699] meta_loss=0.9066 | 28469ms | 20207s
[it   700] meta_loss=0.9139 | 28716ms | 20235s
  [VAL] meta_loss=0.9438 (best=0.4949)
[it   701] meta_loss=0.8905 | 28362ms | 20290s
[it   702] meta_loss=0.8742 | 28195ms | 20318s
[it   703] meta_loss=0.9013 | 28306ms | 20347s
[it   704] meta_loss=0.9425 | 28538ms | 20375s
[it   705] meta_loss=0.9630 | 28225ms | 20403s
[it   706] meta_loss=0.9090 | 27943ms | 20431s
[it   707] meta_loss=0.9070 | 28182ms | 20459s
[it   708] meta_loss=0.9217 | 28328ms | 20488s
[it   709] meta_loss=0.9321 | 28487ms | 20516s
[it   710] meta_loss=0.8901 | 28791ms | 20545s
```
