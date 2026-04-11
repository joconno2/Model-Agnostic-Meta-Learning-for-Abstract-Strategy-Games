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
- Iteration: 1200
- Best val meta-loss: 0.4949
- Latest train meta-loss: 3.4110
- Train loss range (last 50): 3.4110 – 207.7729
- Latest val meta-loss: 3.0764
- Val loss range (last 10): 0.6860 – 178.3184

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 1342 lines)
```
[it  1179] meta_loss=18.7064 | 27858ms | 34117s
[it  1180] meta_loss=16.2807 | 28533ms | 34146s
[it  1181] meta_loss=15.1095 | 28342ms | 34174s
[it  1182] meta_loss=13.7711 | 31901ms | 34206s
[it  1183] meta_loss=13.9282 | 28419ms | 34234s
[it  1184] meta_loss=12.1189 | 28546ms | 34263s
[it  1185] meta_loss=12.2577 | 28407ms | 34291s
[it  1186] meta_loss=11.0749 | 28463ms | 34320s
[it  1187] meta_loss=8.4435 | 28544ms | 34348s
[it  1188] meta_loss=10.0437 | 29567ms | 34378s
[it  1189] meta_loss=9.2483 | 28472ms | 34406s
[it  1190] meta_loss=9.4797 | 28237ms | 34434s
[it  1191] meta_loss=11.3040 | 28738ms | 34463s
[it  1192] meta_loss=12.1796 | 28223ms | 34491s
[it  1193] meta_loss=11.3274 | 28402ms | 34520s
[it  1194] meta_loss=10.4071 | 28642ms | 34548s
[it  1195] meta_loss=11.8346 | 28421ms | 34577s
[it  1196] meta_loss=10.4745 | 27861ms | 34605s
[it  1197] meta_loss=12.0865 | 28465ms | 34633s
[it  1198] meta_loss=7.5379 | 28426ms | 34662s
[it  1199] meta_loss=4.8378 | 28679ms | 34690s
[it  1200] meta_loss=3.4110 | 28513ms | 34719s
  [VAL] meta_loss=3.0764 (best=0.4949)
[it  1201] meta_loss=2.8713 | 28415ms | 34775s
[it  1202] meta_loss=2.4307 | 28370ms | 34803s
[it  1203] meta_loss=2.1321 | 28505ms | 34831s
[it  1204] meta_loss=1.7763 | 31668ms | 34863s
[it  1205] meta_loss=1.6786 | 28543ms | 34892s
[it  1206] meta_loss=1.3984 | 28500ms | 34920s
[it  1207] meta_loss=1.2465 | 28575ms | 34949s
```
