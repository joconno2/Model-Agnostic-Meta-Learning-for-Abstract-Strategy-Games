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
- Iteration: 1300
- Best val meta-loss: 0.4949
- Latest train meta-loss: 205.2226
- Train loss range (last 50): 205.2226 – 284.3658
- Latest val meta-loss: 202.0082
- Val loss range (last 10): 0.6860 – 210.6186

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 1469 lines)
```
[it  1303] meta_loss=194.8135 | 28331ms | 37732s
[it  1304] meta_loss=193.8699 | 28683ms | 37761s
[it  1305] meta_loss=190.2187 | 28474ms | 37789s
[it  1306] meta_loss=186.8876 | 28456ms | 37817s
[it  1307] meta_loss=186.1749 | 27758ms | 37845s
[it  1308] meta_loss=185.5964 | 27828ms | 37873s
[it  1309] meta_loss=186.5468 | 28417ms | 37901s
[it  1310] meta_loss=187.7093 | 28403ms | 37930s
[it  1311] meta_loss=185.2330 | 27652ms | 37958s
[it  1312] meta_loss=186.2227 | 28528ms | 37986s
[it  1313] meta_loss=181.1278 | 28387ms | 38014s
[it  1314] meta_loss=184.8105 | 28295ms | 38043s
[it  1315] meta_loss=181.5400 | 28456ms | 38071s
[it  1316] meta_loss=179.4102 | 28424ms | 38100s
[it  1317] meta_loss=179.0476 | 28241ms | 38128s
[it  1318] meta_loss=179.1138 | 28600ms | 38156s
[it  1319] meta_loss=178.2998 | 28764ms | 38185s
[it  1320] meta_loss=179.0792 | 28482ms | 38214s
[it  1321] meta_loss=180.6423 | 28513ms | 38242s
[it  1322] meta_loss=175.2314 | 28161ms | 38270s
[it  1323] meta_loss=178.6195 | 28604ms | 38299s
[it  1324] meta_loss=175.8098 | 28525ms | 38328s
[it  1325] meta_loss=176.7320 | 28220ms | 38356s
[it  1326] meta_loss=173.5028 | 28508ms | 38384s
[it  1327] meta_loss=179.8909 | 28468ms | 38413s
[it  1328] meta_loss=176.1572 | 28403ms | 38441s
[it  1329] meta_loss=177.7367 | 28441ms | 38470s
[it  1330] meta_loss=179.5979 | 28418ms | 38498s
[it  1331] meta_loss=180.7185 | 28690ms | 38527s
[it  1332] meta_loss=175.4451 | 28347ms | 38555s
```
