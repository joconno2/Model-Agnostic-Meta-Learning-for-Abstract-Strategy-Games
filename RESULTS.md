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
- Iteration: 550
- Best val meta-loss: 0.4949
- Latest train meta-loss: 1.8453
- Train loss range (last 50): 1.1736 – 2.0063
- Latest val meta-loss: 1.9762
- Val loss range (last 10): 1.0196 – 3.3700

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 708 lines)
```
[it   557] meta_loss=1.6665 | 28278ms | 16119s
[it   558] meta_loss=1.6592 | 28521ms | 16148s
[it   559] meta_loss=1.6501 | 28552ms | 16176s
[it   560] meta_loss=1.5927 | 30346ms | 16207s
[it   561] meta_loss=1.6930 | 28555ms | 16235s
[it   562] meta_loss=1.5692 | 28585ms | 16264s
[it   563] meta_loss=1.7354 | 28552ms | 16292s
[it   564] meta_loss=1.6778 | 27795ms | 16320s
[it   565] meta_loss=1.6060 | 28508ms | 16349s
[it   566] meta_loss=1.0867 | 28270ms | 16377s
[it   567] meta_loss=1.8172 | 27777ms | 16405s
[it   568] meta_loss=2.2602 | 28354ms | 16433s
[it   569] meta_loss=2.9018 | 28053ms | 16461s
[it   570] meta_loss=3.4939 | 28329ms | 16489s
[it   571] meta_loss=6.3840 | 27883ms | 16517s
[it   572] meta_loss=4.8604 | 28267ms | 16546s
[it   573] meta_loss=13.5077 | 28013ms | 16574s
[it   574] meta_loss=19.2241 | 28633ms | 16602s
[it   575] meta_loss=9.5813 | 28477ms | 16631s
[it   576] meta_loss=4.1757 | 28136ms | 16659s
[it   577] meta_loss=3.3626 | 28705ms | 16688s
[it   578] meta_loss=2.0857 | 28404ms | 16716s
[it   579] meta_loss=4.5939 | 28374ms | 16744s
[it   580] meta_loss=1.6391 | 28366ms | 16773s
[it   581] meta_loss=1.4185 | 28348ms | 16801s
[it   582] meta_loss=1.9182 | 28567ms | 16830s
[it   583] meta_loss=1.8845 | 28594ms | 16858s
[it   584] meta_loss=1.7986 | 28204ms | 16886s
[it   585] meta_loss=280.0478 | 28555ms | 16915s
[it   586] meta_loss=1.6958 | 28563ms | 16944s
```
