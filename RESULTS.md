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
- Iteration: 950
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.9488
- Train loss range (last 50): 0.8205 – 1.0447
- Latest val meta-loss: 0.9644
- Val loss range (last 10): 0.8068 – 1.9762

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 1089 lines)
```
[it   931] meta_loss=0.9863 | 28521ms | 26939s
[it   932] meta_loss=0.9492 | 28248ms | 26967s
[it   933] meta_loss=0.9964 | 28354ms | 26995s
[it   934] meta_loss=0.9890 | 27397ms | 27023s
[it   935] meta_loss=0.9815 | 28678ms | 27051s
[it   936] meta_loss=0.9953 | 28482ms | 27080s
[it   937] meta_loss=1.0070 | 28647ms | 27109s
[it   938] meta_loss=0.9747 | 28426ms | 27137s
[it   939] meta_loss=0.9977 | 28530ms | 27165s
[it   940] meta_loss=1.0247 | 28442ms | 27194s
[it   941] meta_loss=0.9568 | 28342ms | 27222s
[it   942] meta_loss=1.0013 | 28104ms | 27250s
[it   943] meta_loss=0.9829 | 28252ms | 27279s
[it   944] meta_loss=1.0012 | 28251ms | 27307s
[it   945] meta_loss=1.0447 | 31668ms | 27339s
[it   946] meta_loss=0.9915 | 28604ms | 27367s
[it   947] meta_loss=0.9873 | 28440ms | 27396s
[it   948] meta_loss=0.9854 | 28445ms | 27424s
[it   949] meta_loss=0.9464 | 28655ms | 27453s
[it   950] meta_loss=0.9488 | 29734ms | 27482s
  [VAL] meta_loss=0.9644 (best=0.4949)
[it   951] meta_loss=0.9813 | 28191ms | 27538s
[it   952] meta_loss=0.9809 | 28318ms | 27566s
[it   953] meta_loss=0.9673 | 28435ms | 27595s
[it   954] meta_loss=0.9405 | 28497ms | 27623s
[it   955] meta_loss=0.9913 | 27917ms | 27651s
[it   956] meta_loss=0.9537 | 28249ms | 27679s
[it   957] meta_loss=1.0187 | 28521ms | 27708s
[it   958] meta_loss=1.0103 | 28019ms | 27736s
[it   959] meta_loss=0.9778 | 28558ms | 27764s
```
