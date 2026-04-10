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
- Iteration: 300
- Best val meta-loss: 0.4949
- Latest train meta-loss: 3.1202
- Train loss range (last 50): 2.9852 – 3.7003
- Latest val meta-loss: 3.2201
- Val loss range (last 10): 0.4949 – 3.2291

## best.pt
- Iteration: 50
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.5630
- Train loss range (last 50): 0.5630 – 0.9944
- Latest val meta-loss: 0.4949
- Val loss range (last 10): 0.4949 – 0.4949

## Loss Curve
![Loss](runs/game_task_v1/loss.png)

## Console Log (last 30 of 455 lines)
```
[it   309] meta_loss=3.2838 | 28599ms | 8930s
[it   310] meta_loss=3.1147 | 27595ms | 8958s
[it   311] meta_loss=3.1432 | 27648ms | 8985s
[it   312] meta_loss=3.1147 | 28398ms | 9014s
[it   313] meta_loss=3.3680 | 28321ms | 9042s
[it   314] meta_loss=3.3176 | 27671ms | 9070s
[it   315] meta_loss=3.3598 | 28431ms | 9098s
[it   316] meta_loss=3.1992 | 28183ms | 9126s
[it   317] meta_loss=3.1870 | 28225ms | 9155s
[it   318] meta_loss=3.4447 | 27768ms | 9182s
[it   319] meta_loss=3.2877 | 28400ms | 9211s
[it   320] meta_loss=3.4762 | 28227ms | 9239s
[it   321] meta_loss=3.1445 | 28544ms | 9268s
[it   322] meta_loss=3.5362 | 28142ms | 9296s
[it   323] meta_loss=3.5652 | 28051ms | 9324s
[it   324] meta_loss=3.3869 | 28274ms | 9352s
[it   325] meta_loss=3.5581 | 28282ms | 9380s
[it   326] meta_loss=3.2539 | 28654ms | 9409s
[it   327] meta_loss=3.6956 | 28459ms | 9437s
[it   328] meta_loss=3.5617 | 27541ms | 9465s
[it   329] meta_loss=3.3251 | 27971ms | 9493s
[it   330] meta_loss=3.5325 | 28246ms | 9521s
[it   331] meta_loss=3.5523 | 28597ms | 9550s
[it   332] meta_loss=3.2944 | 28440ms | 9578s
[it   333] meta_loss=3.5448 | 28439ms | 9607s
[it   334] meta_loss=3.1568 | 28390ms | 9635s
[it   335] meta_loss=3.4455 | 27709ms | 9663s
[it   336] meta_loss=3.4665 | 28472ms | 9691s
[it   337] meta_loss=3.2541 | 28419ms | 9720s
[it   338] meta_loss=3.3995 | 28285ms | 9748s
```
