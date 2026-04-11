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
- Iteration: 800
- Best val meta-loss: 0.4949
- Latest train meta-loss: 0.9863
- Train loss range (last 50): 0.7672 – 1.0856
- Latest val meta-loss: 0.9561
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

## Console Log (last 30 of 962 lines)
```
[it   806] meta_loss=0.9971 | 28341ms | 23326s
[it   807] meta_loss=1.0145 | 28396ms | 23354s
[it   808] meta_loss=1.0544 | 27652ms | 23382s
[it   809] meta_loss=1.0351 | 28333ms | 23410s
[it   810] meta_loss=1.0212 | 28428ms | 23439s
[it   811] meta_loss=0.9919 | 27295ms | 23466s
[it   812] meta_loss=1.0235 | 28468ms | 23494s
[it   813] meta_loss=1.0464 | 28594ms | 23523s
[it   814] meta_loss=1.0562 | 28412ms | 23551s
[it   815] meta_loss=0.9961 | 28427ms | 23580s
[it   816] meta_loss=0.9861 | 28625ms | 23608s
[it   817] meta_loss=1.0208 | 28528ms | 23637s
[it   818] meta_loss=1.0120 | 28367ms | 23665s
[it   819] meta_loss=0.9943 | 28465ms | 23694s
[it   820] meta_loss=1.0386 | 28289ms | 23722s
[it   821] meta_loss=1.0318 | 28558ms | 23751s
[it   822] meta_loss=0.9908 | 27719ms | 23778s
[it   823] meta_loss=0.9586 | 28396ms | 23807s
[it   824] meta_loss=0.9719 | 28339ms | 23835s
[it   825] meta_loss=0.9768 | 28304ms | 23863s
[it   826] meta_loss=1.0131 | 28430ms | 23892s
[it   827] meta_loss=1.0151 | 28453ms | 23920s
[it   828] meta_loss=1.0018 | 28279ms | 23949s
[it   829] meta_loss=0.9738 | 28549ms | 23977s
[it   830] meta_loss=0.9866 | 28439ms | 24006s
[it   831] meta_loss=0.9858 | 28251ms | 24034s
[it   832] meta_loss=0.9295 | 28314ms | 24062s
[it   833] meta_loss=1.0080 | 28570ms | 24091s
[it   834] meta_loss=0.9843 | 28560ms | 24119s
[it   835] meta_loss=0.9948 | 31757ms | 24151s
```
