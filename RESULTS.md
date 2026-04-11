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

## Console Log (last 30 of 1492 lines)
```
[it  1326] meta_loss=173.5028 | 28508ms | 38384s
[it  1327] meta_loss=179.8909 | 28468ms | 38413s
[it  1328] meta_loss=176.1572 | 28403ms | 38441s
[it  1329] meta_loss=177.7367 | 28441ms | 38470s
[it  1330] meta_loss=179.5979 | 28418ms | 38498s
[it  1331] meta_loss=180.7185 | 28690ms | 38527s
[it  1332] meta_loss=175.4451 | 28347ms | 38555s
Log channel is reconnecting. Logs produced while the connection was down can be found on the head node of the cluster in `ray_client_server_[port].out`
2026-04-11 03:00:56,203	WARNING dataclient.py:403 -- Encountered connection issues in the data channel. Attempting to reconnect.
2026-04-11 03:01:26,410	WARNING dataclient.py:410 -- Failed to reconnect the data channel
Traceback (most recent call last):
  File "/home/csadmin/research/maml-dasg/train_value_anil_ray.py", line 336, in <module>
    main()
  File "/home/csadmin/research/maml-dasg/train_value_anil_ray.py", line 250, in main
    grad_flat, meta_loss = pool.meta_step(
  File "/home/csadmin/research/maml-dasg/maml_ray_pool.py", line 405, in meta_step
    results = ray.get(futures)
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 102, in wrapper
    return getattr(ray, func.__name__)(*args, **kwargs)
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/util/client/api.py", line 42, in get
    return self.worker.get(vals, timeout=timeout)
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/util/client/worker.py", line 433, in get
    res = self._get(to_get, op_timeout)
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/util/client/worker.py", line 454, in _get
    for chunk in resp:
  File "/home/csadmin/research/maml-dasg/.venv/lib/python3.10/site-packages/ray/util/client/worker.py", line 355, in _get_object_iterator
    raise ConnectionError("Client is shutting down.")
ConnectionError: Client is shutting down.
```
