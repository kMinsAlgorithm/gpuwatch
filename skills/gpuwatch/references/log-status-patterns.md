# Log Status Patterns

`gpuwatch` recognizes these trajectory-style status lines:

```text
[INFO][run_name][Epoch 2/100] train_loss: ... | val_FDE: ... | val_ADE: ... | best_epoch: 1
[ETH][TRAIN] Epoch 2
[ETH][TEST] Epoch 2
[ETH][TRAIN] Epochs: 02/99| It: 0040/0800 | Loss: ...
[BEST][eth][seed 1] epoch=1 ADE=0.62781 FDE=0.89431 score=1.52212
```

Structured heartbeat files are preferred over regex parsing. The default heartbeat directory is:

```text
~/.cache/gpuwatch/runs
```

Additional heartbeat directories can be supplied with `GPUWATCH_EXTRA_RUN_DIRS`, separated by `os.pathsep`.
