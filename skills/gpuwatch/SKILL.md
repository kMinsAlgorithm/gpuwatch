---
name: gpuwatch
description: Use when inspecting GPU utilization, GPU PIDs, VRAM, fan/temp/power, CPU load, or per-training epoch/progress with the local gpuwatch CLI/Python package.
---

# gpuwatch

Use this skill when the user asks what is running on GPUs, which PID owns VRAM, or what epoch/progress a training job has reached.

## Commands

Run read-only checks by default:

```bash
python -m gpuwatch once
python -m gpuwatch
python -m gpuwatch train-status --root /data/kmg/Trajectory_Prediction/related_works/My_MART_HIERAR_HRT_v2
python -m gpuwatch doctor --root /data/kmg/Trajectory_Prediction/related_works/My_MART_HIERAR_HRT_v2
```

If NVML is unavailable in the current environment, use the deterministic fake backend for UI/debug checks:

```bash
python -m gpuwatch once --backend fake
python -m gpuwatch once --backend fake --theme soft-dark
python -m gpuwatch --backend fake --plain --theme soft-dark
python -m gpuwatch --backend fake --plain --ascii --theme terminal
```

`python -m gpuwatch` is shorthand for `python -m gpuwatch top --interval 0.25`; pass
`--interval` directly to change the refresh rate. The default theme is `soft-dark`. Wide and
short terminals use `wide-short` run/GPU ribbons; narrow short terminals use square-ish GPU
cards. Use `--mode micro|wide-short|compact|medium|full` to force a layout and
`--explain-layout` when debugging why a layout was selected.

Useful filters:

```bash
python -m gpuwatch --gpu 0,3 --sort eta
python -m gpuwatch once --backend fake --gpu 1 --json
python -m gpuwatch train-status --failed-only --explain
```

For compact machine-readable output:

```bash
python -m gpuwatch json --watch --interval 1.0
python -m gpuwatch json --watch --limit 1 --no-training
python -m gpuwatch train-status --json --root <project-root>
```

## Training Progress

Prefer `TrainingRun` heartbeat instrumentation for new training code:

```python
from gpuwatch import TrainingRun

with TrainingRun("eth_seed1", total_epochs=100, project="trajectory") as run:
    for epoch in range(100):
        run.epoch_start(epoch, total_steps=len(loader))
        for step, batch in enumerate(loader, start=1):
            loss = train_step(batch)
            run.step(
                epoch,
                step,
                total_steps=len(loader),
                loss=float(loss),
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        run.epoch_end(epoch)
```

`TrainingRun` auto-records GPU/DDP hints from `CUDA_VISIBLE_DEVICES`, `RANK`, `LOCAL_RANK`,
`WORLD_SIZE`, and `NODE_RANK` when they are set. If a training script selects a device
internally, pass the physical GPU explicitly:

```python
with TrainingRun("eth_seed1", total_epochs=100, gpu_index=3) as run:
    ...
```

For existing trajectory experiments, `gpuwatch train-status --root ...` scans recent `.log` files and `run_*.jsonl` heartbeat files. It reports confidence and evidence; low-confidence rows should be treated as hints, not facts.

See `references/log-status-patterns.md` only when updating parser behavior.
