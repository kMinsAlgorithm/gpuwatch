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
python -m gpuwatch top --interval 0.25
python -m gpuwatch train-status --root /data/kmg/Trajectory_Prediction/related_works/My_MART_HIERAR_HRT_v2
```

If NVML is unavailable in the current environment, use the deterministic fake backend for UI/debug checks:

```bash
python -m gpuwatch once --backend fake
python -m gpuwatch once --backend fake --theme soft-dark
python -m gpuwatch top --backend fake --plain --theme soft-dark
python -m gpuwatch top --backend fake --plain --ascii --theme terminal
```

The default theme is `soft-dark`. In short terminals, the monitor renders one square-ish GPU
block per device with utilization, VRAM, temperature, PID, and epoch status compressed inside
the card. Use `--theme terminal` if the caller wants unstyled terminal-native output.

For compact machine-readable output:

```bash
python -m gpuwatch json --watch --interval 1.0
python -m gpuwatch train-status --json --root <project-root>
```

## Training Progress

Prefer `TrainingRun` heartbeat instrumentation for new training code:

```python
from gpuwatch import TrainingRun

with TrainingRun("eth_seed1", total_epochs=100) as run:
    for epoch in range(100):
        run.epoch_start(epoch, total_steps=len(loader))
        for step, batch in enumerate(loader, start=1):
            loss = train_step(batch)
            run.step(epoch, step, total_steps=len(loader), loss=float(loss))
        run.epoch_end(epoch)
```

For existing trajectory experiments, `gpuwatch train-status --root ...` scans recent `.log` files and `run_*.jsonl` heartbeat files. It reports confidence and evidence; low-confidence rows should be treated as hints, not facts.

See `references/log-status-patterns.md` only when updating parser behavior.
