from __future__ import annotations

import math
import os
import time
from typing import Tuple

from gpuwatch.backends.host import HostBackend
from gpuwatch.models import GpuProcessSnapshot, GpuSnapshot


class FakeGpuBackend:
    name = "fake"

    def collect(self, host_backend: HostBackend) -> Tuple[GpuSnapshot, ...]:
        now = time.time()
        wave = (math.sin(now / 3.0) + 1.0) / 2.0
        current_pid = os.getpid()
        gpus = []
        for index in range(4):
            util = int((20 + index * 13 + wave * 55) % 100)
            total = 15109
            used = int(384 + ((index + 1) * 907 + wave * 2800) % 9000)
            processes = []
            if index in (1, 3):
                fake_pid = current_pid if index == 1 else current_pid + index
                processes.append(
                    GpuProcessSnapshot(
                        pid=fake_pid,
                        gpu_index=index,
                        gpu_uuid=f"GPU-FAKE-{index:02d}",
                        gpu_memory_mb=int(used * 0.72),
                        type="C",
                        name="python",
                        username=os.environ.get("USER", "user"),
                        cmdline=(
                            "python",
                            "train.py",
                            "--run-name",
                            f"fake_gpu{index}_seed0",
                            "--gpu",
                            str(index),
                        ),
                        cwd=os.getcwd(),
                        cpu_percent=round(8.0 + wave * 35.0, 1),
                        rss_mb=2048.0 + index * 512.0,
                        create_time=now - 1200 - index * 180,
                        elapsed_s=1200 + index * 180,
                    )
                )
            gpus.append(
                GpuSnapshot(
                    index=index,
                    uuid=f"GPU-FAKE-{index:02d}",
                    name="Fake NVIDIA T4",
                    bus_id=f"00000000:{12 + index:02X}:00.0",
                    temperature_c=int(36 + index * 4 + wave * 20),
                    fan_percent=None if index == 0 else int(25 + wave * 50),
                    utilization_gpu_percent=util,
                    utilization_memory_percent=int(used / total * 100),
                    memory_used_mb=used,
                    memory_total_mb=total,
                    power_draw_w=round(18.0 + wave * 45.0 + index * 2.0, 1),
                    power_limit_w=70.0,
                    processes=tuple(processes),
                )
            )
        return tuple(gpus)
