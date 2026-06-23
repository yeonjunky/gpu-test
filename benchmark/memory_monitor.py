"""Background nvidia-smi poller, used as a context manager around each
combo's full lifetime (model load + all 3 tasks). Chosen over
torch.cuda.max_memory_allocated() alone because that only sees PyTorch's own
allocator and misses CUDA driver overhead, NCCL buffers, and vLLM's
separately managed KV-cache pools -- nvidia-smi reports true device-wide
usage and is robust to any internal vLLM memory-accounting changes.
"""
import csv
import subprocess
import threading
import time
from pathlib import Path


class MemoryMonitor:
    def __init__(self, out_csv_path: str | None = None, poll_interval: float = 1.0, gpu_index: int = 0):
        self.out_csv_path = out_csv_path
        self.poll_interval = poll_interval
        self.gpu_index = gpu_index
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> float | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            mb = self._poll_once()
            if mb is not None:
                self._samples.append((time.time(), mb))
            self._stop_event.wait(self.poll_interval)

    def __enter__(self) -> "MemoryMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval * 2)
        if self.out_csv_path:
            Path(self.out_csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "memory_used_mb"])
                writer.writerows(self._samples)

    @property
    def peak_mb(self) -> float | None:
        if not self._samples:
            return None
        return max(mb for _, mb in self._samples)
