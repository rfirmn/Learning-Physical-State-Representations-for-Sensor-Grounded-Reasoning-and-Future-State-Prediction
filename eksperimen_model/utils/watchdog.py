import time
import threading
from tqdm import tqdm

class StallWatchdog:
    """
    Non-intrusive background watchdog that monitors loop heartbeat.
    If no batch finishes within `timeout` seconds, an alert banner is printed
    with diagnostic instructions without crashing the process.
    """
    def __init__(self, timeout: float = 60.0, check_interval: float = 5.0):
        self.timeout = timeout
        self.check_interval = check_interval
        self.last_tick = time.time()
        self.current_context = ""
        self.is_running = False
        self.stalled_alerted = False
        self.thread = None
        self._lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.last_tick = time.time()
        self.stalled_alerted = False
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def tick(self, context_msg: str = ""):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_tick
            if self.stalled_alerted and elapsed > 0.1:
                tqdm.write(f"\n>>> [STALL RECOVERED] Loop resumed activity after {elapsed:.1f}s delay. Continuing normal execution...\n")
            self.last_tick = now
            self.current_context = context_msg
            self.stalled_alerted = False

    def pause(self):
        with self._lock:
            self.last_tick = time.time()

    def _monitor(self):
        while self.is_running:
            time.sleep(self.check_interval)
            with self._lock:
                elapsed = time.time() - self.last_tick
                if elapsed > self.timeout and not self.stalled_alerted:
                    self.stalled_alerted = True
                    notice = (
                        f"\n"
                        f"+==========================================================================================+\n"
                        f"| [!] STALL NOTICE: Training loop has not completed a batch for {elapsed:.1f}s (> {self.timeout:.0f}s threshold)   |\n"
                        f"+------------------------------------------------------------------------------------------+\n"
                        f"| Current Step: {self.current_context:<74} |\n"
                        f"| Possible Causes & Diagnostics:                                                           |\n"
                        f"|  1. MM-Fi Disk I/O: Reading radar .bin files may be throttled by disk read speed.       |\n"
                        f"|  2. Windows DataLoader Workers: Multi-worker IPC can occasionally serialize or pause.   |\n"
                        f"|     -> If this persists across batches, rerun with `--num_workers 0`.                    |\n"
                        f"|  3. GPU Kernel / VRAM: PyTorch CUDA execution or memory paging might be saturated.       |\n"
                        f"| Action: Process is NOT terminated. Still waiting for batch completion...                 |\n"
                        f"+==========================================================================================+\n"
                    )
                    tqdm.write(notice)
