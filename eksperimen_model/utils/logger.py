import os
import json
import time
from typing import Dict, Any, List

class ExperimentLogger:
    """
    Structured logger for training experiments.
    Saves history to JSON and prints formatted metrics.
    """
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{experiment_name}_history.json")
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float):
        entry = {
            'epoch': epoch,
            'lr': lr,
            'timestamp': time.time(),
            'train': train_metrics,
            'val': val_metrics
        }
        self.history.append(entry)

        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Pretty console printout
        val_mpjpe_mm = val_metrics.get('mpjpe', 0.0) * 1000.0 # Convert meters to mm
        print(f"Epoch [{epoch:03d}] | "
              f"Train Loss: {train_metrics.get('loss', 0.0):.4f} | "
              f"Val MPJPE: {val_metrics.get('mpjpe', 0.0):.4f} m ({val_mpjpe_mm:.1f} mm) | "
              f"LR: {lr:.2e}")
