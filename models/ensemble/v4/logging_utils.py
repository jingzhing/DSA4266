import os
import sys
import time
from pathlib import Path


class TeeLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        sys.__stdout__.write(message)
        self._fh.write(message)
        self._fh.flush()

    def flush(self):
        sys.__stdout__.flush()
        self._fh.flush()


def init_run_logging(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / log_name
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout = TeeLogger(log_path)
    print(f"Logging to: {log_path}")
    return str(log_path)


def log_section(title):
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")
