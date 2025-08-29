from __future__ import annotations
import csv, os, time
from typing import Dict, Any

DEFAULT_LOG = "results/run_log.csv"

def log_result(row: Dict[str, Any], csv_path: str = DEFAULT_LOG) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)