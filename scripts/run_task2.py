#!/usr/bin/env python3
"""
Simple Task 2 runner
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from task2_models import run_task2

if __name__ == "__main__":
    print("Running Task 2: Model Building and Training...")
    run_task2()
