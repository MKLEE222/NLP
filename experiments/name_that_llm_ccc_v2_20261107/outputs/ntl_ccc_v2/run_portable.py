"""Portable entry point with an explicit local MiniLM checkpoint path."""
import sys
from pathlib import Path
from threadpoolctl import threadpool_limits
import run_v2

run_v2.MODEL_PATH = Path(sys.argv[2])
(run_v2.ROOT/'work').mkdir(exist_ok=True,parents=True)
with threadpool_limits(limits=4):
    run_v2.main(Path(sys.argv[1]))
