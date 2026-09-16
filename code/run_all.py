"""Run the packaged analyses; input data are read-only."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    scripts = ["process_examples.py", "analyze_e3.py", "analyze_timing_examples.py", "analyze_timing.py", "summarize_supporting.py", "verify_statistics.py"]
    if not args.skip_plots:
        scripts += ["plot_figure8_10.py", "plot_figure9.py"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for script in scripts:
        print("Running " + script, flush=True)
        subprocess.run([sys.executable, str(HERE / script)], cwd=HERE.parent, env=env, check=True)
    print("Finished. Generated results are in outputs/.")


if __name__ == "__main__":
    main()
