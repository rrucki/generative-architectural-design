"""
Script to launch TensorBoard for viewing experiment logs.

This script scans the main logs directory for all run subdirectories
and launches TensorBoard to visualize them. It constructs a --logdir_spec
argument, allowing TensorBoard to display each run with a descriptive name
based on its folder name.

Example:
    # Launch TensorBoard to view all runs
    python scripts/view_logs.py
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from urban_planner.logger_config import setup_logging

logger = logging.getLogger(__name__)


def launch_tensorboard(log_dir: Path) -> None:
    """
    Finds all run directories in the log directory and launches TensorBoard.

    Args:
        log_dir: The base directory where experiment logs are stored.
    """
    if not log_dir.exists() or not log_dir.is_dir():
        logger.error(f"Log directory not found at: {log_dir}")
        raise FileNotFoundError(f"Log directory not found at: {log_dir}")

    logger.info(f"Scanning '{log_dir}' for all run directories...")

    run_directories = sorted([
        item for item in log_dir.iterdir() if item.is_dir()
    ])

    if not run_directories:
        logger.warning(f"No log directories found in '{log_dir}'. Nothing to do.")
        return

    logdir_spec = ",".join([f"{d.name}:{d.resolve()}" for d in run_directories])

    logger.info(f"Found {len(run_directories)} matching log directories.")
    logger.debug(f"Constructed --logdir_spec: {logdir_spec}")
    logger.info("Launching TensorBoard...")

    command = ["tensorboard", f"--logdir_spec={logdir_spec}"]

    try:
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "1"
        subprocess.run(command, check=True, env=env)
    except FileNotFoundError:
        logger.critical(
            "Could not launch TensorBoard. Is it installed and in your PATH?\n"
            "Try running: pip install tensorboard"
        )
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"TensorBoard exited with an error: {e}")
        raise


def main():
    """
    Parses command-line arguments and launches the TensorBoard viewer.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Launch TensorBoard to view all training logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Path to the main logs directory (default: project's 'logs' folder)."
    )

    args = parser.parse_args()

    try:
        launch_tensorboard(log_dir=args.log_dir)
        logger.info("TensorBoard session finished.")
        sys.exit(0)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("Failed to launch TensorBoard due to a previously reported error.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()