"""End-to-end orchestration for CineMatch.

The full implementation is added incrementally. This module owns the command-line
entry point so the project is runnable from the beginning and remains stable as
pipeline stages are introduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cinematch.config import ProjectConfig, load_config
from cinematch.utils import ensure_directory, save_json, set_random_seed


def run_pipeline(config: ProjectConfig) -> dict[str, str]:
    """Run the end-to-end recommendation pipeline.

    Parameters
    ----------
    config:
        Typed project configuration.

    Returns
    -------
    dict[str, str]
        Minimal run metadata. Later steps will extend this with metrics and
        artifact paths.
    """

    set_random_seed(config.random_seed)
    output_dir = ensure_directory(config.artifacts.output_dir)
    metadata = {
        "project_name": config.project_name,
        "status": "initialized",
        "output_dir": str(output_dir),
    }
    save_json(metadata, output_dir / "run_metadata.json")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the CineMatch MLE pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON configuration file.",
    )
    return parser


def main() -> None:
    """Console entry point for running the recommendation pipeline."""

    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = run_pipeline(config)
    print(f"CineMatch pipeline status: {metadata['status']}")
