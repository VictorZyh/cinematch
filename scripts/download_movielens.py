"""Download and extract MovieLens datasets.

This script intentionally uses only the Python standard library so a fresh
environment can fetch the development dataset before installing the package.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MovieLensDataset:
    """Download metadata for a supported MovieLens dataset."""

    name: str
    url: str
    expected_dir_name: str


SUPPORTED_DATASETS = {
    "latest-small": MovieLensDataset(
        name="latest-small",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        expected_dir_name="ml-latest-small",
    ),
    "latest": MovieLensDataset(
        name="latest",
        url="https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        expected_dir_name="ml-latest",
    ),
}

REQUIRED_FILES = ("ratings.csv", "movies.csv")


def _download_file(url: str, destination: Path) -> None:
    """Download a URL to a local file path."""

    with urllib.request.urlopen(url, timeout=60) as response:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def _validate_extracted_dataset(dataset_dir: Path) -> None:
    """Validate that the extracted dataset contains files used by the pipeline."""

    missing_files = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Extracted dataset at {dataset_dir} is missing required files: {missing_files}"
        )


def download_movielens_dataset(
    dataset_name: str = "latest-small",
    output_dir: str | Path = "data/raw",
    force: bool = False,
) -> Path:
    """Download and extract a supported MovieLens dataset.

    Parameters
    ----------
    dataset_name:
        Dataset key. Supported values are ``latest-small`` and ``latest``.
    output_dir:
        Directory where the extracted dataset folder should be placed.
    force:
        If True, overwrite an existing extracted dataset directory.

    Returns
    -------
    Path
        Path to the extracted dataset directory.
    """

    if dataset_name not in SUPPORTED_DATASETS:
        supported = sorted(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")

    dataset = SUPPORTED_DATASETS[dataset_name]
    raw_output_dir = Path(output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_output_dir / dataset.expected_dir_name

    if dataset_dir.exists() and not force:
        _validate_extracted_dataset(dataset_dir)
        return dataset_dir

    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)

    with tempfile.TemporaryDirectory() as temporary_dir:
        archive_path = Path(temporary_dir) / f"{dataset.expected_dir_name}.zip"
        _download_file(dataset.url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(raw_output_dir)

    _validate_extracted_dataset(dataset_dir)
    return dataset_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset downloads."""

    parser = argparse.ArgumentParser(description="Download a MovieLens dataset.")
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS),
        default="latest-small",
        help="MovieLens dataset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where the dataset folder will be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the dataset directory if it already exists.",
    )
    return parser


def main() -> None:
    """Download the requested MovieLens dataset and print its local path."""

    parser = build_arg_parser()
    args = parser.parse_args()
    dataset_dir = download_movielens_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"MovieLens dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    main()
