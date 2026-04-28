"""Generate batch movie recommendations from trained CineMatch artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from cinematch.inference import generate_recommendations, load_user_ids


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the batch recommendation CLI parser."""

    parser = argparse.ArgumentParser(description="Generate CineMatch recommendations.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing trained CineMatch artifacts.",
    )
    parser.add_argument(
        "--user-file",
        type=Path,
        required=True,
        help="Text file with one user id per line, or CSV with a userId column.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/batch_recommendations.csv"),
        help="CSV path where recommendations will be written.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100,
        help="Number of candidates to retrieve before ranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to keep per user.",
    )
    return parser


def main() -> None:
    """Load users, generate recommendations, and save a CSV."""

    parser = build_arg_parser()
    args = parser.parse_args()
    user_ids = load_user_ids(args.user_file)
    recommendations = generate_recommendations(
        artifact_dir=args.artifact_dir,
        user_ids=user_ids,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output_path, index=False)
    print(f"Wrote {len(recommendations)} recommendations to {args.output_path}")


if __name__ == "__main__":
    main()
