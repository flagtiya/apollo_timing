from __future__ import annotations

import argparse
from pathlib import Path

from timing_accelerator.pipeline.release_order_ablation import run_split_sky_pair_ablation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split sky-pair ablation after the EPS1-first baseline.")
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_split_sky_pair_ablation(
        par_path=args.par,
        tim_path=args.tim,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )
    print(f"Recommended variant: {comparison['recommended_variant']}")
    print(f"Artifacts written to: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
