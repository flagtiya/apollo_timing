from __future__ import annotations

import argparse
from pathlib import Path

from apollo_timing.architecture import architecture_inventory
from apollo_timing.io.save_artifacts import ensure_output_dir, write_json, write_text


def _build_markdown(payload: list[dict[str, str]]) -> str:
    lines = [
        "# Architecture audit",
        "",
        "## Entry points by layer",
    ]
    for item in payload:
        lines.append(f"- `{item['module']}`: layer={item['layer']}, path={item['path']}")
    lines.append("")
    return "\n".join(lines)


def run_architecture_audit(output_dir: str | Path) -> list[dict[str, str]]:
    output_dir = ensure_output_dir(Path(output_dir))
    pipeline_dir = Path(__file__).resolve().parent
    payload = architecture_inventory(pipeline_dir)
    write_json(output_dir / "architecture_audit.json", payload)
    write_text(output_dir / "architecture_audit.md", _build_markdown(payload))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit pipeline modules by architectural layer.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_architecture_audit(args.output_dir)
    print(f"Audited modules: {len(payload)}")
    print(f"Artifacts written to: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
