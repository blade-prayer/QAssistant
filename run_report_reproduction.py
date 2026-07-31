"""Compatibility wrapper for the report-reproduction pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional

from src.pipeline.reproduction_runner import run_reproduction_pipeline


async def run_report_reproduction(args):
    return await run_reproduction_pipeline(
        pdf_path=args.pdf_path,
        config_file_path=args.config,
        config_overrides={"output_dir": args.output_dir} if args.output_dir else {},
        report_id=args.report_id,
        model_name=args.model_name,
        max_pages=args.max_pages,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate sample strategy code from a local PDF report.")
    parser.add_argument("--pdf-path", required=True, help="Local PDF report path.")
    parser.add_argument("--config", default="my_config.yaml", help="Path to YAML/JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Base output directory. Uses config by default.")
    parser.add_argument("--report-id", default=None, help="Optional stable report id for the output folder.")
    parser.add_argument("--model-name", default=None, help="LLM model key from config. Defaults to DS_MODEL_NAME.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional maximum pages to parse.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args(argv)

    result = asyncio.run(run_report_reproduction(args))
    summary = result.get("summary", {})
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Report reproduction finished: output_dir={summary.get('output_dir')}")
        if summary.get("warnings"):
            print("Warnings: " + "; ".join(summary["warnings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
