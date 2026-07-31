"""Compatibility wrapper for the main report-generation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional

from src.pipeline.report_runner import run_report_pipeline


IF_RESUME = True
MAX_CONCURRENT = 3


async def run_report(
    resume: bool = IF_RESUME,
    max_concurrent: Optional[int] = MAX_CONCURRENT,
    config_file_path: str = "my_config.yaml",
    tasks_file: Optional[str] = None,
):
    return await run_report_pipeline(
        config_file_path=config_file_path,
        tasks_file=tasks_file,
        resume=resume,
        max_concurrent=max_concurrent,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FinSight report-generation pipeline.")
    parser.add_argument("--config", default="my_config.yaml", help="Path to YAML/JSON config file.")
    parser.add_argument("--tasks-file", default=None, help="Optional task JSON/YAML file.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=IF_RESUME, help="Resume from checkpoints.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start without loading checkpoints.")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Maximum concurrent agents per phase.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args(argv)

    result = asyncio.run(run_report(
        resume=args.resume,
        max_concurrent=args.max_concurrent,
        config_file_path=args.config,
        tasks_file=args.tasks_file,
    ))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Report pipeline finished: status={result['status']}, working_dir={result['working_dir']}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
