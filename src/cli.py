"""Command-line interface for QAssistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, Optional

from src.pipeline.config_utils import (
    build_config_overrides,
    list_output_artifacts,
    validate_config_file,
)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="my_config.yaml", help="Path to YAML/JSON config file.")


def _add_config_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-name", default=None, help="Override config target_name.")
    parser.add_argument("--stock-code", default=None, help="Override config stock_code.")
    parser.add_argument("--target-type", default=None, help="Override config target_type.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--language", default=None, choices=["zh", "en"], help="Override config language.")


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _handle_report_run(args: argparse.Namespace) -> int:
    from src.pipeline.report_runner import run_report_pipeline

    result = asyncio.run(run_report_pipeline(
        config_file_path=args.config,
        config_overrides=build_config_overrides(args),
        tasks_file=args.tasks_file,
        resume=args.resume,
        max_concurrent=args.max_concurrent,
        use_llm_name=args.model_name,
        use_vlm_name=args.vlm_model_name,
        use_embedding_name=args.embedding_model_name,
        auto_generate_tasks=not args.no_auto_tasks,
    ))
    if args.json:
        _print_json(result)
    else:
        print(f"Status: {result['status']}")
        print(f"Run id: {result['run_id']}")
        print(f"Working dir: {result['working_dir']}")
        print(f"Agents: {result['agent_count']} total, {len(result['completed_agent_ids'])} completed, {len(result['skipped_agent_ids'])} skipped")
        if result.get("errors"):
            print("Errors:")
            for error in result["errors"]:
                print(f"- {error['agent_id']}: {error['error']}")
    return 0 if result.get("status") == "success" else 1


def _handle_reproduce(args: argparse.Namespace) -> int:
    from src.pipeline.reproduction_runner import run_reproduction_pipeline

    result = asyncio.run(run_reproduction_pipeline(
        pdf_path=args.pdf_path,
        config_file_path=args.config,
        config_overrides=build_config_overrides(args),
        report_id=args.report_id,
        model_name=args.model_name,
        max_pages=args.max_pages,
    ))
    summary = result.get("summary", {})
    if args.json:
        _print_json(summary)
    else:
        print("Status: success")
        print(f"Report id: {summary.get('report_id')}")
        print(f"Output dir: {summary.get('output_dir')}")
        print(f"Manifest: {summary.get('manifest')}")
        if summary.get("warnings"):
            print("Warnings:")
            for warning in summary["warnings"]:
                print(f"- {warning}")
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    result = validate_config_file(
        args.config,
        tasks_file=args.tasks_file,
        strict_env=args.strict_env,
    )
    if args.json:
        _print_json(result)
    else:
        print(f"Config status: {result['status']}")
        print(f"Config path: {result['config_path']}")
        if result.get("tasks_file"):
            counts = result.get("task_counts", {})
            print(f"Tasks file: {result['tasks_file']} ({counts.get('collect', 0)} collect, {counts.get('analysis', 0)} analysis)")
        for warning in result.get("warnings", []):
            print(f"Warning: {warning}")
        for error in result.get("errors", []):
            print(f"Error: {error}")
    return 0 if result.get("status") == "valid" else 1


def _handle_outputs_list(args: argparse.Namespace) -> int:
    result = list_output_artifacts(args.config, overrides=build_config_overrides(args))
    if args.json:
        _print_json(result)
    else:
        print(f"Working dir: {result['working_dir']}")
        if not result["exists"]:
            print("No output directory found.")
            return 0
        if not result["artifacts"]:
            print("No report artifacts found.")
            return 0
        for item in result["artifacts"]:
            print(f"- [{item['type']}] {item['relative_path']} ({item['size']} bytes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="QAssistant",
        description="QAssistant CLI for multi-agent financial research workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Run or manage full report generation.")
    report_subparsers = report_parser.add_subparsers(dest="report_command")
    report_run = report_subparsers.add_parser(
        "run",
        help="Run the full financial report pipeline.",
        description="Run the full financial report pipeline.",
    )
    _add_config_arg(report_run)
    _add_config_override_args(report_run)
    report_run.add_argument("--tasks-file", default=None, help="Optional JSON/YAML task file.")
    report_run.add_argument("--resume", action="store_true", help="Resume from existing checkpoints.")
    report_run.add_argument("--max-concurrent", type=int, default=None, help="Maximum concurrent agents per phase.")
    report_run.add_argument("--model-name", default=None, help="Data collection and text-generation model key.")
    report_run.add_argument("--vlm-model-name", default=None, help="Vision-language model key.")
    report_run.add_argument("--embedding-model-name", default=None, help="Embedding model key.")
    report_run.add_argument("--no-auto-tasks", action="store_true", help="Do not ask the LLM to generate extra tasks.")
    report_run.add_argument("--json", action="store_true", help="Print a JSON summary.")
    report_run.set_defaults(func=_handle_report_run)

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Generate sample strategy code from a local PDF report.",
        description="Generate sample strategy code from a local PDF report.",
    )
    _add_config_arg(reproduce_parser)
    _add_config_override_args(reproduce_parser)
    reproduce_parser.add_argument("--pdf-path", required=True, help="Local PDF report path.")
    reproduce_parser.add_argument("--report-id", default=None, help="Stable id for the reproduction output folder.")
    reproduce_parser.add_argument("--model-name", default=None, help="LLM model key. Defaults to DS_MODEL_NAME.")
    reproduce_parser.add_argument("--max-pages", type=int, default=None, help="Optional maximum pages to parse.")
    reproduce_parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    reproduce_parser.set_defaults(func=_handle_reproduce)

    config_parser = subparsers.add_parser("config", help="Validate configuration and task files.")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_validate = config_subparsers.add_parser("validate", help="Validate config and optional tasks file.")
    _add_config_arg(config_validate)
    config_validate.add_argument("--tasks-file", default=None, help="Optional JSON/YAML task file.")
    config_validate.add_argument("--strict-env", action="store_true", help="Treat missing environment variables as errors.")
    config_validate.add_argument("--json", action="store_true", help="Print validation result as JSON.")
    config_validate.set_defaults(func=_handle_config_validate)

    outputs_parser = subparsers.add_parser("outputs", help="Inspect generated artifacts.")
    outputs_subparsers = outputs_parser.add_subparsers(dest="outputs_command")
    outputs_list = outputs_subparsers.add_parser("list", help="List report artifacts for the configured target.")
    _add_config_arg(outputs_list)
    _add_config_override_args(outputs_list)
    outputs_list.add_argument("--json", action="store_true", help="Print artifacts as JSON.")
    outputs_list.set_defaults(func=_handle_outputs_list)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
