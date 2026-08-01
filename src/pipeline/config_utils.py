"""Small configuration helpers used by the CLI layer."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
CONFIG_REQUIRED_KEYS = ("target_name", "output_dir", "llm_config_list")
ARTIFACT_SUFFIXES = {".md", ".docx", ".pdf"}
REPRODUCTION_FILES = {
    "manifest.json",
    "parsed_report.md",
    "strategy_brief.json",
    "sample_strategy.py",
    "README_strategy.md",
}


def load_structured_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping object in {file_path}")
    return payload


def build_config_overrides(args: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for attr in ("target_name", "stock_code", "target_type", "output_dir", "language"):
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value not in (None, ""):
                overrides[attr] = value
    return overrides


def _task_content(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("content", "task", "description", "analysis_task"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def extract_task_lists(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    collect_raw = payload.get("collect_tasks", payload.get("custom_collect_tasks", []))
    analysis_raw = payload.get("analysis_tasks", payload.get("custom_analysis_tasks", []))

    if not isinstance(collect_raw, list):
        collect_raw = []
    if not isinstance(analysis_raw, list):
        analysis_raw = []

    collect_tasks = [task for task in (_task_content(item) for item in collect_raw) if task]
    analysis_tasks = [task for task in (_task_content(item) for item in analysis_raw) if task]
    return collect_tasks, analysis_tasks


def load_task_lists(tasks_file: str | Path) -> Tuple[List[str], List[str]]:
    return extract_task_lists(load_structured_file(tasks_file))


def _iter_env_placeholders(value: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_env_placeholders(sub_value, child_prefix)
    elif isinstance(value, list):
        for idx, sub_value in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_env_placeholders(sub_value, child_prefix)
    elif isinstance(value, str):
        for match in ENV_PATTERN.finditer(value):
            yield prefix, match.group(1)


def validate_config_payload(
    payload: Dict[str, Any],
    *,
    strict_env: bool = False,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for key in CONFIG_REQUIRED_KEYS:
        if key not in payload or payload.get(key) in (None, ""):
            errors.append(f"Missing required config key: {key}")

    llm_config_list = payload.get("llm_config_list")
    if llm_config_list is not None:
        if not isinstance(llm_config_list, list) or not llm_config_list:
            errors.append("llm_config_list must be a non-empty list")
        else:
            for idx, item in enumerate(llm_config_list):
                if not isinstance(item, dict):
                    errors.append(f"llm_config_list[{idx}] must be an object")
                    continue
                for key in ("model_name", "api_key", "base_url"):
                    if key not in item or item.get(key) in (None, ""):
                        errors.append(f"llm_config_list[{idx}] missing required key: {key}")

    for task_key in ("custom_collect_tasks", "custom_analysis_tasks"):
        if task_key in payload and not isinstance(payload[task_key], list):
            errors.append(f"{task_key} must be a list when provided")

    missing_envs = []
    for location, env_name in _iter_env_placeholders(payload):
        if os.getenv(env_name) is None:
            missing_envs.append(f"{location}: {env_name}")
    if missing_envs:
        message = "Unresolved environment variable placeholders: " + ", ".join(missing_envs)
        if strict_env:
            errors.append(message)
        else:
            warnings.append(message)

    return errors, warnings


def validate_tasks_payload(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    has_ui_legacy_keys = "collect_tasks" in payload or "analysis_tasks" in payload
    has_simple_keys = "custom_collect_tasks" in payload or "custom_analysis_tasks" in payload
    if not has_ui_legacy_keys and not has_simple_keys:
        errors.append(
            "Tasks file must contain collect_tasks/analysis_tasks or "
            "custom_collect_tasks/custom_analysis_tasks"
        )
        return errors, warnings

    collect_tasks, analysis_tasks = extract_task_lists(payload)
    if not collect_tasks and not analysis_tasks:
        errors.append("Tasks file contains no usable collect or analysis task content")
    return errors, warnings


def validate_config_file(
    config_path: str | Path,
    *,
    tasks_file: str | Path | None = None,
    strict_env: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    config_payload: Dict[str, Any] = {}
    task_counts = {"collect": 0, "analysis": 0}

    try:
        config_payload = load_structured_file(config_path)
    except Exception as exc:
        errors.append(str(exc))
        return {
            "status": "invalid",
            "config_path": str(config_path),
            "tasks_file": str(tasks_file) if tasks_file else None,
            "errors": errors,
            "warnings": warnings,
            "task_counts": task_counts,
        }

    config_errors, config_warnings = validate_config_payload(config_payload, strict_env=strict_env)
    errors.extend(config_errors)
    warnings.extend(config_warnings)

    if tasks_file:
        try:
            tasks_payload = load_structured_file(tasks_file)
            task_errors, task_warnings = validate_tasks_payload(tasks_payload)
            errors.extend(task_errors)
            warnings.extend(task_warnings)
            collect_tasks, analysis_tasks = extract_task_lists(tasks_payload)
            task_counts = {"collect": len(collect_tasks), "analysis": len(analysis_tasks)}
        except Exception as exc:
            errors.append(str(exc))

    return {
        "status": "valid" if not errors else "invalid",
        "config_path": str(config_path),
        "tasks_file": str(tasks_file) if tasks_file else None,
        "errors": errors,
        "warnings": warnings,
        "task_counts": task_counts,
    }


def resolve_working_dir(
    config_path: str | Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = load_structured_file(config_path)
    payload.update(overrides or {})
    output_dir = payload.get("output_dir", "./outputs")
    target = str(payload.get("target_name", "unknown"))[:50]
    save_note = payload.get("save_note")
    if save_note:
        target = f"{save_note}_{target}"
    return Path(output_dir).expanduser() / target


def list_output_artifacts(
    config_path: str | Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    working_dir = resolve_working_dir(config_path, overrides=overrides)
    artifacts: List[Dict[str, Any]] = []

    if working_dir.exists():
        for path in sorted(item for item in working_dir.rglob("*") if item.is_file()):
            if path.suffix.lower() not in ARTIFACT_SUFFIXES and path.name not in REPRODUCTION_FILES:
                continue
            rel_path = path.relative_to(working_dir)
            artifact_type = "report_reproduction" if "report_reproduction" in path.parts else "report"
            artifacts.append({
                "name": path.name,
                "type": artifact_type,
                "path": str(path),
                "relative_path": str(rel_path),
                "size": path.stat().st_size,
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            })

    return {
        "working_dir": str(working_dir),
        "exists": working_dir.exists(),
        "count": len(artifacts),
        "artifacts": artifacts,
    }
