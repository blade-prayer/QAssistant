"""Reusable report-reproduction runner."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional


def _load_default_runtime():
    from dotenv import load_dotenv

    load_dotenv()
    from src.agents import ReportReproductionAgent
    from src.config import Config
    from src.memory import Memory
    from src.utils import setup_logger

    return {
        "Config": Config,
        "Memory": Memory,
        "ReportReproductionAgent": ReportReproductionAgent,
        "setup_logger": setup_logger,
    }


def resolve_reproduction_model_name(config: Any, requested_model: Optional[str]) -> str:
    if requested_model:
        return requested_model
    env_model = os.getenv("DS_MODEL_NAME")
    if env_model:
        return env_model
    llm_dict = getattr(config, "llm_dict", {}) or {}
    if llm_dict:
        return next(iter(llm_dict.keys()))
    raise ValueError("No LLM model configured. Set DS_MODEL_NAME or pass --model-name.")


async def run_reproduction_pipeline(
    *,
    pdf_path: str,
    config_file_path: str = "my_config.yaml",
    config_overrides: Optional[Dict[str, Any]] = None,
    report_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_pages: Optional[int] = None,
    config_cls: Any = None,
    memory_cls: Any = None,
    agent_cls: Any = None,
    runtime_helpers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime = runtime_helpers or {}
    if config_cls is None or memory_cls is None or agent_cls is None:
        runtime.update(_load_default_runtime())
        config_cls = config_cls or runtime["Config"]
        memory_cls = memory_cls or runtime["Memory"]
        agent_cls = agent_cls or runtime["ReportReproductionAgent"]

    config = config_cls(config_file_path=config_file_path, config_dict=config_overrides or {})
    selected_model_name = resolve_reproduction_model_name(config, model_name)
    memory = memory_cls(config=config)

    setup_logger = runtime.get("setup_logger")
    logger = logging.getLogger("QAssistant.reproduce")
    if setup_logger:
        logger = setup_logger(log_dir=os.path.join(config.working_dir, "logs"), log_level=logging.INFO)

    logger.info(f"Report reproduction started: pdf_path={pdf_path}")
    agent = agent_cls(
        config=config,
        memory=memory,
        use_llm_name=selected_model_name,
        enable_code=False,
    )
    result = await agent.async_run(
        input_data={
            "pdf_path": pdf_path,
            "report_id": report_id,
            "max_pages": max_pages,
        },
        resume=False,
    )
    summary = {
        "status": "success",
        "report_id": result.get("report_id"),
        "output_dir": result.get("output_dir"),
        "manifest": result.get("final_result"),
        "warnings": (result.get("manifest") or {}).get("warnings", []),
    }
    logger.info("Report reproduction finished: " + json.dumps(summary, ensure_ascii=False))
    return {**result, "summary": summary}
