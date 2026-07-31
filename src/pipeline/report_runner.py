"""Reusable main report-generation runner."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from .config_utils import load_task_lists


def _load_default_runtime():
    from dotenv import load_dotenv

    load_dotenv()
    from src.agents import DataAnalyzer, DataCollector, ReportGenerator
    from src.config import Config
    from src.memory import Memory
    from src.utils import RunContext, get_logger, make_run_id, set_run_context, setup_logger

    return {
        "Config": Config,
        "Memory": Memory,
        "agent_classes": {
            "collector": DataCollector,
            "analyzer": DataAnalyzer,
            "generator": ReportGenerator,
        },
        "RunContext": RunContext,
        "get_logger": get_logger,
        "make_run_id": make_run_id,
        "set_run_context": set_run_context,
        "setup_logger": setup_logger,
    }


def _resolve_model_name(config: Any, requested: Optional[str], env_name: str) -> Optional[str]:
    if requested:
        return requested
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    llm_dict = getattr(config, "llm_dict", {}) or {}
    if llm_dict:
        return next(iter(llm_dict.keys()))
    return None


def _merge_unique(base_items, generated_items):
    merged = list(base_items or [])
    for item in generated_items or []:
        if item not in merged:
            merged.append(item)
    return merged


async def run_report_pipeline(
    *,
    config_file_path: str = "my_config.yaml",
    config_overrides: Optional[Dict[str, Any]] = None,
    tasks_file: Optional[str] = None,
    resume: bool = False,
    max_concurrent: Optional[int] = None,
    use_llm_name: Optional[str] = None,
    use_vlm_name: Optional[str] = None,
    use_embedding_name: Optional[str] = None,
    auto_generate_tasks: bool = True,
    collect_max_iterations: int = 20,
    analysis_max_iterations: int = 20,
    report_max_iterations: int = 20,
    echo: bool = True,
    config_cls: Any = None,
    memory_cls: Any = None,
    agent_classes: Optional[Dict[str, Any]] = None,
    runtime_helpers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the collect -> analyze -> report pipeline."""
    runtime = runtime_helpers or {}
    if config_cls is None or memory_cls is None or agent_classes is None:
        runtime.update(_load_default_runtime())
        config_cls = config_cls or runtime["Config"]
        memory_cls = memory_cls or runtime["Memory"]
        agent_classes = agent_classes or runtime["agent_classes"]

    config = config_cls(config_file_path=config_file_path, config_dict=config_overrides or {})
    memory = memory_cls(config=config)

    setup_logger = runtime.get("setup_logger")
    get_logger = runtime.get("get_logger")
    logger = logging.getLogger("QAssistant.report")
    if setup_logger:
        logger = setup_logger(log_dir=os.path.join(config.working_dir, "logs"), log_level=logging.INFO)
    if get_logger:
        try:
            get_logger().set_agent_context("runner", "main")
        except Exception:
            pass

    if max_concurrent is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT", "0")) or None
    if max_concurrent:
        logger.info(f"Concurrency limit: {max_concurrent} tasks")
    else:
        logger.info("No concurrency limit (unlimited)")

    if resume and hasattr(memory, "load"):
        loaded_state = memory.load()
        if loaded_state is not None:
            logger.info("Memory state loaded")
        else:
            logger.info("No persisted memory state found; starting fresh")

    run_id = (
        (getattr(memory, "metadata", {}) or {}).get("run_id") if resume else None
    ) or os.getenv("RUN_ID") or getattr(config, "run_id", None)
    if not run_id:
        make_run_id = runtime.get("make_run_id")
        run_id = make_run_id() if make_run_id else "run_cli"
    try:
        setattr(config, "run_id", run_id)
        config.config["run_id"] = run_id
        memory.run_id = run_id
        memory.metadata["run_id"] = run_id
    except Exception:
        pass

    RunContext = runtime.get("RunContext")
    set_run_context = runtime.get("set_run_context")
    if RunContext and set_run_context:
        set_run_context(RunContext(
            run_id=run_id,
            agent_id="runner",
            agent_name="main",
            task_id="run_report",
            step_id=0,
            phase="pipeline",
        ))
    logger.info(f"Run id: {run_id}")

    if tasks_file:
        collect_tasks, analysis_tasks = load_task_lists(tasks_file)
    else:
        collect_tasks = list(config.config.get("custom_collect_tasks", []) or [])
        analysis_tasks = list(config.config.get("custom_analysis_tasks", []) or [])

    use_llm_name = _resolve_model_name(config, use_llm_name, "DS_MODEL_NAME")
    use_vlm_name = _resolve_model_name(config, use_vlm_name, "VLM_MODEL_NAME")
    use_embedding_name = _resolve_model_name(config, use_embedding_name, "EMBEDDING_MODEL_NAME")

    if auto_generate_tasks:
        research_query = (
            f"Research target: {config.config.get('target_name', '')} "
            f"(ticker: {config.config.get('stock_code', '')}), "
            f"target type: {config.config.get('target_type', 'company')}"
        )
        if not getattr(memory, "generated_collect_tasks", []):
            logger.info("Generating collect tasks using LLM...")
            generated_collect_tasks = await memory.generate_collect_tasks(
                query=research_query,
                use_llm_name=use_llm_name,
                max_num=5,
                existing_tasks=collect_tasks,
            )
            logger.info(f"Generated {len(generated_collect_tasks)} collect tasks")
        else:
            generated_collect_tasks = memory.generated_collect_tasks
            logger.info(f"Using {len(generated_collect_tasks)} previously generated collect tasks")

        if not getattr(memory, "generated_analysis_tasks", []):
            logger.info("Generating analysis tasks using LLM...")
            generated_analysis_tasks = await memory.generate_analyze_tasks(
                query=research_query,
                use_llm_name=use_llm_name,
                max_num=5,
                existing_tasks=analysis_tasks,
            )
            logger.info(f"Generated {len(generated_analysis_tasks)} analysis tasks")
        else:
            generated_analysis_tasks = memory.generated_analysis_tasks
            logger.info(f"Using {len(generated_analysis_tasks)} previously generated analysis tasks")

        collect_tasks = _merge_unique(collect_tasks, generated_collect_tasks)
        analysis_tasks = _merge_unique(analysis_tasks, generated_analysis_tasks)

    logger.info(f"Total collect tasks: {len(collect_tasks)}")
    logger.info(f"Total analysis tasks: {len(analysis_tasks)}")

    collector_cls = agent_classes["collector"]
    analyzer_cls = agent_classes["analyzer"]
    generator_cls = agent_classes["generator"]
    research_target = (
        f"Research target: {config.config.get('target_name', '')} "
        f"(ticker: {config.config.get('stock_code', '')})"
    )

    tasks_to_run = []
    for task in collect_tasks:
        tasks_to_run.append({
            "agent_class": collector_cls,
            "task_input": {
                "input_data": {"task": f"{research_target}, task: {task}"},
                "echo": echo,
                "max_iterations": collect_max_iterations,
                "resume": resume,
            },
            "agent_kwargs": {"use_llm_name": use_llm_name},
            "priority": 1,
            "task_content": task,
        })

    for task in analysis_tasks:
        tasks_to_run.append({
            "agent_class": analyzer_cls,
            "task_input": {
                "input_data": {
                    "task": research_target,
                    "analysis_task": task,
                },
                "echo": echo,
                "max_iterations": analysis_max_iterations,
                "resume": resume,
            },
            "agent_kwargs": {
                "use_llm_name": use_llm_name,
                "use_vlm_name": use_vlm_name,
                "use_embedding_name": use_embedding_name,
            },
            "priority": 2,
            "task_content": task,
        })

    tasks_to_run.append({
        "agent_class": generator_cls,
        "task_input": {
            "input_data": {
                "task": research_target,
                "task_type": config.config.get("target_type", "company"),
            },
            "echo": echo,
            "max_iterations": report_max_iterations,
            "resume": resume,
        },
        "agent_kwargs": {
            "use_llm_name": use_llm_name,
            "use_embedding_name": use_embedding_name,
        },
        "priority": 3,
        "task_content": "Final Report Generation",
    })

    agents_info = []
    for task_info in tasks_to_run:
        agent = await memory.get_or_create_agent(
            agent_class=task_info["agent_class"],
            task_input=task_info["task_input"],
            resume=resume,
            priority=task_info["priority"],
            **task_info["agent_kwargs"],
        )
        actual_priority = task_info["priority"]
        for saved_task in getattr(memory, "task_mapping", []):
            if saved_task.get("agent_id") == getattr(agent, "id", None):
                actual_priority = saved_task.get("priority", task_info["priority"])
                break
        agents_info.append({
            "agent": agent,
            "task_input": task_info["task_input"],
            "priority": actual_priority,
            "task_content": task_info["task_content"],
        })

    if hasattr(memory, "save"):
        memory.save()

    agents_info.sort(key=lambda item: item["priority"])
    priority_groups = defaultdict(list)
    for agent_info in agents_info:
        priority_groups[agent_info["priority"]].append(agent_info)

    errors = []
    completed = []
    skipped = []
    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]
        logger.info(f"Executing priority {priority} group ({len(group)} task(s))")
        tasks_to_run_now = []
        for agent_info in group:
            agent = agent_info["agent"]
            if resume and hasattr(memory, "is_agent_finished") and memory.is_agent_finished(agent.id):
                logger.info(f"Agent {agent.id} already completed; skip")
                skipped.append(agent.id)
                continue
            tasks_to_run_now.append(agent_info)

        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def run_agent_with_limit(item):
            agent = item["agent"]
            if semaphore:
                async with semaphore:
                    logger.info(f"  Starting agent {agent.id}")
                    return await agent.async_run(**item["task_input"])
            logger.info(f"  Starting agent {agent.id}")
            return await agent.async_run(**item["task_input"])

        async_tasks = [asyncio.create_task(run_agent_with_limit(item)) for item in tasks_to_run_now]
        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for agent_info, result in zip(tasks_to_run_now, results):
                agent = agent_info["agent"]
                if isinstance(result, Exception):
                    tb_str = "".join(traceback.format_exception(type(result), result, result.__traceback__))
                    logger.error(f"  Task failed: Agent {agent.id}, error: {result}\n{tb_str}")
                    errors.append({
                        "agent_id": agent.id,
                        "agent_name": getattr(agent, "AGENT_NAME", ""),
                        "error": str(result),
                    })
                else:
                    logger.info(f"  Task finished: Agent {agent.id}")
                    completed.append(agent.id)
        logger.info(f"Priority {priority} group finished")

    if hasattr(memory, "save"):
        memory.save()
    logger.info("All tasks completed")

    status = "success" if not errors else "error"
    return {
        "status": status,
        "run_id": run_id,
        "working_dir": getattr(config, "working_dir", ""),
        "config_path": config_file_path,
        "collect_task_count": len(collect_tasks),
        "analysis_task_count": len(analysis_tasks),
        "agent_count": len(agents_info),
        "completed_agent_ids": completed,
        "skipped_agent_ids": skipped,
        "errors": errors,
    }
