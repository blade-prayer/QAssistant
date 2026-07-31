import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.agents import ReportReproductionAgent  # noqa: E402
from src.config import Config  # noqa: E402
from src.memory import Memory  # noqa: E402
from src.utils import setup_logger  # noqa: E402


def _resolve_model_name(config: Config, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    env_model = os.getenv("DS_MODEL_NAME")
    if env_model:
        return env_model
    if config.llm_dict:
        return next(iter(config.llm_dict.keys()))
    raise ValueError("No LLM model configured. Set DS_MODEL_NAME or pass --model-name.")


async def run_report_reproduction(args):
    config_overrides = {}
    if args.output_dir:
        config_overrides["output_dir"] = args.output_dir

    config = Config(config_file_path="my_config.yaml", config_dict=config_overrides)
    model_name = _resolve_model_name(config, args.model_name)
    memory = Memory(config=config)

    log_dir = os.path.join(config.working_dir, "logs")
    setup_logger(log_dir=log_dir, log_level=logging.INFO)

    agent = ReportReproductionAgent(
        config=config,
        memory=memory,
        use_llm_name=model_name,
        enable_code=False,
    )
    result = await agent.async_run(
        input_data={
            "pdf_path": args.pdf_path,
            "report_id": args.report_id,
            "max_pages": args.max_pages,
        },
        resume=False,
    )
    print(json.dumps({
        "report_id": result["report_id"],
        "output_dir": result["output_dir"],
        "manifest": result["final_result"],
        "warnings": result["manifest"].get("warnings", []),
    }, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate sample strategy code from a local PDF report.")
    parser.add_argument("--pdf-path", required=True, help="Local PDF report path.")
    parser.add_argument("--output-dir", default=None, help="Base output directory. Uses my_config.yaml by default.")
    parser.add_argument("--report-id", default=None, help="Optional stable report id for the output folder.")
    parser.add_argument("--model-name", default=None, help="LLM model key from my_config.yaml. Defaults to DS_MODEL_NAME.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional maximum pages to parse.")
    args = parser.parse_args()
    asyncio.run(run_report_reproduction(args))


if __name__ == "__main__":
    main()
