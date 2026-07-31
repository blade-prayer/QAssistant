from pathlib import Path
import sys


root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.append(root)


from src.utils.prompt_loader import get_prompt_loader  # noqa: E402


def test_deep_search_prompt_protocol_and_citation_format():
    loader = get_prompt_loader("search_agent", report_type="general")

    prompt = loader.get_prompt(
        "deep_search",
        basic_task="Research target: Example Corp",
        question="What are the latest revenue drivers?",
        current_time="2026-07-31 12:00:00",
        max_iterations=5,
        target_language="English",
    )

    assert "<search>" in prompt
    assert "<click>" in prompt
    assert "<report>" in prompt
    assert "[Source: exact source title]" in prompt
    assert "[N]" not in prompt
    assert "Company filings" in prompt
    assert "Browse 2-3" in prompt
    assert "fewer than 5" not in prompt


def test_data_collector_prompt_tool_priority_and_save_contract():
    loader = get_prompt_loader("data_collector", report_type="general")

    prompt = loader.get_prompt(
        "data_collect",
        api_descriptions="Tool: Deep Search\nTool: Balance sheet",
        current_time="2026-07-31 12:00:00",
        task="Collect revenue and management background",
        target_language="English",
        research_target="Example Corp (ticker: 000001)",
    )

    assert "call_tool(...)" in prompt
    assert "save_result" in prompt
    assert "structured financial, market, macro, and industry API tools first" in prompt
    assert "Use Deep Search for non-structured evidence" in prompt
    assert "do not save raw HTML dumps" in prompt


def test_report_reproduction_prompts_define_brief_and_code_contracts():
    loader = get_prompt_loader("report_reproduction", report_type="general")

    brief_prompt = loader.get_prompt(
        "strategy_brief",
        source_pdf="sample.pdf",
        parsed_report_markdown="## Page 1\nUse momentum factor and monthly rebalance.",
        target_language="Chinese",
    )
    code_prompt = loader.get_prompt(
        "strategy_code",
        source_pdf="sample.pdf",
        strategy_brief_json="{}",
        parsed_report_excerpt="## Page 1\nUse momentum factor.",
        target_language="Chinese",
    )

    assert "confirmed_from_pdf" in brief_prompt
    assert "inferred_from_context" in brief_prompt
    assert "implementation_assumptions" in brief_prompt
    assert "missing_information" in brief_prompt
    assert "page_evidence" in brief_prompt

    assert "sample_strategy_py" in code_prompt
    assert "readme_strategy_md" in code_prompt
    assert "PARAMS" in code_prompt
    assert "load_data_placeholder()" in code_prompt
    assert "validate_input_schema(df)" in code_prompt
    assert "compute_factors(df, params)" in code_prompt
    assert "generate_signals(df, params)" in code_prompt
    assert "build_target_weights(signals, params)" in code_prompt
    assert "run_strategy_example(df, params)" in code_prompt
    assert "must not fetch data" in code_prompt
