"""Local PDF parsing tool for report-reproduction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolResult


class LocalPDFReportParser(Tool):
    """Extract page text and simple tables from a local text-based PDF."""

    def __init__(self):
        super().__init__(
            name="Local PDF report parser",
            description=(
                "Parse a local text-based PDF report into page-level text and simple "
                "table rows. Scanned PDFs and OCR are not supported in this MVP."
            ),
            parameters=[
                {"name": "pdf_path", "type": "str", "description": "Local PDF path", "required": True},
                {"name": "max_pages", "type": "int", "description": "Optional maximum pages to parse", "required": False},
            ],
        )
        self.type = "tool_document"

    def parse_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict[str, Any]:
        path = Path(pdf_path).expanduser()
        warnings: List[str] = []
        pages: List[Dict[str, Any]] = []

        if not path.exists():
            return {
                "source_pdf": str(path),
                "page_count": 0,
                "parsed_page_count": 0,
                "pages": [],
                "warnings": [f"PDF file not found: {path}"],
            }
        if not path.is_file():
            return {
                "source_pdf": str(path),
                "page_count": 0,
                "parsed_page_count": 0,
                "pages": [],
                "warnings": [f"PDF path is not a file: {path}"],
            }

        try:
            import pdfplumber  # type: ignore
        except Exception as exc:
            return {
                "source_pdf": str(path),
                "page_count": 0,
                "parsed_page_count": 0,
                "pages": [],
                "warnings": [f"pdfplumber is unavailable; cannot parse PDF: {exc}"],
            }

        try:
            page_limit = int(max_pages) if max_pages is not None else None
            if page_limit is not None and page_limit <= 0:
                page_limit = None
        except Exception:
            page_limit = None
            warnings.append(f"Invalid max_pages value ignored: {max_pages}")

        try:
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(getattr(pdf, "pages", []) or [])
                selected_pages = (getattr(pdf, "pages", []) or [])[:page_limit]
                for idx, page in enumerate(selected_pages, start=1):
                    text = ""
                    try:
                        text = page.extract_text() or ""
                    except Exception as exc:
                        warnings.append(f"Page {idx}: text extraction failed: {exc}")

                    table_items: List[Dict[str, Any]] = []
                    try:
                        raw_tables = page.extract_tables() or []
                    except Exception as exc:
                        raw_tables = []
                        warnings.append(f"Page {idx}: table extraction failed: {exc}")

                    for table_idx, raw_table in enumerate(raw_tables, start=1):
                        rows = []
                        for row in raw_table or []:
                            rows.append(["" if cell is None else str(cell) for cell in row])
                        table_items.append({"table_index": table_idx, "rows": rows})

                    if not text.strip() and not table_items:
                        warnings.append(
                            f"Page {idx}: no extractable text or table found; scanned PDFs/OCR are not supported."
                        )

                    pages.append({
                        "page_no": idx,
                        "text": text,
                        "tables": table_items,
                    })
        except Exception as exc:
            return {
                "source_pdf": str(path),
                "page_count": 0,
                "parsed_page_count": 0,
                "pages": pages,
                "warnings": warnings + [f"Failed to open or parse PDF: {exc}"],
            }

        if not any((page.get("text") or "").strip() for page in pages):
            warnings.append("No extractable body text found. This may be a scanned PDF; OCR is not supported in this MVP.")

        return {
            "source_pdf": str(path),
            "page_count": page_count,
            "parsed_page_count": len(pages),
            "pages": pages,
            "warnings": warnings,
        }

    def to_markdown(self, parsed_report: Dict[str, Any], max_table_rows: int = 20) -> str:
        lines = [
            "# Parsed Research Report",
            "",
            f"Source PDF: {parsed_report.get('source_pdf', '')}",
            f"Page count: {parsed_report.get('page_count', 0)}",
            f"Parsed pages: {parsed_report.get('parsed_page_count', 0)}",
            "",
        ]

        warnings = parsed_report.get("warnings") or []
        if warnings:
            lines.extend(["## Warnings", ""])
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

        for page in parsed_report.get("pages", []) or []:
            page_no = page.get("page_no", "")
            lines.extend([f"## Page {page_no}", ""])
            text = (page.get("text") or "").strip()
            lines.append(text if text else "_No extractable text on this page._")
            lines.append("")

            tables = page.get("tables") or []
            if tables:
                lines.append("### Tables")
                lines.append("")
                for table in tables:
                    lines.append(f"Table {table.get('table_index', '')}:")
                    for row in (table.get("rows") or [])[:max_table_rows]:
                        safe_row = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
                        lines.append("| " + " | ".join(safe_row) + " |")
                    lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    async def api_function(self, pdf_path: str, max_pages: Optional[int] = None):
        parsed = self.parse_pdf(pdf_path=pdf_path, max_pages=max_pages)
        description = (
            f"Parsed {parsed.get('parsed_page_count', 0)} of "
            f"{parsed.get('page_count', 0)} page(s)."
        )
        if parsed.get("warnings"):
            description += " Warnings: " + "; ".join(parsed["warnings"][:3])
        return [
            ToolResult(
                name=f"Parsed PDF report: {Path(pdf_path).name}",
                description=description,
                data=parsed,
                source=f"Local PDF: {parsed.get('source_pdf', pdf_path)}",
            )
        ]
