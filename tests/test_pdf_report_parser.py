import asyncio
import sys
import types
from pathlib import Path


root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.append(root)


from src.tools.document.pdf_report_parser import LocalPDFReportParser  # noqa: E402


class FakePage:
    def __init__(self, text="", tables=None):
        self._text = text
        self._tables = tables or []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _install_fake_pdfplumber(monkeypatch, pages):
    fake_pdfplumber = types.ModuleType("pdfplumber")
    fake_pdfplumber.open = lambda path: FakePDF(pages)
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)


def test_pdf_parser_extracts_page_text_and_tables(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    _install_fake_pdfplumber(
        monkeypatch,
        [
            FakePage(
                text="Page one strategy text.",
                tables=[[["field", "meaning"], ["momentum_20d", "20 day return"]]],
            ),
            FakePage(text="Page two should be skipped by max_pages."),
        ],
    )

    parser = LocalPDFReportParser()
    parsed = parser.parse_pdf(str(pdf_path), max_pages=1)
    markdown = parser.to_markdown(parsed)

    assert parsed["source_pdf"].endswith("sample.pdf")
    assert parsed["page_count"] == 2
    assert parsed["parsed_page_count"] == 1
    assert parsed["pages"][0]["text"] == "Page one strategy text."
    assert parsed["pages"][0]["tables"][0]["rows"][1] == ["momentum_20d", "20 day return"]
    assert "## Page 1" in markdown
    assert "| momentum_20d | 20 day return |" in markdown


def test_pdf_parser_missing_path_returns_clear_warning(tmp_path):
    parser = LocalPDFReportParser()
    parsed = parser.parse_pdf(str(tmp_path / "missing.pdf"))

    assert parsed["parsed_page_count"] == 0
    assert "PDF file not found" in parsed["warnings"][0]


def test_pdf_parser_empty_text_warns_without_exception(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    _install_fake_pdfplumber(monkeypatch, [FakePage(text="", tables=[])])

    parser = LocalPDFReportParser()
    parsed = parser.parse_pdf(str(pdf_path))

    assert parsed["parsed_page_count"] == 1
    assert any("OCR is not supported" in warning for warning in parsed["warnings"])


def test_pdf_parser_api_function_returns_tool_result(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    _install_fake_pdfplumber(monkeypatch, [FakePage(text="Strategy rule text.")])

    parser = LocalPDFReportParser()
    result = asyncio.run(parser.api_function(str(pdf_path)))[0]

    assert result.name == "Parsed PDF report: sample.pdf"
    assert result.data["pages"][0]["text"] == "Strategy rule text."
    assert "Local PDF:" in result.source
