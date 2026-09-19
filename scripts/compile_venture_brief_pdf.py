#!/usr/bin/env python3
"""Render the BodyFit venture brief HTML as a print-ready PDF."""

from pathlib import Path
import shutil
import subprocess


MAX_PDF_BYTES: int = 100 * 1024 * 1024
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
HTML_PATH: Path = ROOT_DIR / "docs" / "venture" / "venture_brief.html"
PDF_PATH: Path = ROOT_DIR / "docs" / "venture" / "bodyfit_venture_brief.pdf"


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    if not html_path.is_file():
        raise FileNotFoundError(f"Venture brief HTML does not exist: {html_path}")

    npx_path: str | None = shutil.which("npx")
    if npx_path is None:
        raise FileNotFoundError("npx is required to run the installed Playwright renderer")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            npx_path,
            "--yes",
            "playwright",
            "pdf",
            "--channel",
            "chrome",
            "--paper-format",
            "Letter",
            "--wait-for-selector",
            "body",
            html_path.as_uri(),
            str(pdf_path),
        ],
        check=True,
    )


def verify_pdf(pdf_path: Path, max_pdf_bytes: int) -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF was not created: {pdf_path}")

    pdf_size: int = pdf_path.stat().st_size
    if pdf_size == 0:
        raise ValueError(f"PDF is empty: {pdf_path}")
    if pdf_size >= max_pdf_bytes:
        raise ValueError(
            f"PDF is {pdf_size} bytes, which exceeds the {max_pdf_bytes}-byte limit: {pdf_path}"
        )
    if pdf_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"Output does not have a PDF header: {pdf_path}")
    return pdf_size


def main() -> None:
    render_pdf(HTML_PATH, PDF_PATH)
    pdf_size: int = verify_pdf(PDF_PATH, MAX_PDF_BYTES)
    print(f"Created {PDF_PATH} ({pdf_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
