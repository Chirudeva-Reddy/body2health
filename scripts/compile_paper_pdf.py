#!/usr/bin/env python3
"""Compile xplan docx research paper into a publication-grade PDF using Playwright."""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright


def extract_docx_paragraphs(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read("word/document.xml")
        tree = ET.fromstring(xml_content)
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in tree.iterfind(".//w:p", namespaces):
            texts = [node.text for node in p.iterfind(".//w:t", namespaces) if node.text]
            if texts:
                paragraphs.append("".join(texts).strip())
    return [p for p in paragraphs if p]


def build_academic_html(paragraphs: list[str]) -> str:
    title = paragraphs[0] if paragraphs else "Research Direction for SMPL-Gated Phone Anthropometry"
    
    body_html_parts = []
    current_section = None
    
    for p in paragraphs[1:]:
        # Detect headers
        if len(p) < 60 and not p.endswith(".") and not p.startswith("["):
            body_html_parts.append(f'<h2 class="section-title">{p}</h2>')
        elif p.startswith("“") or p.startswith('"') or "Table" in p[:15]:
            body_html_parts.append(f'<div class="callout"><p>{p}</p></div>')
        else:
            body_html_parts.append(f'<p>{p}</p>')

    content_html = "\n".join(body_html_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  @page {{
    size: letter;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {{
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 9pt;
    }}
  }}
  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #111827;
    margin: 0;
    padding: 0;
  }}
  .header {{
    text-align: center;
    margin-bottom: 24px;
    border-bottom: 1.5px solid #1e3a8a;
    padding-bottom: 16px;
  }}
  h1 {{
    font-size: 18pt;
    font-weight: bold;
    color: #0f172a;
    margin: 0 0 10px 0;
    line-height: 1.25;
  }}
  .authors {{
    font-size: 11pt;
    font-style: italic;
    color: #334155;
    margin-bottom: 6px;
  }}
  .affil {{
    font-size: 9pt;
    color: #64748b;
    margin-bottom: 12px;
  }}
  .abstract-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    padding: 12px 16px;
    margin: 16px 0 24px 0;
    font-size: 9.5pt;
  }}
  .abstract-title {{
    font-weight: bold;
    text-transform: uppercase;
    font-size: 9pt;
    letter-spacing: 0.5px;
    color: #1e40af;
    margin-bottom: 4px;
  }}
  .two-column {{
    column-count: 2;
    column-gap: 20px;
    text-align: justify;
  }}
  h2.section-title {{
    font-size: 11pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #1e3a8a;
    margin-top: 16px;
    margin-bottom: 6px;
    border-bottom: 0.5px solid #cbd5e1;
    padding-bottom: 2px;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 10px 0;
    text-indent: 12pt;
  }}
  .callout {{
    background: #f1f5f9;
    border-left: 3px solid #64748b;
    padding: 8px 12px;
    margin: 10px 0;
    font-size: 9pt;
    break-inside: avoid;
  }}
  .callout p {{
    text-indent: 0;
    margin: 0;
  }}
  .diagram-figure {{
    margin: 16px 0;
    text-align: center;
    break-inside: avoid;
    column-span: all;
  }}
  .diagram-figure img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
  }}
  .caption {{
    font-size: 8.5pt;
    color: #475569;
    margin-top: 6px;
    font-style: italic;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="authors">Chirudeva Reddy</div>
  <div class="affil">BodyFit Computer Vision & Healthcare AI Research &bull; Dubai, UAE</div>
</div>

<div class="abstract-box">
  <div class="abstract-title">Abstract</div>
  Mobile 2D anthropometry promises accessible health screening but historically suffers from two critical failure modes: conflating lean muscle with visceral fat (the fundamental limitation of BMI), and acting unconditionally confident when presented with clothing or posture artifacts. We introduce <strong>BodyFit</strong>, a dual-view silhouette anthropometry architecture featuring a 3D geometry reliability gate. By supervising on true tape-measured girths (waist, hip, chest) rather than synthetic body-fat labels, and deriving clinical biomarkers (WHtR, WHR, BRI) through physiological arithmetic, BodyFit provides interpretable cardiometabolic screening. Crucially, BodyFit couples deep contrastive estimation with an SMPL-X render-back gate: a personalized 3D mesh is re-projected into the camera plane, triggering autonomous abstention whenever render IoU falls below 0.55. This abstention mechanism eliminates silent corruption, ensuring safe deployment in tele-health applications.
</div>

<div class="diagram-figure">
  <img src="{Path('docs/diagrams/pipeline_architecture.png').resolve().as_uri()}" alt="BodyFit Pipeline Architecture">
  <div class="caption">Figure 1: Full end-to-end architecture of BodyFit showing dual-view smartphone preprocessing, Siamese ResNet-18 contrastive encoders, dimension regression heads, derived clinical indices, and the SMPL-X render-back reliability gate.</div>
</div>

<div class="two-column">
  {content_html}
</div>

</body>
</html>
"""
    return html


def compile_pdf():
    base = Path("/Users/tacticalcamel/Desktop/Projects/bodyfit")
    docx_path = (base / "xplan/Research Direction for SMPL-Gated Phone Anthropometry.docx").resolve()
    out_dir = (base / "docs/paper").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = out_dir / "paper.html"
    pdf_path = out_dir / "BodyFit_Research_Paper.pdf"

    paragraphs = extract_docx_paragraphs(docx_path)
    html_content = build_academic_html(paragraphs)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML at {html_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html_path.as_uri())
        page.pdf(
            path=str(pdf_path),
            format="Letter",
            margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
            print_background=True,
        )
        browser.close()
    print(f"Successfully compiled academic paper PDF: {pdf_path}")


if __name__ == "__main__":
    compile_pdf()
