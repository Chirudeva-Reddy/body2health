#!/usr/bin/env python3
"""Render SVG to PNG using Playwright."""

from pathlib import Path
from playwright.sync_api import sync_playwright


def render_svg():
    base_dir = Path("/Users/tacticalcamel/Desktop/Projects/bodyfit")
    svg_path = (base_dir / "docs/diagrams/pipeline_architecture.svg").resolve()
    png_path = (base_dir / "docs/diagrams/pipeline_architecture.png").resolve()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(device_scale_factor=2)
        page.goto(svg_path.as_uri())
        svg_el = page.query_selector("svg")
        if svg_el:
            svg_el.screenshot(path=str(png_path))
            print(f"Successfully rendered {png_path}")
        else:
            page.screenshot(path=str(png_path), full_page=True)
            print(f"Successfully rendered full page {png_path}")
        browser.close()


if __name__ == "__main__":
    render_svg()
