#!/usr/bin/env python3
"""Convert pipeline_architecture.excalidraw JSON to a standalone SVG file for GitHub README."""

import json
from pathlib import Path
import html


def excalidraw_to_svg(json_path: Path, svg_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements = [e for e in data.get("elements", []) if not e.get("isDeleted")]
    if not elements:
        return

    # Compute bounding box
    min_x = min(e.get("x", 0) for e in elements)
    min_y = min(e.get("y", 0) for e in elements)
    max_x = max(e.get("x", 0) + e.get("width", 0) for e in elements)
    max_y = max(e.get("y", 0) + e.get("height", 0) for e in elements)

    pad = 40
    width = int(max_x - min_x + pad * 2)
    height = int(max_y - min_y + pad * 2)
    shift_x = -min_x + pad
    shift_y = -min_y + pad

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="auto" style="background:#ffffff; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">',
        '<defs>',
        '  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#1e3a5f"/>',
        '  </marker>',
        '  <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#047857"/>',
        '  </marker>',
        '  <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>',
        '  </marker>',
        '  <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#6d28d9"/>',
        '  </marker>',
        '  <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#b45309"/>',
        '  </marker>',
        '  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.05"/>',
        '  </filter>',
        '</defs>',
    ]

    # First pass: draw rectangles, ellipses, diamonds, lines, arrows
    for e in elements:
        etype = e.get("type")
        x = e.get("x", 0) + shift_x
        y = e.get("y", 0) + shift_y
        w = e.get("width", 0)
        h = e.get("height", 0)
        bg = e.get("backgroundColor", "transparent")
        stroke = e.get("strokeColor", "#1e3a5f")
        sw = e.get("strokeWidth", 2)
        style = e.get("strokeStyle", "solid")
        dash = ' stroke-dasharray="6,4"' if style == "dashed" else ""

        if etype == "rectangle":
            rx = 8 if e.get("roundness") else 0
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{bg}" stroke="{stroke}" stroke-width="{sw}"{dash} filter="url(#shadow)"/>'
            )
        elif etype == "ellipse":
            cx = x + w / 2
            cy = y + h / 2
            svg_parts.append(
                f'<ellipse cx="{cx}" cy="{cy}" rx="{w/2}" ry="{h/2}" fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
        elif etype == "diamond":
            p1 = f"{x + w/2},{y}"
            p2 = f"{x + w},{y + h/2}"
            p3 = f"{x + w/2},{y + h}"
            p4 = f"{x},{y + h/2}"
            svg_parts.append(
                f'<polygon points="{p1} {p2} {p3} {p4}" fill="{bg}" stroke="{stroke}" stroke-width="{sw}" filter="url(#shadow)"/>'
            )
        elif etype == "arrow":
            points = e.get("points", [])
            if len(points) >= 2:
                path_d = f"M {x + points[0][0]} {y + points[0][1]}"
                for pt in points[1:]:
                    path_d += f" L {x + pt[0]} {y + pt[1]}"
                
                marker = "arrow"
                if stroke == "#047857":
                    marker = "arrow-green"
                elif stroke == "#dc2626":
                    marker = "arrow-red"
                elif stroke == "#6d28d9":
                    marker = "arrow-purple"
                elif stroke == "#b45309":
                    marker = "arrow-amber"

                svg_parts.append(
                    f'<path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{marker})"/>'
                )

    # Second pass: draw text elements on top
    for e in elements:
        if e.get("type") == "text":
            x = e.get("x", 0) + shift_x
            y = e.get("y", 0) + shift_y
            w = e.get("width", 0)
            text = e.get("text", "")
            size = e.get("fontSize", 14)
            color = e.get("strokeColor", "#1e3a5f")
            align = e.get("textAlign", "left")

            anchor = "start"
            tx = x
            if align == "center":
                anchor = "middle"
                tx = x + w / 2
            elif align == "right":
                anchor = "end"
                tx = x + w

            lines = text.split("\n")
            line_height = size * 1.3
            svg_parts.append(f'<text fill="{color}" font-size="{size}px" font-weight="500" text-anchor="{anchor}">')
            for i, line in enumerate(lines):
                escaped = html.escape(line)
                svg_parts.append(f'  <tspan x="{tx}" y="{y + (i+1)*line_height - 2}">{escaped}</tspan>')
            svg_parts.append('</text>')

    svg_parts.append('</svg>')

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"Successfully generated {svg_path} ({width}x{height})")


if __name__ == "__main__":
    src = Path("docs/diagrams/pipeline_architecture.excalidraw")
    dst = Path("docs/diagrams/pipeline_architecture.svg")
    excalidraw_to_svg(src, dst)
