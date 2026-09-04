#!/usr/bin/env python3
"""Render the README pipeline diagram as paired light and dark SVGs.

The README embeds these through a <picture> element, which is the only
theme-switching mechanism GitHub supports: it strips <style> and <script>
from markdown, so a single self-theming SVG cannot be relied on.

Both variants come out of one layout pass so they can never drift apart.
Detail belongs in the prose underneath; this diagram carries the shape of the
system only, and the branch at the gate is the thing it exists to show.
"""

import html
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "docs" / "diagrams"

WIDTH = 1120
HEIGHT = 500

CARD_W = 320
CARD_H = 120
ROW_A_Y = 50
ROW_B_Y = 230
COL_X = (40, 400, 760)

CHIP_W = 180
CHIP_H = 58
CHIP_Y = 404

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# GitHub Primer tokens, so the diagram sits in the page rather than on top of it.
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "border": "#d0d7de",
        "text": "#1f2328", "muted": "#656d76", "accent": "#0969da",
        "accept": "#1a7f37", "accept_bg": "#dafbe1",
        "abstain": "#9a6700", "abstain_bg": "#fff8c5",
        "gate_bg": "#ddf4ff",
    },
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
        "text": "#e6edf3", "muted": "#8b949e", "accent": "#4493f8",
        "accept": "#3fb950", "accept_bg": "#0f2d17",
        "abstain": "#d29922", "abstain_bg": "#2d2410",
        "gate_bg": "#0c2d478c",
    },
}

# (index, title, detail). Kept to one short line each; the numbered list in the
# README carries the real specifics.
STAGES: List[Tuple[str, str, str]] = [
    ("1", "Dual-view capture", "Front and side RGB, plus height"),
    ("2", "Segmentation", "YOLOv11m detector, SAM 2.1 masks"),
    ("3", "Siamese encoders", "Twin ResNet-18, InfoNCE aligned"),
    ("4", "Girth regression", "Waist, hip and chest in cm"),
    ("5", "Clinical indices", "WHtR, WHR and BRI, derived"),
]

GATE = ("6", "SMPL-X geometry gate", "Fits a 3D body, renders it back")


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def card(x: int, y: int, num: str, title: str, detail: str, t: Dict[str, str],
         emphasis: bool) -> str:
    fill = t["gate_bg"] if emphasis else t["panel"]
    stroke = t["accent"] if emphasis else t["border"]
    stroke_w = 2 if emphasis else 1
    return f"""
  <g>
    <rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="12"
          fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>
    <text x="{x + 24}" y="{y + 38}" font-family="{MONO}" font-size="15"
          fill="{t['accent']}" font-weight="600">{esc(num)}</text>
    <text x="{x + 50}" y="{y + 38}" font-family="{SANS}" font-size="20"
          fill="{t['text']}" font-weight="600">{esc(title)}</text>
    <text x="{x + 24}" y="{y + 74}" font-family="{SANS}" font-size="16"
          fill="{t['muted']}">{esc(detail)}</text>
  </g>"""


def chip(x: int, label: str, detail: str, fill: str, stroke: str,
         text_col: str, t: Dict[str, str], css_class: str) -> str:
    return f"""
  <g class="{css_class}">
    <rect x="{x}" y="{CHIP_Y}" width="{CHIP_W}" height="{CHIP_H}" rx="10"
          fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>
    <text x="{x + CHIP_W // 2}" y="{CHIP_Y + 25}" text-anchor="middle"
          font-family="{SANS}" font-size="17" font-weight="600"
          fill="{text_col}">{esc(label)}</text>
    <text x="{x + CHIP_W // 2}" y="{CHIP_Y + 45}" text-anchor="middle"
          font-family="{SANS}" font-size="14"
          fill="{t['muted']}">{esc(detail)}</text>
  </g>"""


def arrow(x1: int, y1: int, x2: int, y2: int, t: Dict[str, str]) -> str:
    return (f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{t["border"]}" stroke-width="2" marker-end="url(#tip)"/>')


# Motion exists to show one thing: the same pipeline runs twice, and the gate
# sends the result somewhere different each time. Pass one is accepted and
# reaches Report; pass two is rejected and reaches Abstain. Nothing else moves.
# CSS animation rather than SMIL so prefers-reduced-motion can switch it off.
PACKET_SECONDS = 9
CYCLE_SECONDS = PACKET_SECONDS * 2


def animation_css() -> str:
    return f"""  <style>
    .packet      {{ animation: flow {PACKET_SECONDS}s linear infinite; }}
    .out-accept  {{ animation: to-report {CYCLE_SECONDS}s linear infinite; }}
    .out-abstain {{ animation: to-abstain {CYCLE_SECONDS}s linear infinite; }}
    .chip-accept {{ animation: lit-report {CYCLE_SECONDS}s linear infinite; }}
    .chip-abstain{{ animation: lit-abstain {CYCLE_SECONDS}s linear infinite; }}

    /* One trip down the pipeline, ending at the gate. Runs twice per cycle. */
    @keyframes flow {{
      0%   {{ transform: translate(200px, 150px); opacity: 0; }}
      2%   {{ opacity: 1; }}
      13%  {{ transform: translate(560px, 150px); }}
      24%  {{ transform: translate(920px, 150px); }}
      30%  {{ transform: translate(920px, 200px); }}
      40%  {{ transform: translate(200px, 200px); }}
      46%  {{ transform: translate(200px, 330px); }}
      57%  {{ transform: translate(560px, 330px); }}
      68%  {{ transform: translate(920px, 330px); opacity: 1; }}
      73%  {{ transform: translate(920px, 330px); opacity: 0; }}
      100% {{ transform: translate(920px, 330px); opacity: 0; }}
    }}

    /* Pass one: the gate accepts, so the result reaches Report. */
    @keyframes to-report {{
      0%, 33%   {{ transform: translate(920px, 350px); opacity: 0; }}
      34%       {{ transform: translate(920px, 350px); opacity: 1; }}
      40%       {{ transform: translate(750px, 370px); opacity: 1; }}
      44%       {{ transform: translate(750px, 396px); opacity: 1; }}
      47%, 100% {{ transform: translate(750px, 396px); opacity: 0; }}
    }}

    /* Pass two: the gate rejects, so the result reaches Abstain instead. */
    @keyframes to-abstain {{
      0%, 83%   {{ transform: translate(920px, 350px); opacity: 0; }}
      84%       {{ transform: translate(920px, 350px); opacity: 1; }}
      90%       {{ transform: translate(970px, 370px); opacity: 1; }}
      94%       {{ transform: translate(970px, 396px); opacity: 1; }}
      97%, 100% {{ transform: translate(970px, 396px); opacity: 0; }}
    }}

    @keyframes lit-report {{
      0%, 43%   {{ opacity: 0.4; }}
      45%, 56%  {{ opacity: 1; }}
      58%, 100% {{ opacity: 0.4; }}
    }}
    @keyframes lit-abstain {{
      0%, 4%    {{ opacity: 1; }}
      6%, 93%   {{ opacity: 0.4; }}
      95%, 100% {{ opacity: 1; }}
    }}

    @media (prefers-reduced-motion: reduce) {{
      .packet, .out-accept, .out-abstain {{ animation: none; opacity: 0; }}
      .chip-accept, .chip-abstain {{ animation: none; opacity: 1; }}
    }}
  </style>
"""


def build(theme_name: str, animated: bool) -> str:
    t = THEMES[theme_name]
    parts: List[str] = [
        # Intrinsic width/height give the image a real aspect ratio to scale
        # inside GitHub's column. height="auto" is not valid SVG.
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="Six pipeline stages feed an SMPL-X geometry gate. The gate '
        f'either reports the measurements or abstains and asks for a recapture.">',
        '  <defs>',
        f'    <marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">',
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{t["border"]}"/>',
        '    </marker>',
        '  </defs>',
    ]
    if animated:
        parts.append(animation_css())
    parts += [
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{t["bg"]}"/>',
    ]

    # Row A: stages 1 to 3.
    for i in range(3):
        num, title, detail = STAGES[i]
        parts.append(card(COL_X[i], ROW_A_Y, num, title, detail, t, emphasis=False))
    for i in range(2):
        parts.append(arrow(COL_X[i] + CARD_W, ROW_A_Y + CARD_H // 2,
                           COL_X[i + 1] - 8, ROW_A_Y + CARD_H // 2, t))

    # Wrap from the end of row A back to the start of row B, drawn as an elbow so
    # it reads the way a line of text wraps.
    wrap_y = (ROW_A_Y + CARD_H + ROW_B_Y) // 2
    end_x = COL_X[0] + CARD_W // 2
    start_x = COL_X[2] + CARD_W // 2
    parts.append(
        f'  <path d="M {start_x} {ROW_A_Y + CARD_H} '
        f'L {start_x} {wrap_y - 14} '
        f'Q {start_x} {wrap_y} {start_x - 14} {wrap_y} '
        f'L {end_x + 14} {wrap_y} '
        f'Q {end_x} {wrap_y} {end_x} {wrap_y + 14} '
        f'L {end_x} {ROW_B_Y - 8}" '
        f'fill="none" stroke="{t["border"]}" stroke-width="2" marker-end="url(#tip)"/>')

    # Row B: stages 4, 5, then the gate.
    for i in range(3, 5):
        num, title, detail = STAGES[i]
        parts.append(card(COL_X[i - 3], ROW_B_Y, num, title, detail, t, emphasis=False))
    parts.append(card(COL_X[2], ROW_B_Y, GATE[0], GATE[1], GATE[2], t, emphasis=True))
    for i in range(2):
        parts.append(arrow(COL_X[i] + CARD_W, ROW_B_Y + CARD_H // 2,
                           COL_X[i + 1] - 8, ROW_B_Y + CARD_H // 2, t))

    # The branch. This is the point of the diagram.
    gate_cx = COL_X[2] + CARD_W // 2
    accept_x = 660
    abstain_x = 880
    for target_x, colour in ((accept_x + CHIP_W // 2, t["accept"]),
                             (abstain_x + CHIP_W // 2, t["abstain"])):
        parts.append(
            f'  <path d="M {gate_cx} {ROW_B_Y + CARD_H} L {gate_cx} {ROW_B_Y + CARD_H + 20} '
            f'L {target_x} {ROW_B_Y + CARD_H + 20} L {target_x} {CHIP_Y - 9}" '
            f'fill="none" stroke="{colour}" stroke-width="2" marker-end="url(#tip)"/>')

    parts.append(chip(accept_x, "Report", "render-back agrees",
                      t["accept_bg"], t["accept"], t["accept"], t,
                      "chip-accept" if animated else ""))
    parts.append(chip(abstain_x, "Abstain", "recapture requested",
                      t["abstain_bg"], t["abstain"], t["abstain"], t,
                      "chip-abstain" if animated else ""))

    # The moving parts. Drawn last so they sit above the cards they travel over.
    # The static variant omits them entirely rather than freezing them mid-path.
    if animated:
        parts.append(
            f'  <circle class="packet" r="7" fill="{t["accent"]}" opacity="0"/>')
        parts.append(
            f'  <circle class="out-accept" r="7" fill="{t["accept"]}" opacity="0"/>')
        parts.append(
            f'  <circle class="out-abstain" r="7" fill="{t["abstain"]}" opacity="0"/>')

    parts.append(
        f'  <text x="40" y="{CHIP_Y + 36}" font-family="{SANS}" font-size="16" '
        f'fill="{t["muted"]}">No number is reported unless the fitted 3D body</text>')
    parts.append(
        f'  <text x="40" y="{CHIP_Y + 58}" font-family="{SANS}" font-size="16" '
        f'fill="{t["muted"]}">explains the silhouettes it came from.</text>')

    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        for animated, suffix in ((True, ""), (False, "_static")):
            path = OUT_DIR / f"pipeline_{theme}{suffix}.svg"
            path.write_text(build(theme, animated), encoding="utf-8")
            print(f"{path.relative_to(BASE_DIR)}  ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
