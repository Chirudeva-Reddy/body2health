#!/usr/bin/env python3
"""Record the README walkthrough of the live Body2Fit web application using Playwright and ffmpeg.

Selectors here are asserted, not guarded. A missing element raises instead of
silently skipping the step, so a UI rename fails the recording rather than
producing a video that quietly omits half the pipeline.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "docs" / "assets"
RECORDINGS_DIR = BASE_DIR / "docs" / "recordings_tmp"

SERVER_PORT = 8088
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800

# README renders the GIF at 760px; encoding at that width avoids paying for
# pixels the reader never sees.
GIF_WIDTH = 760
GIF_FPS = 12

FFMPEG = shutil.which("ffmpeg")


def show_section(page: Page, target_y: int, dwell_ms: int) -> None:
    """Jump to target_y and hold still.

    Deliberately a snap rather than an eased scroll. Every frame of a moving
    scroll is unique, which is what a GIF pays for; a held frame costs almost
    nothing. Snapping keeps the walkthrough legible and the asset small.
    """
    page.evaluate("y => window.scrollTo(0, y)", target_y)
    page.wait_for_timeout(dwell_ms)


def orbit_mesh(page: Page) -> None:
    """Drag the studio Three.js canvas so the recovered mesh rotates on camera."""
    # The pipeline tab has its own #three-canvas-container; the studio tab, which
    # is the one on screen here, renders into #three-studio-container.
    canvas = page.wait_for_selector("#three-studio-container canvas", timeout=15000)
    box = canvas.bounding_box()
    if box is None:
        raise RuntimeError("3D canvas has no bounding box; the mesh viewer did not lay out")
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for dx in range(-90, 100, 14):
        page.mouse.move(cx + dx, cy + 12, steps=2)
        page.wait_for_timeout(30)
    page.mouse.up()


def drive_demo(page: Page) -> None:
    """Walk the demo through inference, the clinical readout, the gate, and the 3D mesh."""
    page.goto(f"http://localhost:{SERVER_PORT}/", wait_until="networkidle")
    page.wait_for_timeout(700)

    # Run a real forward pass and wait for the gate verdict to render.
    page.get_by_role("button", name="1-Click Test").click()
    page.wait_for_function(
        "() => { const e = document.getElementById('gate-iou'); return e && e.textContent.trim() !== '--'; }",
        timeout=30000,
    )
    page.wait_for_timeout(900)

    # Segmentation and anatomical decomposition.
    show_section(page, 470, 1700)

    # Recovered girths, clinical indices, WHO risk categories.
    show_section(page, 980, 2200)

    # SMPL-X render-back gate panel: IoU, chamfer, accept/reject badge.
    show_section(page, 1305, 2400)

    # 3D studio.
    show_section(page, 0, 400)
    page.click("#tab-btn-mesh3d", timeout=15000)
    page.wait_for_timeout(1200)

    # The studio auto-rotates by default, which makes every captured frame
    # unique. Stop it so the orbit below is the only motion here.
    page.locator("#tab-content-mesh3d button", has_text="Auto Rotate").click(timeout=15000)
    page.wait_for_timeout(600)
    orbit_mesh(page)
    page.wait_for_timeout(900)

    # The studio's wireframe control carries no id, so scope by tab and label.
    wireframe = page.locator("#tab-content-mesh3d button", has_text="Wireframe")
    wireframe.click(timeout=15000)
    page.wait_for_timeout(1300)
    wireframe.click(timeout=15000)
    page.wait_for_timeout(700)

    page.click("#tab-btn-pipeline", timeout=15000)
    page.wait_for_timeout(800)


def encode_outputs(source_video: Path) -> None:
    """Write the MP4 and a palette-optimised GIF from the raw capture."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode the walkthrough")

    mp4_out = ASSETS_DIR / "bodyfit_demo_walkthrough.mp4"
    gif_out = ASSETS_DIR / "bodyfit_pipeline_walkthrough.gif"
    palette = RECORDINGS_DIR / "palette.png"

    subprocess.run(
        [FFMPEG, "-y", "-i", str(source_video),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
         "-movflags", "+faststart", str(mp4_out)],
        check=True, capture_output=True,
    )

    # stats_mode=diff weights the palette toward the pixels that actually change,
    # which is what a mostly-static dark dashboard needs.
    scale = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos"
    subprocess.run(
        [FFMPEG, "-y", "-i", str(source_video),
         "-vf", f"{scale},palettegen=stats_mode=diff:max_colors=128", str(palette)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [FFMPEG, "-y", "-i", str(source_video), "-i", str(palette),
         "-lavfi", f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
         "-loop", "0", str(gif_out)],
        check=True, capture_output=True,
    )

    print(f"MP4: {mp4_out} ({mp4_out.stat().st_size / 1e6:.2f} MB)")
    print(f"GIF: {gif_out} ({gif_out.stat().st_size / 1e6:.2f} MB)")


def record_live_demo() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    server = subprocess.Popen(
        [".venv/bin/python3", "web/server.py", "--port", str(SERVER_PORT)],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3.0)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                record_video_dir=str(RECORDINGS_DIR),
                record_video_size={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            )
            page = context.new_page()
            drive_demo(page)
            context.close()
            browser.close()
    finally:
        server.terminate()
        server.wait()

    videos = list(RECORDINGS_DIR.glob("*.webm"))
    if not videos:
        raise RuntimeError(f"Playwright wrote no video into {RECORDINGS_DIR}")

    encode_outputs(max(videos, key=os.path.getctime))
    shutil.rmtree(RECORDINGS_DIR, ignore_errors=True)


if __name__ == "__main__":
    record_live_demo()
