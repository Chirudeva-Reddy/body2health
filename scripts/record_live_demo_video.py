#!/usr/bin/env python3
"""Record a buttery-smooth video & GIF walkthrough of the live BodyFit web application using Playwright & ffmpeg."""

import os
from pathlib import Path
import subprocess
import time
import shutil
from playwright.sync_api import sync_playwright

BASE_DIR = Path("/Users/tacticalcamel/Desktop/Projects/bodyfit")
ASSETS_DIR = BASE_DIR / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR = BASE_DIR / "docs" / "recordings_tmp"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


def record_live_demo():
    # 1. Start web/server.py in a background process
    print("Starting background server on port 8088...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    server_proc = subprocess.Popen(
        [".venv/bin/python3", "web/server.py", "--port", "8088"],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server ready
    time.sleep(2.5)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                device_scale_factor=2,
                record_video_dir=str(RECORDINGS_DIR),
                record_video_size={"width": 1280, "height": 800},
            )

            page = context.new_page()
            print("Navigating to live application...")
            page.goto("http://localhost:8088/", wait_until="networkidle")
            page.wait_for_timeout(1000)

            # 1. Click 1-Click Test
            print("Clicking 1-Click Test...")
            quick_btn = page.query_selector("#btn-quick-run")
            if quick_btn:
                quick_btn.click()
            page.wait_for_timeout(2500)

            # 2. Scroll smoothly down the pipeline
            page.mouse.wheel(0, 300)
            page.wait_for_timeout(1200)
            page.mouse.wheel(0, -300)
            page.wait_for_timeout(800)

            # 3. Switch to 3D Viewer tab
            print("Switching to 3D Viewer tab...")
            tab_3d = page.query_selector('[data-tab="tab-mesh"]')
            if tab_3d:
                tab_3d.click()
            page.wait_for_timeout(2000)

            # 4. Smoothly orbit/drag the 3D mesh
            print("Rotating 3D SMPL-X mesh...")
            canvas = page.query_selector("#three-canvas-container canvas")
            if canvas:
                box = canvas.bounding_box()
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    page.mouse.move(cx, cy)
                    page.mouse.down()
                    for dx in range(-120, 140, 15):
                        page.mouse.move(cx + dx, cy + 20, steps=3)
                        page.wait_for_timeout(40)
                    page.mouse.up()
            page.wait_for_timeout(1000)

            # Toggle wireframe
            wire_btn = page.query_selector("#btn-mesh-wireframe")
            if wire_btn:
                wire_btn.click()
                page.wait_for_timeout(1200)
                wire_btn.click()
                page.wait_for_timeout(800)

            # 5. Switch to Architecture & Novelty tab
            print("Switching to Architecture & Novelty tab...")
            arch_tab = page.query_selector('[data-tab="tab-architecture"]')
            if arch_tab:
                arch_tab.click()
            page.wait_for_timeout(1500)
            page.mouse.wheel(0, 250)
            page.wait_for_timeout(1500)

            # 6. Switch back to Pipeline
            pipe_tab = page.query_selector('[data-tab="tab-pipeline"]')
            if pipe_tab:
                pipe_tab.click()
            page.wait_for_timeout(1200)

            context.close()
            browser.close()

    finally:
        server_proc.terminate()
        server_proc.wait()

    # Find recorded video file
    video_files = list(RECORDINGS_DIR.glob("*.webm"))
    if not video_files:
        print("ERROR: No video recorded!")
        return

    latest_video = max(video_files, key=os.path.getctime)
    mp4_out = ASSETS_DIR / "bodyfit_demo_walkthrough.mp4"
    gif_out = ASSETS_DIR / "bodyfit_pipeline_walkthrough.gif"

    print(f"Converting {latest_video} to smooth MP4: {mp4_out}...")
    # Convert webm to mp4
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(latest_video),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast",
        str(mp4_out)
    ], check=True)

    print(f"Converting {latest_video} to smooth 25fps GIF with palettegen: {gif_out}...")
    # High-quality two-pass GIF conversion with palettegen
    palette_path = RECORDINGS_DIR / "palette.png"
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(latest_video),
        "-vf", "fps=20,scale=900:-1:flags=lanczos,palettegen",
        str(palette_path)
    ], check=True)

    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(latest_video), "-i", str(palette_path),
        "-lavfi", "fps=20,scale=900:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
        str(gif_out)
    ], check=True)

    # Cleanup temp recordings
    shutil.rmtree(RECORDINGS_DIR, ignore_errors=True)
    print("ALL DONE! Generated smooth MP4 and GIF.")


if __name__ == "__main__":
    record_live_demo()
