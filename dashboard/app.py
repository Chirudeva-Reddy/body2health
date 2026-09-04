from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CKPT = "checkpoints/best_640x480_v4_resnet.pt"
DEFAULT_NLF_MODEL_PATH = "models/nlf/nlf_l_multi.torchscript"
DEFAULT_OUTPUT_ROOT = "outputs/dashboard_runs"
DEFAULT_SMPLX_DIR = "models/smplx"
CAPTURE_MODE_DUAL = "Front + side photos"
CAPTURE_MODE_SINGLE_FRONT = "Single front photo"


class PipelineRunError(RuntimeError):
    def __init__(self, command: List[str], completed: subprocess.CompletedProcess[str]) -> None:
        self.command = command
        self.completed = completed
        message = (
            "pipeline command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}"
        )
        super().__init__(message)


def main() -> None:
    st.set_page_config(page_title="Body2Fit Demo", layout="wide")
    st.title("Body2Fit")
    st.caption(
        "Phone-image anthropometry for waist, hip, and chest, with central-adiposity "
        "and cardiometabolic screening interpretation. This is not a diagnosis."
    )

    with st.sidebar:
        st.header("Inputs")
        capture_mode = st.radio("Capture mode", (CAPTURE_MODE_DUAL, CAPTURE_MODE_SINGLE_FRONT))
        front_file = st.file_uploader("Front RGB image", type=("png", "jpg", "jpeg"))
        side_file = None
        if capture_mode == CAPTURE_MODE_DUAL:
            side_file = st.file_uploader("Side RGB image", type=("png", "jpg", "jpeg"))
        height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=240.0, value=175.0, step=0.5)
        sex = st.selectbox("Sex for risk thresholds", ("male", "female"))
        ckpt_path = st.text_input("Checkpoint", DEFAULT_CKPT)
        device = st.selectbox("Device", _available_devices())
        run_smplx = st.checkbox("Run SMPL-X reliability gate", value=True)
        smplx_model_path = st.text_input("SMPL-X model path", _default_smplx_model_path(str(sex)))
        nlf_model_path = st.text_input("NLF model path", DEFAULT_NLF_MODEL_PATH)
        run_button = st.button("Run pipeline", type="primary")

    if not run_button:
        _show_existing_demo()
        return
    if front_file is None:
        st.error("Upload a front RGB image.")
        return
    if capture_mode == CAPTURE_MODE_DUAL and side_file is None:
        st.error("Upload both front and side RGB images for dual-view mode.")
        return

    output_dir = _new_output_dir(DEFAULT_OUTPUT_ROOT)
    front_path = output_dir / _safe_upload_name("front", front_file.name)
    _write_upload(front_file.getvalue(), front_path)
    side_path = None
    if side_file is not None:
        side_path = output_dir / _safe_upload_name("side", side_file.name)
        _write_upload(side_file.getvalue(), side_path)

    command = _build_command(
        front_path=front_path,
        side_path=side_path,
        capture_mode=str(capture_mode),
        ckpt_path=str(ckpt_path),
        height_cm=float(height_cm),
        sex=str(sex),
        device=str(device),
        run_smplx=bool(run_smplx),
        smplx_model_path=str(smplx_model_path),
        nlf_model_path=str(nlf_model_path),
        output_dir=output_dir,
    )
    st.code(" ".join(command), language="bash")

    try:
        with st.spinner("Running segmentation, dimension prediction, and reliability checks..."):
            completed = _run_command(command)
    except PipelineRunError as exc:
        _render_pipeline_error(exc)
        return

    result_path = output_dir / "result.json"
    if not result_path.exists():
        st.error(f"Pipeline finished but did not write result JSON: {result_path}")
        st.text(completed.stdout[-4000:])
        st.text(completed.stderr[-4000:])
        return
    payload = json.loads(result_path.read_text())
    payload["_result_json_path"] = str(result_path)
    _render_dashboard(payload)


def _build_command(
    front_path: Path,
    side_path: Path | None,
    capture_mode: str,
    ckpt_path: str,
    height_cm: float,
    sex: str,
    device: str,
    run_smplx: bool,
    smplx_model_path: str,
    nlf_model_path: str,
    output_dir: Path,
) -> List[str]:
    command = [
        sys.executable,
        "4-infer/1infer.py",
        "--front_rgb",
        str(front_path),
        "--ckpt",
        ckpt_path,
        "--height_cm",
        f"{height_cm:.2f}",
        "--sex",
        sex,
        "--device",
        device,
        "--save_silhouettes",
        str(output_dir),
        "--json",
        str(output_dir / "result.json"),
    ]
    if capture_mode == CAPTURE_MODE_DUAL:
        if side_path is None:
            raise ValueError("dual-view mode requires side_path")
        command.extend(["--side_rgb", str(side_path)])
    elif capture_mode == CAPTURE_MODE_SINGLE_FRONT:
        command.append("--single_front")
    else:
        raise ValueError(f"unsupported capture mode: {capture_mode}")
    if run_smplx:
        command.extend(
            [
                "--smplx_fit",
                "--save_smplx",
                str(output_dir / "smplx"),
                "--smplx_model_path",
                smplx_model_path,
                "--nlf_model_path",
                nlf_model_path,
            ]
        )
    return command


def _run_command(command: List[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PipelineRunError(command, completed)
    return completed


def _render_pipeline_error(error: PipelineRunError) -> None:
    st.error("Pipeline failed. The message below is from the strict inference command.")
    st.write("Exit code:", error.completed.returncode)
    st.code(" ".join(error.command), language="bash")
    if error.completed.stdout:
        st.subheader("stdout")
        st.text(error.completed.stdout[-4000:])
    if error.completed.stderr:
        st.subheader("stderr")
        st.text(error.completed.stderr[-4000:])


def _render_dashboard(payload: Dict[str, Any]) -> None:
    if payload.get("estimate_mode") == "single_front_view":
        st.warning(
            "Single-front-photo mode duplicates the front silhouette for the side branch. "
            "Use it for quick demos and SMPL-X visualization; dual-view capture remains preferred for reporting."
        )

    st.subheader("Central-Adiposity and Cardiometabolic Screening Summary")
    summary = _health_summary_from_payload(payload)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Estimated screening risk", str(summary["overall_risk"]).replace("_", " "))
    col_b.metric("Primary driver", str(summary["primary_driver"]))
    col_c.metric("Reportable", "Yes" if bool(summary["reportable"]) else "No")
    st.info(str(summary["message"]))
    st.caption(str(summary["disclaimer"]))

    st.subheader("Predicted Measurements")
    measurement_cols = st.columns(3)
    for index, (name, value) in enumerate(payload["measurements"].items()):
        measurement_cols[index % 3].metric(name, f"{float(value):.2f} cm")

    st.subheader("Derived Indices")
    index_cols = st.columns(3)
    for index, (name, value) in enumerate(payload["indices"].items()):
        index_cols[index % 3].metric(name, f"{float(value):.4f}")

    st.subheader("Risk Components")
    st.dataframe(summary["components"], width="stretch")

    st.subheader("Visuals")
    _render_source_images(payload)
    _render_silhouettes(payload)
    _render_silhouette_regions(payload)
    _render_smplx(payload)

    result_path = _result_json_path(payload)
    if result_path is not None:
        _download_file("Download result JSON", str(result_path), "result.json")


def _render_source_images(payload: Dict[str, Any]) -> None:
    source_paths = payload.get("source_image_paths")
    if not isinstance(source_paths, dict) or len(source_paths) == 0:
        return
    columns = st.columns(len(source_paths))
    columns[0].image(str(source_paths["front"]), caption="Front RGB")
    if "side" in source_paths and len(columns) > 1:
        columns[1].image(str(source_paths["side"]), caption="Side RGB")


def _render_silhouettes(payload: Dict[str, Any]) -> None:
    seg_paths = payload["segmentation_paths"]
    left, right = st.columns(2)
    left.image(seg_paths["front"], caption="Front silhouette")
    side_caption = (
        "Side silhouette"
        if payload.get("estimate_mode") != "single_front_view"
        else "Duplicated front silhouette for side branch"
    )
    right.image(seg_paths["side"], caption=side_caption)


def _render_silhouette_regions(payload: Dict[str, Any]) -> None:
    region_artifacts = payload.get("silhouette_region_artifacts")
    if not isinstance(region_artifacts, dict) or len(region_artifacts) == 0:
        return
    st.subheader("Silhouette Research Boundaries")
    columns = st.columns(2)
    front = region_artifacts.get("front")
    side = region_artifacts.get("side")
    if isinstance(front, dict) and "overlay" in front:
        columns[0].image(str(front["overlay"]), caption="Front: shoulders, torso, waist, hip, legs")
    if isinstance(side, dict) and "overlay" in side:
        side_caption = (
            "Side: profile research boundaries"
            if payload.get("estimate_mode") != "single_front_view"
            else "Duplicated front-view boundaries for side branch"
        )
        columns[1].image(str(side["overlay"]), caption=side_caption)
    with st.expander("Boundary metrics"):
        st.json(region_artifacts, expanded=False)


def _render_smplx(payload: Dict[str, Any]) -> None:
    smplx_payload = payload.get("smplx_fit")
    if smplx_payload is None:
        return
    st.subheader("SMPL-X Reliability")
    st.write("Accepted:", bool(smplx_payload["accepted"]))
    st.json(smplx_payload["metrics"], expanded=False)
    visuals = smplx_payload["visuals"]
    view_cols = st.columns(3)
    view_cols[0].image(visuals["rendered_front"], caption="Rendered SMPL-X")
    view_cols[1].image(visuals["front_overlay"], caption="Full overlay")
    view_cols[2].image(visuals["front_shoulder_overlay"], caption="Shoulder overlay")
    _download_file("Download SMPL-X OBJ", visuals["obj"], "smplx_fit.obj")


def _show_existing_demo() -> None:
    demo = Path("outputs/final/deva/result.json")
    if not demo.exists():
        st.write("Upload one front photo or a front/side pair, then run the pipeline.")
        st.caption(
            "Best inputs are full-body RGB photos with one person, visible head-to-feet, "
            "simple background, good lighting, and arms away from the torso."
        )
        return
    st.write("Current saved demo:")
    payload = json.loads(demo.read_text())
    payload["_result_json_path"] = str(demo)
    _render_dashboard(payload)


def _health_summary_from_payload(payload: Dict[str, Any]) -> Dict[str, object]:
    existing = payload.get("health_summary")
    if isinstance(existing, dict):
        return existing
    raise KeyError("payload is missing health_summary")


def _download_file(label: str, path: str, filename: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        st.warning(f"Missing artifact: {file_path}")
        return
    st.download_button(label, file_path.read_bytes(), file_name=filename)


def _result_json_path(payload: Dict[str, Any]) -> Path | None:
    direct_path = payload.get("_result_json_path")
    if isinstance(direct_path, str) and direct_path:
        return Path(direct_path)
    output_dir = payload.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        return Path(output_dir) / "result.json"
    return None


def _available_devices() -> Tuple[str, ...]:
    try:
        import torch
    except ImportError:
        return ("cpu",)
    if torch.cuda.is_available():
        return "cuda", "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", "cpu"
    return ("cpu",)


def _default_smplx_model_path(sex: str) -> str:
    normalized = sex.strip().lower()
    if normalized == "male":
        return f"{DEFAULT_SMPLX_DIR}/SMPLX_MALE.npz"
    if normalized == "female":
        return f"{DEFAULT_SMPLX_DIR}/SMPLX_FEMALE.npz"
    return f"{DEFAULT_SMPLX_DIR}/SMPLX_NEUTRAL.npz"


def _new_output_dir(root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _safe_upload_name(prefix: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"unsupported upload extension for {filename}")
    return f"{prefix}{suffix}"


def _write_upload(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


if __name__ == "__main__":
    main()
