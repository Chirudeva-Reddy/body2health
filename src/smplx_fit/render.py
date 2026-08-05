"""Projection and silhouette rendering utilities for SMPL-X meshes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import cv2
import numpy as np
import pyrender
import trimesh


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ProjectionParams:
    scale_px_per_m: float
    x_scale: float
    center_x_px: float
    center_y_px: float
    yaw_deg: float
    roll_deg: float


def initial_projection_params(vertices_m: np.ndarray, target_bbox: BBox) -> ProjectionParams:
    _validate_vertices(vertices_m)
    x0, y0, x1, y1 = target_bbox
    bbox_height = max(1, y1 - y0 + 1)
    mesh_height = float(np.max(vertices_m[:, 1]) - np.min(vertices_m[:, 1]))
    if mesh_height <= 0.0:
        raise ValueError("SMPL-X mesh has non-positive height")
    return ProjectionParams(
        scale_px_per_m=float(bbox_height) / mesh_height,
        x_scale=1.0,
        center_x_px=float(x0 + x1) / 2.0,
        center_y_px=float(y0 + y1) / 2.0,
        yaw_deg=0.0,
        roll_deg=0.0,
    )


def project_vertices_to_pixels(
    vertices_m: np.ndarray,
    params: ProjectionParams,
) -> Tuple[np.ndarray, np.ndarray]:
    _validate_vertices(vertices_m)
    _validate_projection_params(params)
    centered = vertices_m - _mesh_bbox_center(vertices_m)
    rotated = _rotate_vertices(centered, params.yaw_deg, params.roll_deg)
    x_pixels = params.center_x_px + (rotated[:, 0] * params.scale_px_per_m * params.x_scale)
    y_pixels = params.center_y_px - (rotated[:, 1] * params.scale_px_per_m)
    depth = rotated[:, 2] * params.scale_px_per_m
    points = np.stack([x_pixels, y_pixels], axis=1).astype(np.float64)
    return points, depth.astype(np.float64)


def render_silhouette(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    output_shape: Tuple[int, int],
    params: ProjectionParams,
    backend: str,
) -> np.ndarray:
    _validate_faces(faces)
    if backend == "pyrender":
        return _render_pyrender(vertices_m, faces, output_shape, params)
    if backend == "opencv":
        return _render_opencv_union(vertices_m, faces, output_shape, params)
    raise ValueError(f"unsupported_render_backend: {backend}")


def _render_pyrender(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    output_shape: Tuple[int, int],
    params: ProjectionParams,
) -> np.ndarray:
    height, width = int(output_shape[0]), int(output_shape[1])
    points, depth = project_vertices_to_pixels(vertices_m, params)
    world_vertices = np.zeros((points.shape[0], 3), dtype=np.float64)
    world_vertices[:, 0] = points[:, 0] - (float(width) / 2.0)
    world_vertices[:, 1] = (float(height) / 2.0) - points[:, 1]
    world_vertices[:, 2] = depth
    mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera_z = max(1000.0, float(np.max(np.abs(world_vertices[:, 2]))) + 1000.0)
    camera = pyrender.OrthographicCamera(
        xmag=float(width) / 2.0,
        ymag=float(height) / 2.0,
        znear=0.01,
        zfar=camera_z + 2000.0,
    )
    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[2, 3] = camera_z
    scene.add(camera, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        _, depth_buffer = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()
    mask = (depth_buffer > 0.0).astype(np.uint8) * 255
    return _largest_component(mask)


def _render_opencv_union(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    output_shape: Tuple[int, int],
    params: ProjectionParams,
) -> np.ndarray:
    points, _ = project_vertices_to_pixels(vertices_m, params)
    int_points = np.rint(points).astype(np.int32)
    mask = np.zeros(output_shape, dtype=np.uint8)
    for face in faces:
        polygon = int_points[np.asarray(face, dtype=np.int64)]
        cv2.fillConvexPoly(mask, polygon, 255)
    return _largest_component(mask)


def _mesh_bbox_center(vertices_m: np.ndarray) -> np.ndarray:
    mins = np.min(vertices_m, axis=0)
    maxs = np.max(vertices_m, axis=0)
    return ((mins + maxs) / 2.0).astype(np.float64)


def _rotate_vertices(centered_vertices: np.ndarray, yaw_deg: float, roll_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    roll = np.deg2rad(roll_deg)
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    cos_roll = float(np.cos(roll))
    sin_roll = float(np.sin(roll))
    yaw_x = (centered_vertices[:, 0] * cos_yaw) + (centered_vertices[:, 2] * sin_yaw)
    yaw_y = centered_vertices[:, 1]
    yaw_z = (-centered_vertices[:, 0] * sin_yaw) + (centered_vertices[:, 2] * cos_yaw)
    roll_x = (yaw_x * cos_roll) - (yaw_y * sin_roll)
    roll_y = (yaw_x * sin_roll) + (yaw_y * cos_roll)
    return np.stack([roll_x, roll_y, yaw_z], axis=1).astype(np.float64)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        raise ValueError("rendered SMPL-X silhouette has no foreground component")
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cast(np.ndarray, ((labels == largest_index).astype(np.uint8) * 255))


def _validate_vertices(vertices_m: np.ndarray) -> None:
    if vertices_m.size == 0:
        raise ValueError("vertices_m is empty")
    if vertices_m.ndim != 2 or vertices_m.shape[1] != 3:
        raise ValueError(f"vertices_m must be (N, 3), got {vertices_m.shape!r}")
    if not bool(np.isfinite(vertices_m).all()):
        raise ValueError("vertices_m contains non-finite values")


def _validate_faces(faces: np.ndarray) -> None:
    if faces.size == 0:
        raise ValueError("faces is empty")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape!r}")


def _validate_projection_params(params: ProjectionParams) -> None:
    values = (
        params.scale_px_per_m,
        params.x_scale,
        params.center_x_px,
        params.center_y_px,
        params.yaw_deg,
        params.roll_deg,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"projection params contain non-finite values: {params!r}")
    if params.x_scale < 0.75 or params.x_scale > 1.35:
        raise ValueError(f"projection x_scale outside supported range: {params.x_scale:.3f}")
    if abs(params.yaw_deg) > 45.0:
        raise ValueError(f"projection yaw outside supported range: {params.yaw_deg:.3f}")
    if abs(params.roll_deg) > 20.0:
        raise ValueError(f"projection roll outside supported range: {params.roll_deg:.3f}")
