#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file verification.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 11-18-2025
@version 1.0
@license Copyright (c) 2025
@desc None
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from scipy.spatial.transform import Rotation


def F_from_poses(T0: np.ndarray, T1: np.ndarray, K: np.ndarray) -> np.ndarray:
    assert T0.shape == (4, 4)
    assert T1.shape == (4, 4)
    assert K.shape == (3, 3)
    T_rel = np.linalg.inv(T1) @ T0
    return F_from_relative_pose(T_rel, K)


def F_from_relative_pose(T_rel: np.ndarray, K: np.ndarray) -> np.ndarray:
    assert T_rel.shape == (4, 4)
    assert K.shape == (3, 3)
    t = T_rel[:3, 3]
    R = T_rel[:3, :3]
    tmat = np.array(
        [
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ]
    )
    F = np.linalg.inv(K).T @ tmat @ R @ np.linalg.inv(K)
    return F


def to_homogeneous(pts: np.ndarray):
    assert pts.shape[0] > 0
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


def compute_epipolar_errors(pts0, pts1, F) -> float:
    assert pts0.shape[0] == pts1.shape[0]
    distances = []
    for p0, p1 in zip(pts0, pts1):
        l1 = F @ p0
        # Normalize line (a x + b y + c = 0)
        norm = np.sqrt(l1[0] ** 2 + l1[1] ** 2)
        if norm == 0:
            dist = 0  # Degenerate
        else:
            dist = np.abs(np.dot(l1, p1)) / norm
        distances.append(dist)

    avg_distance = np.mean(distances) if distances else np.inf
    return avg_distance


def compute_pose_error(R_est, t_est, R_gt, t_gt) -> Tuple[float, float]:
    t_err = np.linalg.norm(t_est - t_gt)
    r_err = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_est) - 1) / 2, -1, 1)))
    return (t_err, r_err)


def recover_3d_from_depth_image(
    pts: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    depth_scale: float = 1000.0,
    kernel_size: int = 5,
    method: str = "median",
    z_min: float = 0.1,
    z_max: float = 5.0,
) -> np.ndarray:
    assert pts.shape[0] > 0
    assert pts.shape[1] == 2

    H, W = depth.shape
    pad = kernel_size // 2
    depth_m = depth.astype(np.float32) / depth_scale
    pts3d = np.full((len(pts), 3), -1.0, dtype=np.float32)
    K_inv = np.linalg.inv(K)
    for i, pt in enumerate(pts):
        u, v = int(round(pt[0])), int(round(pt[1]))
        if u < pad or v < pad or u >= W - pad or v >= H - pad:
            continue  # skip border points

        patch = depth_m[v - pad : v + pad + 1, u - pad : u + pad + 1]
        valid = (patch > z_min) & (patch < z_max) & np.isfinite(patch)

        if np.count_nonzero(valid) < 3:
            continue  # insufficient valid pixels

        values = patch[valid]
        if method == "mean":
            z = np.mean(values)
        else:
            z = np.median(values)

        if z > 0:
            pts3d[i] = (K_inv @ np.array([u, v, 1.0])) * z

    return pts3d


def pose_array_to_mat(arr):
    """_summary_

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(arr) == 7
    mat = np.eye(4)
    mat[:3, 3] = arr[:3]
    mat[:3, :3] = Rotation.from_quat(arr[3:]).as_matrix()
    return mat


def ransac_pnp(
    obj_pts,
    img_pts,
    K,
    distCoeffs=None,
    ratio=0.8,
    reproj_thresh_px=3.0,
    max_iters=1000,
    confidence=0.999,
    min_required=6,  # >4 tends to be more stable with noise
):
    """
    Returns:
      {
        'success': bool,
        'rvec': (3,1) or None,
        'tvec': (3,1) or None,
        'inlier_count': int,
        'inlier_ratio': float,
        'num_used': int,          # candidates given to PnPRansac
        'inlier_idx': np.array,   # indices into the candidate list
      }
    """
    if distCoeffs is None:
        distCoeffs = np.zeros((4, 1), dtype=np.float64)

    if len(obj_pts) < min_required:
        return {
            "success": False,
            "rvec": None,
            "tvec": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "inlier_idx": np.array([], dtype=np.int32),
        }

    # 3) RANSAC PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K.astype(np.float64),
        distCoeffs=distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,  # EPnP also fine: SOLVEPNP_EPNP
        reprojectionError=float(reproj_thresh_px),
        iterationsCount=int(max_iters),
        confidence=float(confidence),
    )

    if not success or inliers is None or len(inliers) == 0:
        return {
            "success": False,
            "rvec": None,
            "tvec": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "inlier_idx": np.array([], dtype=np.int32),
        }

    inliers = inliers.reshape(-1)
    inlier_count = int(len(inliers))
    inlier_ratio = inlier_count / float(len(obj_pts))

    return {
        "success": True,
        "rvec": rvec,  # Rodrigues vector
        "tvec": tvec,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "inlier_idx": inliers,
    }
