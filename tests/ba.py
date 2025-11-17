#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file ba.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 11-19-2025
@version 1.0
@license Copyright (c) 2025
@desc None
"""


import gtsam
import numpy as np
from typing import Tuple


def two_view_ba(pts0, pts1, initial_lmks, K, initial_pose1):
    """
    Perform two-view BA using GTSAM.

    Args:
        pts0/pts1: Nx2 inlier points (image 0 and 1).
        initial_lmks: Nx3 initial 3D landmarks from triangulation/depth_image initialization.
        K: 3x3 intrinsics (fx, fy, cx, cy; assume skew=0).
        initial_pose1: gtsam.Pose3 initial relative pose for camera 1 (from PnP).

    Returns:
        optimized_poses: Dict of Pose3 for cameras 0 and 1.
        optimized_lmks: Nx3 array of optimized landmarks.
    """
    # Calibration (assume same for both)
    calibration = gtsam.Cal3_S2(fx=K[0, 0], fy=K[1, 1], s=0, u0=K[0, 2], v0=K[1, 2])

    # Noise models
    pose_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)  # Small noise for prior (6 DoF)
    projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # 1 pixel reprojection noise
    # Create a robust noise model using the Huber kernel
    k_parameter = 1.345  # default
    robust_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(k_parameter), projection_noise
    )

    # Symbols
    pose0_sym = gtsam.symbol("x", 0)
    pose1_sym = gtsam.symbol("x", 1)

    # Factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Prior on pose 0 (fixed as identity)
    graph.add(gtsam.PriorFactorPose3(pose0_sym, gtsam.Pose3(), pose_prior_noise))
    depth_sigma = 1e-6

    # Add projection factors for each landmark
    num_lmks = len(initial_lmks)
    for i in range(num_lmks):
        lmk_sym = gtsam.symbol("l", i)

        # Observation in camera 0
        meas0 = gtsam.Point2(pts0[i, 0], pts0[i, 1])
        graph.add(gtsam.GenericProjectionFactorCal3_S2(meas0, robust_noise, pose0_sym, lmk_sym, calibration))

        # Observation in camera 1
        meas1 = gtsam.Point2(pts1[i, 0], pts1[i, 1])
        graph.add(gtsam.GenericProjectionFactorCal3_S2(meas1, robust_noise, pose1_sym, lmk_sym, calibration))

        depth_noise = gtsam.noiseModel.Isotropic.Sigma(3, depth_sigma)  # Tune sigma based on sensor noise
        # measured_depth = depth_img[pts1[i, 1], pts1[i, 0]]  # Example from image 1
        # Custom factor or approximate with point prior along ray
        graph.add(gtsam.PriorFactorPoint3(lmk_sym, initial_lmks[i], depth_noise))  # Simple prior

    # Initial estimates
    initial = gtsam.Values()
    initial.insert(pose0_sym, gtsam.Pose3())  # Identity for camera 0
    initial.insert(pose1_sym, gtsam.Pose3(initial_pose1))
    for i in range(num_lmks):
        initial.insert(gtsam.symbol("l", i), gtsam.Point3(initial_lmks[i]))

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # Extract results
    optimized_poses = {0: result.atPose3(pose0_sym).matrix(), 1: result.atPose3(pose1_sym).matrix()}
    optimized_lmks = np.array([result.atPoint3(gtsam.symbol("l", i)) for i in range(num_lmks)])

    pre_error = graph.error(initial)
    post_error = graph.error(result)
    print(f"Error reduction: {pre_error - post_error}")

    return True, optimized_poses, optimized_lmks
