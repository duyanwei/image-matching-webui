#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file run_bench.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 11-18-2025
@version 1.0
@license Copyright (c) 2025
@desc None
"""

from pathlib import Path
import os
import json
import numpy as np
import cv2
import sys

from typing import Optional, Dict
import time

import pandas as pd

from scipy.spatial.transform import Rotation

from ba import two_view_ba


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


from imcui.hloc.utils.geometry import (
    ransac_pnp,
    recover_3d_from_depth_image,
    pose_array_to_mat,
    F_from_relative_pose,
    compute_pose_error,
    compute_epipolar_errors_bench,
    draw_matches,
)


from imcui.hloc import logger
from imcui.ui.utils import DEVICE, get_matcher_zoo, load_config
from imcui.api import ImageMatchingAPI


def load_gt_poses(filepath):
    """_summary_

    Args:
        filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert os.path.exists(filepath), f"{filepath} does NOT exist."
    with open(filepath, "r") as file:
        # ... proceed to load the data
        data = json.load(file)
        query_pose = data["query_pose"]
        train_pose = data["train_pose"]
        return pose_array_to_mat(query_pose), pose_array_to_mat(train_pose)


def find_img_pose(query_timestamp: float, poses: np.ndarray) -> Optional[np.ndarray]:
    assert poses.shape[0] > 0
    assert poses.shape[1] == 8  # timestamp, tx, ty, tz, qx, qy, qz, qw

    diff = np.abs(query_timestamp - poses[:, 0])
    minval = np.min(diff)
    if minval > 5e-2:
        return None
    index = np.argmin(diff)
    return pose_array_to_mat(poses[index, 1:])


def find_img_path(query_timestamp: float, stamped_images: Dict[float, str]) -> str:
    stamps = np.array(list(stamped_images.keys()))
    return ""


def load_images_with_poses(seq_dir: Path, poses: np.ndarray, color_stamp: float):
    image_name = f"{color_stamp:.6f}.png"

    color_img_path = seq_dir / "color" / image_name
    depth_img_path = seq_dir / "depth" / image_name

    status = False
    color_im = None
    depth_im = None
    Twc = find_img_pose(color_stamp, poses)
    if color_img_path.is_file() and depth_img_path.is_file() and Twc is not None:
        status = True
        color_im = cv2.imread(color_img_path)[:, :, ::-1]  # RGB
        depth_im = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    return status, color_im, depth_im, Twc


def load_kfs_with_poses(kf_file: Path, poses: np.ndarray):
    kf_poses = []
    kf_stamps = np.loadtxt(kf_file, ndmin=2, skiprows=1, delimiter=",")[:, 0]
    for stamp in kf_stamps:
        p = find_img_pose(stamp, poses)
        if p is None:
            continue
        kf_poses.append(p)
    return kf_stamps, np.array(kf_poses)


def retrieve_train_images(
    query_timestamp: float,
    query_pose: np.ndarray,
    kfs_stamps: np.ndarray,
    kfs_poses: np.ndarray,
    max_pose_dist: float = 5.0,
):
    train_list = []
    for kf_stamp, kf_pose in zip(kfs_stamps, kfs_poses):
        diff = np.linalg.norm(query_pose[:3, 3] - kf_pose[:3, 3])
        if diff > max_pose_dist:
            continue
        train_list.append((kf_stamp, kf_pose))
    return train_list


def load_matching_set(filepath: Path, only_kf=False):
    df = pd.read_csv(filepath)
    result = df.groupby("frame_stamp")["kf_stamp"].apply(list).to_dict()
    if only_kf:
        # Collect all unique kf_stamps
        all_kf_stamps = set(df["kf_stamp"].unique())
        # Filter the result to only include frame_stamps that are in all_kf_stamps
        result = {k: v for k, v in result.items() if k in all_kf_stamps}
    return result


def main():

    # Load config for detection and matching.
    config = load_config(ROOT / "config/dense.yaml")
    matcher_zoo_restored = get_matcher_zoo(config["matcher_zoo"])

    # Load intrinsic.
    K = np.eye(3)
    K[0][0] = 386.52199190267083
    K[1][1] = 387.32300428823663
    K[0][2] = 326.5103569741365
    K[1][2] = 237.40293732598795

    # TSRB
    # K[0][0] = 378.7210693359375
    # K[0][0] = 324.4961853027344
    # K[0][0] = 378.7210693359375
    # K[0][0] = 239.1063690185547

    D = np.array(
        [
            -0.04604118637879282,
            0.03505887527496214,
            0.0001787943036668921,
            -0.00024723627967045646,
            0.0,
        ]
    )

    seqs = [
        "floor14_3",
        # "floor13_1",
        # "floor3_1",
        # "apartment1_1",
        # "apartment2_1",
        # "apartment3_1",
        # "office_1",
        # "14-13-14",
    ]
    dataname = "cid"
    data_dir = Path("/mnt/DATA/datasets/cid/")
    result_dir = Path("/mnt/DATA/experiments/semantic/signature/")

    enable_pnp = True

    for k, v in matcher_zoo_restored.items():
        enable = config["matcher_zoo"][k].get("enable", True)
        skip_ci = config["matcher_zoo"][k].get("skip_ci", False)
        if not enable or skip_ci:
            print(f"Skipping {k} ...")
            continue
        if not v["dense"]:
            # if "xfeat+lightglue" not in k:
            continue
        print(f"Testing {k} ...")

        # Disable F/H verification since we already have PnP.
        api = ImageMatchingAPI(conf=v, device=DEVICE, max_keypoints=1024)
        if enable_pnp:
            api.get_conf()["ransac"]["enable"] = False

        for seq in seqs:
            seq_dir = data_dir / f"{seq}/{seq}/"
            pose_file = data_dir / f"gt_poses/{seq}_cam0.txt"
            poses = np.loadtxt(pose_file, ndmin=2)

            output_dir = result_dir / dataname / seq
            output_dir.mkdir(exist_ok=True, parents=True)

            # Load set.
            graph_file = result_dir / f"{seq}_pcd_cg_edges.csv"
            if not graph_file.is_file():
                continue
            graph = load_matching_set(graph_file, only_kf=True)
            print(f"{len(graph)} set loaded.")
            # exit(-1)

            # # Load KFS.
            # kf_file = result_dir / f"{seq}_pcd_kfs.csv"
            # if not kf_file.is_file():
            #     continue
            # kfs_stamps, kfs_poses = load_kfs_with_poses(kf_file, poses)
            # Load images.
            # assos = np.loadtxt(seq_dir / "associations.txt", ndmin=2)
            # for img_index, (query_stamp, depth_stamp) in enumerate(assos):

            matching_stats = []
            lmks_change = []
            for query_idx, (query_stamp, train_list) in enumerate(graph.items()):

                print(f"Matching {query_idx+1} out of {len(graph)}")

                status, query_im, query_depth, query_Twc = load_images_with_poses(seq_dir, poses, query_stamp)

                if not status:
                    continue

                # train_list = retrieve_train_images(query_stamp, query_Twc, kfs_stamps, kfs_poses)

                print(f"Found {len(train_list)} train images for matching.")

                for train_idx, train_stamp in enumerate(train_list):
                    # train_im_path = seq_dir / "color" / f"{train_stamp:.6f}.png"
                    train_status, train_im, train_depth, train_Twc = load_images_with_poses(seq_dir, poses, train_stamp)

                    if not train_status:
                        continue
                    # if not train_im_path.is_file():
                    # continue
                    # train_im = cv2.imread(train_im_path)[:, :, ::-1]  # [RGB]

                    # Run matching.
                    pred = api(query_im, train_im)
                    assert pred is not None

                    img0_t = np.inf
                    img1_t = np.inf
                    match_t = np.inf
                    pnp_t = np.inf
                    ba_t = np.inf

                    img0_t = pred["img0_extract_t"]
                    img1_t = pred["img1_extract_t"]
                    match_t = pred["match_t"]

                    # Log and save visual result.
                    if v["dense"]:
                        method_name = str(api.get_conf()["matcher"]["model"]["name"])
                    else:
                        method_name = "{}_{}".format(
                            str(api.get_conf()["feature"]["model"]["name"]),
                            str(api.get_conf()["matcher"]["model"]["name"]),
                        )
                    # log_dir = output_dir / method_name
                    # log_dir.mkdir(exist_ok=True, parents=True)
                    # api.visualize(log_path=log_dir)

                    query_pts = None
                    if "mkeypoints0_orig" in pred.keys():
                        query_pts = pred["mkeypoints0_orig"]
                    train_pts = None
                    if "mkeypoints1_orig" in pred.keys():
                        train_pts = pred["mkeypoints1_orig"]
                        # print(train_pts)

                    if (
                        query_pts is None
                        or train_pts is None
                        or len(query_pts) <= 3
                        or len(train_pts) <= 3
                        or train_pts.shape[1] < 2
                        or query_pts.shape[1] < 2
                    ):
                        continue

                    # Init obj points
                    obj_pts, valid_query_pts, valid_train_pts, valid_indices = recover_3d_from_depth_image(
                        query_pts, query_depth, train_pts, train_depth, K, z_min=0.1, z_max=5.0, kernel_size=5
                    )

                    # Run Pnp Verification.
                    pnp_t0 = time.time()
                    pnp_result = ransac_pnp(
                        obj_pts, valid_train_pts, K, refine=False, max_iters=5, confidence=0.95, reproj_thresh_px=5
                    )
                    pnp_t = time.time() - pnp_t0
                    pnp_status = pnp_result["success"]
                    pnp_ic = pnp_result["inlier_count"]
                    pnp_ir = pnp_result["inlier_ratio"]
                    pnp_indices = pnp_result["inlier_idx"]
                    # print(f"PF: IR = {pnp_ir}, IC = {pnp_ic}")

                    # print(pnp_result)

                    # Run metrics.
                    T_rel = np.linalg.inv(train_Twc) @ query_Twc
                    R_gt = T_rel[:3, :3]
                    t_gt = T_rel[:3, 3]
                    t_diff = np.linalg.norm(t_gt)
                    R_diff = np.linalg.norm(Rotation.from_matrix(R_gt).as_rotvec(degrees=True))
                    # print(f"t_diff = {t_diff:.2f} m , R_diff = {R_diff:.2f} degree")

                    t_err = np.inf
                    R_err = np.inf
                    t_err_ba = np.inf
                    R_err_ba = np.inf
                    T_err = np.inf
                    T_err_ba = np.inf
                    lmk_change = np.inf
                    if pnp_result["success"]:
                        t_est = np.squeeze(pnp_result["tvec"])
                        R_est, _ = cv2.Rodrigues(pnp_result["rvec"])
                        t_err, R_err, T_err = compute_pose_error(R_est, t_est, R_gt, t_gt)
                        print(f"PS: t_err = {t_err:.2f} m, R_err = {R_err:.2f} degree")
                        # if R_diff > 90.0 and R_err < 10.0:
                        #     log_dir = output_dir / method_name
                        #     log_dir.mkdir(exist_ok=True, parents=True)
                        #     # api.visualize(log_path=log_dir, prefix=f"{query_idx}_{train_idx}")
                        #     out_im = draw_matches(
                        #         query_im, query_pts, train_im, train_pts, valid_indices, pnp_result["inlier_idx"]
                        #     )
                        #     cv2.imwrite(log_dir / f"{query_idx}_{train_idx}.png", out_im)

                        run_ba = True
                        if run_ba:
                            pts0_ba = np.array([valid_query_pts[i] for i in pnp_indices])
                            pts1_ba = np.array([valid_train_pts[i] for i in pnp_indices])
                            lmks_ba = np.array([obj_pts[i] for i in pnp_indices])
                            init_pose_ba = np.eye(4)
                            init_pose_ba[:3, :3] = R_est
                            init_pose_ba[:3, 3] = t_est
                            ba_t0 = time.time()
                            ba_result = two_view_ba(pts0_ba, pts1_ba, lmks_ba, K, np.linalg.inv(init_pose_ba))
                            ba_t = time.time() - ba_t0
                            est_pose_ba = np.linalg.inv(ba_result[1][1]) @ ba_result[1][0]
                            t_err_ba, R_err_ba, T_err_ba = compute_pose_error(
                                est_pose_ba[:3, :3], est_pose_ba[:3, 3], R_gt, t_gt
                            )
                            print(f"After BA: t_err = {t_err_ba:.2f} m, R_err = {R_err_ba:.2f} degree")

                            pt_changed = np.linalg.norm(ba_result[2] - lmks_ba, axis=1)
                            lmk_change = np.mean(pt_changed)
                            lmks_change.append(lmk_change)

                            if T_err_ba > 1.0 or T_err_ba < 0.3:
                                if R_diff > 90.0:
                                    category = "bin90-180"
                                elif R_diff > 60.0:
                                    category = "bin60-90"
                                elif R_diff > 30.0:
                                    category = "bin30-60"
                                else:
                                    category = "bin0-30"

                                if T_err_ba > 1.0:
                                    level = "hard"
                                else:
                                    level = "easy"

                                log_dir = output_dir / method_name / category / level
                                if not log_dir.is_dir():
                                    log_dir.mkdir(exist_ok=True, parents=True)
                                # api.visualize(log_path=log_dir, prefix=f"{query_idx}_{train_idx}")
                                out_im = draw_matches(
                                    query_im, query_pts, train_im, train_pts, valid_indices, pnp_result["inlier_idx"]
                                )
                                cv_text = f"t Err: {t_err_ba:.2f} m, R_Err: {R_err_ba:.2f} degree, R_diff: {R_diff:.2f} deg, t_diff: {t_diff:.2f} m"
                                cv2.putText(out_im, cv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                cv2.imwrite(log_dir / f"{query_idx}_{train_idx}.png", out_im)

                    F = F_from_relative_pose(T_rel, K)
                    matching_err = compute_epipolar_errors_bench(query_pts, train_pts, F)
                    # print(f"Epipolar Error = {matching_err:.2f}")

                    # Draw matches.
                    # cv2.imshow("matching", out_im)
                    # cv2.waitKey()
                    matching_stats.append(
                        (
                            query_stamp,
                            train_stamp,
                            pnp_status,
                            pnp_ic,
                            pnp_ir,
                            t_err,
                            R_err,
                            T_err,
                            t_err_ba,
                            R_err_ba,
                            T_err_ba,
                            lmk_change,
                            matching_err,
                            t_diff,
                            R_diff,
                            img0_t,
                            img1_t,
                            match_t,
                            pnp_t,
                            ba_t,
                        )
                    )
            df = pd.DataFrame(
                matching_stats,
                columns=[
                    "query_stamp",
                    "train_stamp",
                    "pnp_status",
                    "pnp_ic",
                    "pnp_ir",
                    "t_err",
                    "R_err",
                    "T_err",
                    "t_err_ba",
                    "R_err_ba",
                    "T_err_ba",
                    "lmk_change",
                    "epipolar_err",
                    "t_diff",
                    "R_diff",
                    "query_t",
                    "train_t",
                    "match_t",
                    "pnp_t",
                    "ba_t",
                ],
            )
            cg_path = output_dir / f"{method_name}_cg_edges_v1.csv"
            df.to_csv(cg_path, index=False)

            if len(lmks_change) > 0:
                print(
                    f"LMKs change stats: max = {np.max(lmks_change)}, min = {np.min(lmks_change)}, avg = {np.mean(lmks_change)}"
                )
                # np.savetxt(
                #     output_dir / f"{method_name}_lmk_change.txt",
                #     np.array(lmks_change),
                #     fmt="%.6f",
                # )


if __name__ == "__main__":
    main()
