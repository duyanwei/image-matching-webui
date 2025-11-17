import cv2
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from imcui.hloc import logger
from imcui.ui.utils import DEVICE, get_matcher_zoo, load_config
from imcui.api import ImageMatchingAPI


def F_from_relative_pose(T_rel, K):
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


def verify_with_gt_poses(kpts0, kpts1, H, F, Twc0_gt, Twc1_gt, K):

    T_rel = np.linalg.inv(Twc1_gt) @ Twc0_gt
    R_gt = T_rel[:3, :3]
    t_gt = T_rel[:3, 3]
    rot_diff = np.linalg.norm(Rotation.from_matrix(R_gt).as_rotvec(degrees=True))
    trans_diff = np.linalg.norm(t_gt)

    rot_error = np.nan
    trans_error = np.nan
    if F is not None and len(kpts0) > 10:
        E = K.T @ F @ K
        num_inliers, R_rec, t_rec, _ = cv2.recoverPose(E, kpts0, kpts1, K)
        if num_inliers > len(kpts0) * 0.1:
            rot_error = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_rec) - 1) / 2, -1, 1)))
            trans_error = np.degrees(
                np.arccos(np.dot(t_gt.squeeze() / np.linalg.norm(t_gt), t_rec.squeeze() / np.linalg.norm(t_rec)))
            )

    F_gt = F_from_relative_pose(T_rel, K)
    # Compute distances
    distances = []
    if len(kpts1) > 3:
        if kpts1.shape[1] == 2:
            kpts1 = np.hstack((kpts1, np.ones((kpts1.shape[0], 1))))
        if kpts0.shape[1] == 2:
            kpts0 = np.hstack((kpts0, np.ones((kpts0.shape[0], 1))))
        for p0, p1 in zip(kpts0, kpts1):
            l1 = F_gt @ p0
            # Normalize line (a x + b y + c = 0)
            norm = np.sqrt(l1[0] ** 2 + l1[1] ** 2)
            if norm == 0:
                dist = 0  # Degenerate
            else:
                dist = np.abs(np.dot(l1, p1)) / norm
            distances.append(dist)

    avg_distance = np.mean(distances) if distances else np.inf
    return avg_distance, rot_error, trans_error, rot_diff, trans_diff


def pose_array_to_mat(arr):
    assert len(arr) == 7
    mat = np.eye(4)
    mat[:3, 3] = arr[:3]
    mat[:3, :3] = Rotation.from_quat(arr[3:]).as_matrix()
    return mat


def load_gt_poses(filepath):
    with open(filepath, "r") as file:
        # ... proceed to load the data
        data = json.load(file)
        query_pose = data["query_pose"]
        train_pose = data["train_pose"]
        return pose_array_to_mat(query_pose), pose_array_to_mat(train_pose)


def compute_metrics(name, pred, query_Twc=None, train_Twc=None, K=None, D=None):

    # Epipolar metrics.
    H_reproj_err = np.nan
    F_ep_dist = np.nan
    rot_err = np.nan
    trans_err = np.nan
    rot_diff = np.nan
    trans_diff = np.nan

    if query_Twc is not None and train_Twc is not None and K is not None:
        F_ep_dist, rot_err, trans_err, rot_diff, trans_diff = verify_with_gt_poses(
            pred["mmkeypoints0_orig"],  # Inliers after ransac
            pred["mmkeypoints1_orig"],  # Inliers after ransac
            pred.get("H", None),
            pred["geom_info"].get("Fundamental", None),
            query_Twc,
            train_Twc,
            K,
        )

    matched_ratio0 = np.nan
    matched_ratio1 = np.nan
    inlier_ratio0 = np.nan
    inlier_ratio1 = np.nan

    if len(pred["keypoints0_orig"]) > 0:
        matched_ratio0 = len(pred["mkeypoints0_orig"]) / len(pred["keypoints0_orig"])
    if len(pred["keypoints1_orig"]) > 0:
        matched_ratio1 = len(pred["mkeypoints1_orig"]) / len(pred["keypoints1_orig"])
    if len(pred["mkeypoints1_orig"]) > 0:
        inlier_ratio1 = len(pred["mmkeypoints1_orig"]) / len(pred["mkeypoints1_orig"])
    if len(pred["mkeypoints0_orig"]) > 0:
        inlier_ratio0 = len(pred["mmkeypoints0_orig"]) / len(pred["mkeypoints0_orig"])

    return (
        name,
        len(pred["keypoints0_orig"]),
        len(pred["keypoints1_orig"]),
        matched_ratio0,
        matched_ratio1,
        inlier_ratio0,
        inlier_ratio1,
        pred["img0_extract_t"],
        pred["img1_extract_t"],
        pred["match_t"],
        F_ep_dist,
        rot_err,
        trans_err,
        rot_diff,
        trans_diff,
    )


def main():

    # Query image-pair subdir.
    dataroot = Path("/mnt/DATA/experiments/semantic/miniset/median/cid/")
    datadir = sorted([x for x in dataroot.iterdir() if x.is_dir()])
    print(f"Found {len(datadir)} image pairs.")

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

    # Load config for detection and matching.
    config = load_config(ROOT / "config/batch.yaml")
    matcher_zoo_restored = get_matcher_zoo(config["matcher_zoo"])

    for pair_dir in datadir:
        query_path = pair_dir / "query"
        train_path = pair_dir / "train"

        query_list = list(query_path.glob("*.png")) + list(query_path.glob("*.jpg"))
        train_list = list(train_path.glob("*.png")) + list(train_path.glob("*.jpg"))

        # Skip empty image pair.
        if len(query_list) < 1 or len(train_list) < 1:
            continue

        # Load images.
        query = cv2.imread(str(query_list[0]))[:, :, ::-1]  # RGB
        train = cv2.imread(str(train_list[0]))[:, :, ::-1]  # RGB

        result_dir = pair_dir / "result"
        result = []
        for k, v in matcher_zoo_restored.items():
            enable = config["matcher_zoo"][k].get("enable", True)
            skip_ci = config["matcher_zoo"][k].get("skip_ci", False)
            if not enable or skip_ci:
                print(f"Skipping {k} ...")
                continue
            if v["dense"]:
                if "xfeat+lightglue" not in k:
                    continue
            print(f"Testing {k} ...")
            # break
            api = ImageMatchingAPI(conf=v, device=DEVICE, max_keypoints=1024)
            # Run core.
            pred = api(query, train)
            assert pred is not None

            # Log and save visual result.
            if v["dense"]:
                method_name = str(api.get_conf()["matcher"]["model"]["name"])
            else:
                method_name = "{}_{}".format(
                    str(api.get_conf()["feature"]["model"]["name"]),
                    str(api.get_conf()["matcher"]["model"]["name"]),
                )
            log_dir = result_dir / method_name
            log_dir.mkdir(exist_ok=True, parents=True)
            api.visualize(log_path=log_dir)

            # Check if gt pose is available for verification.
            query_Twc = None
            train_Twc = None
            meta_file = pair_dir / "meta.json"
            if meta_file.is_file():
                query_Twc, train_Twc = load_gt_poses(meta_file)
            result.append(compute_metrics(method_name, pred, query_Twc, train_Twc, K, D))

        columns_names = [
            "method",
            "kpts0",
            "kpts1",
            "matched_kpts0_ratio",
            "matched_kpts1_ratio",
            "matched_kpts0_inlier_ratio",
            "matched_kpts1_inlier_ratio",
            "kpts0_t",
            "kpts1_t",
            "match_t",
            "F_ep_dist",
            "rot_error",
            "trans_error",
            "rot_diff",
            "trans_diff",
        ]
        df = pd.DataFrame(result, columns=columns_names)
        out_filename = result_dir / "metrics.csv"
        df.to_csv(out_filename, index=False)  # index=False prevents writing the DataFrame index as a column
        print(f"Saved to {out_filename}")


if __name__ == "__main__":
    main()
