import warnings

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale
from omegaconf import OmegaConf
from packaging import version

from .. import logger

from ..utils.base_model import BaseModel


def run_nms(points, scales, angles, image_shape, nms_radius, scores=None):
    if len(points) < 1:
        return []
    h, w = image_shape
    ij = np.round(points - 0.5).astype(int).T[::-1]

    # Remove duplicate points (identical coordinates).
    # Pick highest scale or score
    s = scales if scores is None else scores
    buffer = np.zeros((h, w))
    np.maximum.at(buffer, tuple(ij), s)
    keep = np.where(buffer[tuple(ij)] == s)[0]

    # Pick lowest angle (arbitrary).
    ij = ij[:, keep]
    buffer[:] = np.inf
    o_abs = np.abs(angles[keep])
    np.minimum.at(buffer, tuple(ij), o_abs)
    mask = buffer[tuple(ij)] == o_abs
    ij = ij[:, mask]
    keep = keep[mask]

    if nms_radius > 0:
        # Apply NMS on the remaining points
        buffer[:] = 0
        buffer[tuple(ij)] = s[keep]  # scores or scale

        local_max = torch.nn.functional.max_pool2d(
            torch.from_numpy(buffer).unsqueeze(0),
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(0)
        is_local_max = buffer == local_max.numpy()
        keep = keep[is_local_max[tuple(ij)]]
    return keep


def run_opencv_orb(features: cv2.Feature2D, image: np.ndarray) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector.
    Optionally, perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
    Returns:
        keypoints: 1D array of detected cv2.KeyPoint
        scores: 1D array of responses
        descriptors: 1D array of descriptors
    """
    detections, descriptors = features.detectAndCompute(image, None)
    points = np.array([k.pt for k in detections], dtype=np.float32)
    scores = np.array([k.response for k in detections], dtype=np.float32)
    scales = np.array([k.size for k in detections], dtype=np.float32)
    angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))
    return points, scores, scales, angles, descriptors


class ORB(BaseModel):
    default_conf = {
        "nms_radius": 0,  # None to disable filtering entirely.
        "max_keypoints": 4096,
        "scale_factor": 1.2,
        "num_levels": 8,
        "edge_threshold": 10,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.conf = OmegaConf.create(self.conf)
        self.orb = cv2.ORB_create(
            nfeatures = self.conf.max_keypoints,
            scaleFactor = self.conf.scale_factor,
            nlevels = self.conf.num_levels,
            edgeThreshold = self.conf.edge_threshold,
        )
        # print(self.conf.max_keypoints)
        logger.info("Load ORB model done.")

    def extract_single_image(self, image: torch.Tensor):
        image_np = image.cpu().numpy().squeeze(0)
        keypoints, scores, scales, angles, descriptors = run_opencv_orb(
            self.orb, (image_np * 255.0).astype(np.uint8)
        )
        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": angles,
            "descriptors": descriptors,
        }
        if scores is not None:
            pred["scores"] = scores

        if len(keypoints) > 0 and self.conf.nms_radius is not None:
            keep = run_nms(
                pred["keypoints"],
                pred["scales"],
                pred["oris"],
                image_np.shape,
                self.conf.nms_radius,
                scores=pred.get("scores"),
            )
            pred = {k: v[keep] for k, v in pred.items()}

        pred = {k: torch.from_numpy(v) for k, v in pred.items()}
        if scores is not None:
            # Keep the k keypoints with highest score
            num_points = self.conf.max_keypoints
            if num_points is not None and len(pred["keypoints"]) > num_points:
                indices = torch.topk(pred["scores"], num_points).indices
                pred = {k: v[indices] for k, v in pred.items()}
        return pred

    def _forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        device = image.device
        image = image.cpu()
        pred = []
        for k in range(len(image)):
            img = image[k]
            if "image_size" in data.keys():
                # avoid extracting points in padded areas
                w, h = data["image_size"][k]
                img = img[:, :h, :w]
            p = self.extract_single_image(img)
            pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0).to(device) for k in pred[0]}
        pred["descriptors"] = pred["descriptors"].permute(0, 2, 1)
        pred["keypoint_scores"] = pred["scores"].clone()
        return pred
