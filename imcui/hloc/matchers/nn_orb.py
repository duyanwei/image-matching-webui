import torch

from ..utils.base_model import BaseModel

import cv2
import numpy as np

class NearestNeighborORB(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold": None,
        "do_mutual_check": True,
    }
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = self.conf["do_mutual_check"])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    def _forward(self, data):
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2],
                -1,
                device=data["descriptors0"].device,
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }
        desc0_cpu = data["descriptors0"].cpu().numpy()
        desc1_cpu = data["descriptors1"].cpu().numpy()
        print(desc0_cpu.shape)

        desc0 = np.squeeze(desc0_cpu, axis=0).transpose(1, 0)
        desc1 = np.squeeze(desc1_cpu, axis=0).transpose(1, 0)

        assert len(desc0.shape) == 2
        assert desc0.shape[0] > 0

        # Match descriptors.
        # matches = self.matcher.match(desc0, desc1)
        knn = self.matcher.knnMatch(desc0, desc1, k=2)
        good = []
        dists = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance > self.conf["ratio_threshold"] * n.distance:
                continue
            dists.append(m.distance)
            good.append(m)
        matches0 = np.full((1, len(desc0)), -1, dtype=int)
        scores0 = np.full_like(matches0, 0, dtype=float)
        # max_dist = np.max(dists)
        # steepness = 0.1
        for dmatch in good:
            matches0[0, dmatch.queryIdx] = dmatch.trainIdx
            # scores0[0, dmatch.queryIdx] = 1.0 / (1.0 + dmatch.distance)
            # scores0[0, dmatch.queryIdx] = 1.0 / (1.0 + np.exp(steepness * (dmatch.distance - max_dist / 2)))
            scores0[0, dmatch.queryIdx] = np.clip(1.0 - dmatch.distance / 256.0, 0.0, 1.0)
        matches0 = torch.from_numpy(matches0)
        scores0 = torch.from_numpy(scores0)

        return {
            "matches0": matches0,
            "matching_scores0": scores0,
        }
