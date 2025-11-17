import torch

from .. import logger

from ..utils.base_model import BaseModel

import time


class XFeatLightGlue(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": 8000,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        # TODO: this is hardcoded, should be changed at the first level.
        # self.conf["max_keypoints"] = 1024
        self.net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(dense) model done.")

    def _forward(self, data):
        # we use results from one batch
        im0 = data["image0"]
        im1 = data["image1"]
        # Compute coarse feats
        t0 = time.time()
        out0 = self.net.detectAndCompute(im0, top_k=self.conf["max_keypoints"])[0]
        t1 = time.time()
        out1 = self.net.detectAndCompute(im1, top_k=self.conf["max_keypoints"])[0]
        t2 = time.time()
        out0.update({"image_size": (im0.shape[-1], im0.shape[-2])})  # W H
        out1.update({"image_size": (im1.shape[-1], im1.shape[-2])})  # W H
        t3 = time.time()
        pred = self.net.match_lighterglue(out0, out1)
        t4 = time.time()
        if len(pred) == 3:
            mkpts_0, mkpts_1, _ = pred
        else:
            mkpts_0, mkpts_1 = pred
        mkpts_0 = torch.from_numpy(mkpts_0)  # n x 2
        mkpts_1 = torch.from_numpy(mkpts_1)  # n x 2
        pred = {
            "keypoints0": out0["keypoints"].squeeze(),
            "keypoints1": out1["keypoints"].squeeze(),
            "mkeypoints0": mkpts_0,
            "mkeypoints1": mkpts_1,
            "mconf": torch.ones_like(mkpts_0[:, 0]),
            "img0_extract_t": t1 - t0,
            "img1_extract_t": t2 - t1,
            "match_t": t4 - t3,
        }
        return pred
