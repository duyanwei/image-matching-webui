#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file plot_result.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 11-17-2025
@version 1.0
@license Copyright (c) 2025
@desc None
"""


from pathlib import Path
from matplotlib import pyplot as plt
import cv2

dataroot = Path("/mnt/DATA/experiments/semantic/miniset/median/cid/")
datadir = sorted([x for x in dataroot.iterdir() if x.is_dir()])
print(f"Found {len(datadir)} image pairs.")


methods = [
    "xfeat_lightglue",
    # "xfeat_nearest_neighbor",
    # "superpoint_lightglue",
    "superpoint_superglue",
    # "sift_lightglue",
    "sift_nearest_neighbor",
    "orb_nn_orb",
]

result_dir = dataroot / "result"
result_dir.mkdir(exist_ok=True, parents=True)

for pair_dir in datadir:
    pair_result_dir = pair_dir / "result"

    num = len(methods)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs = axs.flatten()
    for ax in axs:
        ax.set_axis_off()

    for ax, mname in zip(axs, methods):
        im_path = pair_result_dir / mname / f"img_matches_ransac_{mname}.png"
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im)
        ax.set_title(mname, y=0.9)

    plt.tight_layout()
    plt.show()
    fig.savefig(result_dir / f"{pair_dir.stem}_vis.png", dpi=200)
