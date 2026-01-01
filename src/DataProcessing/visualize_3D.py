#!/usr/bin/env python
# coding: utf-8
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

import os
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

seed_str = sys.argv[1]
p_str = sys.argv[2]
tau_B_str = sys.argv[3]

args = [
    "100",
    "30",
    "0.05",
    "300000",
    "0",
    "0",
    "2.5",
    seed_str,
    "10",
    "1",
    # "0",
    p_str,
    "1000",
    tau_B_str,
]
# file_id = "redev"
# for arg in args:
#     file_id += arg + "_"
id_ = seed_str + "_" + p_str + "_" + tau_B_str + "_"
type_ = "model_base_3D"
# Read the time series xyz data from the file in ../data3D_model01 into a pandas dataframe
file_path = "../data_" + type_ + "/redev" + id_ + type_ + ".txt"

# Assuming the file has columns 'time', 'x', 'y', 'z'
df_raw = pd.read_csv(file_path, sep=" ", header=None)
df_raw.reset_index(inplace=True)
df_raw = df_raw.shift(axis=1)
df_raw.drop("index", axis=1, inplace=True)

print(df_raw)


def p1(phi, theta):
    return np.sin(phi) * np.cos(theta)


def p2(phi, theta):
    return np.sin(phi) * np.sin(theta)


def p3(phi, theta):
    return np.cos(phi)


fig_p, ax_ = plt.subplots(figsize=(10, 10))
p_abs = 0.5

system_L = 100
window = 10
xmin, xmax = system_L / 2 - window, system_L / 2 + window
ymin, ymax = system_L / 2 - window, system_L / 2 + window
zmin, zmax = system_L / 2 - window, system_L / 2 + window
ax_.set_xlim(xmin, xmax)
ax_.set_ylim(ymin, ymax)

ppi = 72  # points per inche
size = 1.0
ax_.set_aspect("auto")  # アスペクト比を等しくする
ax_length = ax_.bbox.get_points()[1][0] - ax_.bbox.get_points()[0][0]
ax_point = ax_length * ppi / fig_p.dpi
xsize = xmax - xmin
fact = ax_point / xsize
size *= 2 * fact


eq_r = 1
# Filter the data for the cross-sectional view at z = a

os.makedirs(
    "../fig/" + p_str + "_" + tau_B_str,
    exist_ok=True,
)

t_devos = [15000, 60000, 105000, 150000, 195000, 240000, 285000]
a_list = [41, 44, 47, 50, 53, 56, 59]
for t_devo in t_devos:
    df_t = df_raw[df_raw[2] == t_devo]
    os.makedirs(
        "../fig/" + p_str + "_" + tau_B_str + "/" + str(t_devo),
        exist_ok=True,
    )
    for a in a_list:
        cross_section_data = df_t[(df_t[7] <= a + eq_r) & (df_t[7] >= a - eq_r)]
        e2s = 1700
        # Create a 2D scatter plot for the cross-sectional view
        ax_.scatter(
            cross_section_data[5].values,
            cross_section_data[6].values,
            c="b",
            alpha=0.6,
            s=e2s * np.sqrt(eq_r**2 - (a - cross_section_data[7]) ** 2),
            edgecolors="k",
            linewidths=(1 / 100),
        )
        fontsize = 90
        ax_.set_xlabel(r"$x$", fontsize=fontsize)
        ax_.set_ylabel(r"$y$", fontsize=fontsize)
        ax_.set_xlim(xmin, xmax)
        ax_.set_ylim(ymin, ymax)
        ax_.set_xticks([40, 60])
        ax_.set_yticks([40, 60])
        ax_.tick_params(bottom=False, left=False, right=False, top=False)
        pad = 10
        fold = 0.75
        ax_.xaxis.set_tick_params(labelsize=fontsize * fold, pad=pad)
        ax_.yaxis.set_tick_params(labelsize=fontsize * fold, pad=pad)
        ax_.xaxis.set_label_coords(0.5, -0.1)
        ax_.yaxis.set_label_coords(-0.2, 0.5)
        ax_.set_title(f"$z = {a}$", fontsize=fontsize, pad=30)
        fig_p.tight_layout()
        # add margin to fig on top and bottom with length l

        fig_p.savefig(
            "../fig/"
            + p_str
            + "_"
            + tau_B_str
            + "/"
            + str(t_devo)
            + "/"
            + str(a)
            + "_final.png"
        )
        ax_.cla()


args = sys.argv


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [
        cv2.resize(
            im,
            (int(im.shape[1] * h_min / im.shape[0]), h_min),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.hconcat(im_list_resize)


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


time_list = [15000, 60000, 105000, 150000, 195000, 240000, 285000]
z_list = [41, 44, 47, 50, 53, 56, 59]

# p_str = args[2]
# tau_B_str = args[3]

im_list = []
for t in time_list:
    zview_list = []
    # obj_view_im = cv2.imread(
    #     "../fig/" + p_str + "_" + tau_B_str + "/" + str(t) + "_final.png"
    # )
    # cv2.putText(
    #     obj_view_im,
    #     r"t = " + str(t),
    #     (0, 150),
    #     cv2.FONT_HERSHEY_PLAIN,
    #     13,
    #     (0, 0, 0),
    #     5,
    #     cv2.LINE_AA,
    # )
    for z in z_list:
        im = cv2.imread(
            "../fig/"
            + p_str
            + "_"
            + tau_B_str
            + "/"
            + str(t)
            + "/"
            + str(z)
            + "_final.png"
        )
        # add margin on top and bottom with length l
        l = 50
        im = np.vstack([np.ones((l, im.shape[1], 3), dtype=np.uint8) * 255, im])
        im = np.vstack([im, np.ones((l, im.shape[1], 3), dtype=np.uint8) * 255])
        zview_list.append(im)
    im_tile = hconcat_resize_min(
        [
            # obj_view_im,
            zview_list[0],
            zview_list[1],
            zview_list[2],
            zview_list[3],
            zview_list[4],
            zview_list[5],
            zview_list[6],
        ]
    )
    im_list.append(im_tile)

final_tile = cv2.vconcat(im_list)
cv2.imwrite(
    "../fig/" + seed_str + "_" + p_str + "_" + tau_B_str + ".png",
    final_tile,
)  # Save the image
