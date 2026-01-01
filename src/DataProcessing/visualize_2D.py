import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

args = sys.argv
seed = args[1]
p = args[2]
tau_B = args[3]

system_L = 100
window = 50
xmin, xmax = system_L / 2 - window, system_L / 2 + window
ymin, ymax = system_L / 2 - window, system_L / 2 + window

os = [0]
us = [1]
cs = [0]

p_max = 1.0
ppi = 72  # points per inche
interaction_radius_fold = 2.5


def plot_hist_new(df_t, fig, ax_n, t_devo, fontsize=12):
    ax_n.set_aspect("equal")  # アスペクト比を等しくする
    ax_length = ax_n.bbox.get_points()[1][0] - ax_n.bbox.get_points()[0][0]
    ax_point = ax_length * ppi / fig.dpi
    xsize = xmax - xmin
    fact = ax_point / xsize
    eq_radius = 1.0
    size = eq_radius
    size *= 2 * fact

    cells = ax_n.quiver(
        df_t[5],
        df_t[6],
        np.cos(df_t[7]),
        np.sin(df_t[7]),
        color="w",
        edgecolor="k",
        linewidth=1.0,
    )  # color = nutrient
    ax_n.scatter(
        df_t[5], df_t[6], c="None", s=size**2, edgecolors="k", linewidths=(size / 100)
    )
    ax_n.set_xlim(xmin, xmax)
    ax_n.set_ylim(ymin, ymax)
    ax_n.set_xlabel(r"$x$", fontsize=fontsize)
    ax_n.set_ylabel(r"$y$", fontsize=fontsize)
    ax_n.set_xticks(np.arange(xmin, xmax + 1, 20))
    ax_n.set_yticks(np.arange(ymin, ymax + 1, 20))


def read_data_linear_cell_div(p, tau_B, seed="1"):
    arg = [
        "",
        "100",
        "30",
        "0.05",
        "300000",
        "2.5",
        seed,
        "10",
        "1",
        p,
        "1000",
        tau_B,
    ]
    id_ = seed + "_" + p + "_" + tau_B + "_"
    type_ = "model_base_2D"
    raw_file = "../data_" + type_ + "/redev" + id_ + type_ + ".txt"
    df_raw = pd.read_csv(raw_file, sep=" ", header=None)
    df_raw.reset_index(inplace=True)
    df_raw = df_raw.shift(axis=1)
    df_raw.drop("index", axis=1, inplace=True)

    return df_raw, id_


time_points = [5000, 10000, 50000, 100000]
fld = 4
fontsize = 15
resolution = 5
base = pow(10, 1.0 / resolution)

# 先に全データ読み込み
dfs = []
ids = []
df, id_ = read_data_linear_cell_div(p, tau_B, seed=seed)
dfs.append(df)
ids.append(id_)


# 描画
fig, axes = plt.subplots(
    1,
    len(time_points),
    figsize=(10 * fld, 2.0 * fld),
    gridspec_kw={"hspace": 0.01, "wspace": -0.5},
)

# p, tau_B のサイズを計算

p_size = 0.01 * pow(base, float(p))
tau_B_size = 0.01 * pow(base, float(tau_B))


# 数値を整形
def format_sci(num):
    num_fmt = "{:.2e}".format(num)
    num_mant = num_fmt[:4]
    num_exp = num_fmt[5:]
    if num_exp[1] == "0":
        num_exp = num_exp[0] + num_exp[2:]
    if num_exp[0] == "+":
        num_exp = num_exp[1:]
    return num_mant, num_exp


p_num, p_exp = format_sci(p_size)
tau_B_num, tau_B_exp = format_sci(tau_B_size)

for col, t in enumerate(time_points):
    df_t = df[df[2] == t]
    ax = axes[col]
    plot_hist_new(df_t, fig, ax, t, fontsize=fontsize * fld)

    # 軸ラベルなど非表示
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


# 図のレイアウトをコンパクトに
fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.1)

fig.savefig(
    "../fig/fig" + id_ + ".pdf",
    dpi=300,
    bbox_inches="tight",
)
