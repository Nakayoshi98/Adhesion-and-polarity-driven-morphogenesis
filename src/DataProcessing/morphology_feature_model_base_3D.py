# coding: utf-8

import sys
import numpy as np
import pandas as pd
from ripser import ripser
import networkx as nx

args = sys.argv
id_ = args[1]
for i in range(2, len(args)):
    id_ += "_" + args[i]
id_ += "_"

import os

os.makedirs(
    "../feature_analysis/",
    exist_ok=True,
)


type_ = "model_base_3D"
filename = "../data_" + type_ + "/redev" + id_ + type_ + ".txt"
# print(filename)
table = pd.read_csv(filename, sep=" ", header=None)
table.reset_index(inplace=True)
table = table.shift(axis=1)
table.drop("index", axis=1, inplace=True)

T_DEVO = 300000
interval = 2000
tau_div = 1000
tau_rec = 100

system_L = 100
p_max = 1.0
interaction_radius_fold = 2.5
eq_radius = 1.0
size = eq_radius


def persistence_homology(df_t):
    data = df_t[[5, 6]].values
    return ripser(data, distance_matrix=False)["dgms"]


def generate_nx_graph(df_t):
    data = df_t[[5, 6, 7]].values
    G_l = nx.Graph()
    G_a = nx.Graph()
    for i in range(len(data)):
        G_l.add_node(i, pos=(data[i][0], data[i][1], data[i][2]))
        G_a.add_node(i, pos=(data[i][0], data[i][1], data[i][2]))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if np.linalg.norm(data[i] - data[j]) < interaction_radius_fold * eq_radius:
                G_l.add_edge(i, j, weight=np.linalg.norm(data[i] - data[j]))
                G_a.add_edge(i, j)
    return G_l, G_a


def count_periphery(G_a):
    degree = np.array(list(dict(nx.degree(G_a)).values()))
    return degree[degree <= 4].size


def signed_curvature(p1, p2, dr):
    # Calculate the difference in normal vectors
    dp = p2 - p1
    # Calculate the curvature
    curvature = np.linalg.norm(dp) / np.linalg.norm(dr)
    # Determine the sign of the curvature
    sign = np.sign(np.dot(dp, dr))
    return curvature * sign


def polarity_correlation_function(df_t, G_a):
    polarity = df_t[7].values
    pos = df_t[[5, 6]].values
    correlation_list = np.zeros(len(polarity))
    mean_curvature_list = np.zeros(len(polarity))
    median_curvature_list = np.zeros(len(polarity))
    for i in range(len(polarity)):
        pi = np.array([np.cos(polarity[i]), np.sin(polarity[i])])
        pi_prop = np.array([-pi[1], pi[0]])
        ri = pos[i]
        if len(G_a.adj[i]) == 0:
            correlation_list[i] = 0
            continue  # skip if the cell has no neighbors
        curvature_list = np.zeros(len(G_a.adj[i]))
        n = 0
        for j in G_a.adj[i]:
            pj = np.array([np.cos(polarity[j]), np.sin(polarity[j])])
            rj = pos[j]
            dot_product = np.dot(pi, pj)
            correlation_list[i] += dot_product / len(G_a.adj[i])
            dr_eff_cos = np.dot(rj - ri, pi_prop) / np.linalg.norm(rj - ri)
            if abs(dr_eff_cos) < 0.8:
                curvature_list[n] = None
            else:
                curvature_list[n] = signed_curvature(pi, pj, rj - ri)
            n += 1

        median_curvature_list[i] = np.median(curvature_list[~np.isnan(curvature_list)])
        mean_curvature_list[i] = np.mean(curvature_list[~np.isnan(curvature_list)])
    notnan_mean_curvature_list = mean_curvature_list[~np.isnan(mean_curvature_list)]
    notnan_median_curvature_list = median_curvature_list[
        ~np.isnan(median_curvature_list)
    ]
    if len(notnan_mean_curvature_list) == 0:
        min_mean_curvature = None
        max_mean_curvature = None
        mean_mean_curvature = None
        std_mean_curvature = None
        negative_mean_curvature_ratio = None
        under005_mean_curvature_ratio = None
        under01_mean_curvature_ratio = None
        under02_mean_curvature_ratio = None
        under03_mean_curvature_ratio = None
    else:
        min_mean_curvature = notnan_mean_curvature_list.min()
        max_mean_curvature = notnan_mean_curvature_list.max()
        mean_mean_curvature = notnan_mean_curvature_list.mean()
        std_mean_curvature = notnan_mean_curvature_list.std()
        negative_mean_curvature_ratio = (notnan_mean_curvature_list < 0).sum() / len(
            notnan_mean_curvature_list
        )
        under005_mean_curvature_ratio = (
            notnan_mean_curvature_list < -0.05
        ).sum() / len(notnan_mean_curvature_list)
        under01_mean_curvature_ratio = (notnan_mean_curvature_list < -0.1).sum() / len(
            notnan_mean_curvature_list
        )
        under02_mean_curvature_ratio = (notnan_mean_curvature_list < -0.2).sum() / len(
            notnan_mean_curvature_list
        )
        under03_mean_curvature_ratio = (notnan_mean_curvature_list < -0.3).sum() / len(
            notnan_mean_curvature_list
        )
    if len(notnan_median_curvature_list) == 0:
        min_median_curvature = None
        max_median_curvature = None
        mean_median_curvature = None
        std_median_curvature = None
    else:
        min_median_curvature = notnan_median_curvature_list.min()
        max_median_curvature = notnan_median_curvature_list.max()
        mean_median_curvature = notnan_median_curvature_list.mean()
        std_median_curvature = notnan_median_curvature_list.std()
    notnan_ratio_mean = notnan_mean_curvature_list.size / len(mean_curvature_list)

    mean_field = np.array([np.cos(polarity).mean(), np.sin(polarity).mean()])

    return (
        correlation_list.mean(),
        mean_field,
        min_mean_curvature,
        max_mean_curvature,
        min_median_curvature,
        max_median_curvature,
        mean_mean_curvature,
        mean_median_curvature,
        std_mean_curvature,
        std_median_curvature,
        negative_mean_curvature_ratio,
        under005_mean_curvature_ratio,
        under01_mean_curvature_ratio,
        under02_mean_curvature_ratio,
        under03_mean_curvature_ratio,
        notnan_ratio_mean,
    )


def count_layers(df_t):
    polarity = df_t[7].values
    pos = df_t[[5, 6]].values
    layer_list = np.zeros(len(polarity))
    for i in range(len(polarity)):
        list_i = []
        list_i.append(i)
        pi = np.array([np.cos(polarity[i]), np.sin(polarity[i])])
        # equation of line passing through pos_i and parallel to pi
        a = pi[1]
        b = -pi[0]
        c = -a * pos[i][0] - b * pos[i][1]
        for j in range(len(polarity)):
            pj = np.array([np.cos(polarity[j]), np.sin(polarity[j])])
            # distance from pos_j to the line
            dist = np.abs(a * pos[j][0] + b * pos[j][1] + c) / np.sqrt(a**2 + b**2)
            if dist < eq_radius and np.dot(pi, pj) > np.sqrt(3) / 2:
                list_i.append(j)
        for j in list_i:
            for k in list_i:
                if (
                    j != k
                    and np.linalg.norm(pos[j] - pos[k])
                    < eq_radius * interaction_radius_fold
                ):
                    layer_list[i] += 1
                    break
    mean, std, median = layer_list.mean(), layer_list.std(), np.median(layer_list)
    mode = np.bincount(layer_list.astype(int)).argmax()
    return mean, std, median, mode


dictio = {
    "t_devo": [],
    "num_peri": [],
    "max_dist": [],
    "max_pair": [],
    "straight_dist": [],
    "curve_index": [],
    "curve_index_inverse": [],
    "circular_index": [],
    "polarity_correlation": [],
    "birth_max": [],
    "death_max": [],
    "persitence_max": [],
    "birth_from_beginning_max": [],
    "death_from_beginning_max": [],
    "persitence_from_beginning_max": [],
    "mean_layer": [],
    "std_layer": [],
    "median_layer": [],
    "mode_layer": [],
    "touch_boundary": [],
    "polarity_mean_field": [],
    "polarity_mean_field_size": [],
    "min_mean_curvature": [],
    "max_mean_curvature": [],
    "min_median_curvature": [],
    "max_median_curvature": [],
    "mean_curvature_mean": [],
    "median_curvature_mean": [],
    "std_mean_curvature": [],
    "std_median_curvature": [],
    "negative_mean_curvature_ratio": [],
    "under005_mean_curvature_ratio": [],
    "under01_mean_curvature_ratio": [],
    "under02_mean_curvature_ratio": [],
    "under03_mean_curvature_ratio": [],
    "notnan_ratio_mean": [],
}

for t_devo in range(
    tau_div - tau_rec, T_DEVO - tau_rec, interval
):  # 分裂する直前での特徴量を取得
    df = table[table[2] == t_devo]
    G_l, G_a = generate_nx_graph(df)
    path_mat = np.ma.masked_invalid(nx.floyd_warshall_numpy(G_l))
    num_peri = count_periphery(G_a)
    max_dist = np.max(path_mat)
    max_pair = np.unravel_index(np.argmax(path_mat), path_mat.shape)
    straight_dist = np.linalg.norm(
        df[[5, 6]].values[max_pair[0]] - df[[5, 6]].values[max_pair[1]]
    )
    curve_index = straight_dist / max_dist  # wraparound or sphere
    curve_index_inverse = max_dist / straight_dist  # wraparound or sphere
    circular_index = G_a.size() / (num_peri**2)

    (
        polarity_correlation,
        mean_field,
        min_mean_curvature,
        max_mean_curvature,
        min_median_curvature,
        max_median_curvature,
        mean_curvature_mean,
        median_curvature_mean,
        std_mean_curvature,
        std_median_curvature,
        negative_mean_curvature_ratio,
        under005_mean_curvature_ratio,
        under01_mean_curvature_ratio,
        under02_mean_curvature_ratio,
        under03_mean_curvature_ratio,
        notnan_ratio_mean,
    ) = polarity_correlation_function(df, G_a)

    diagram = persistence_homology(df)
    H1_array = diagram[1]  # ndarray
    if H1_array.size != 0:
        hole_id_max = (H1_array[:, 1] - H1_array[:, 0]).argmax()
        birth_max = H1_array[hole_id_max, 0]
        death_max = H1_array[hole_id_max, 1]
        persitence_max = death_max - birth_max  # persistence of the largest hole
    else:
        birth_max = 0
        death_max = 0
        persitence_max = 0

    H1_0_from_beginning = H1_array[:, 0][H1_array[:, 0] < 2.5]
    H1_1_from_beginning = H1_array[:, 1][
        H1_array[:, 0] < 2.5
    ]  # 1.25じゃない？？？または1じゃない？？
    if H1_0_from_beginning.size != 0:
        hole_id_from_beginning_max = (
            H1_1_from_beginning - H1_0_from_beginning
        ).argmax()
        birth_from_beginning_max = H1_0_from_beginning[hole_id_from_beginning_max]
        death_from_beginning_max = H1_1_from_beginning[hole_id_from_beginning_max]
        persitence_from_beginning_max = (
            death_from_beginning_max - birth_from_beginning_max
        )  # persistence of the largest hole from the beginning
    else:
        birth_from_beginning_max = 0
        death_from_beginning_max = 0
        persitence_from_beginning_max = 0

    mean_layer, std_layer, median_layer, mode_layer = count_layers(df)

    # check if any of the cells touch the boundary
    touch_boundary = 0
    pos = df[[5, 6]].values
    if (pos < size).any() or (pos > system_L - size).any():
        touch_boundary = 1

    dictio["t_devo"].append(t_devo)
    dictio["num_peri"].append(num_peri)
    dictio["max_pair"].append(max_pair)
    dictio["max_dist"].append(max_dist)
    dictio["straight_dist"].append(straight_dist)
    dictio["curve_index"].append(curve_index)
    dictio["curve_index_inverse"].append(curve_index_inverse)
    dictio["circular_index"].append(circular_index)
    dictio["polarity_correlation"].append(polarity_correlation)
    dictio["birth_max"].append(birth_max)
    dictio["death_max"].append(death_max)
    dictio["persitence_max"].append(persitence_max)
    dictio["birth_from_beginning_max"].append(birth_from_beginning_max)
    dictio["death_from_beginning_max"].append(death_from_beginning_max)
    dictio["persitence_from_beginning_max"].append(persitence_from_beginning_max)
    dictio["mean_layer"].append(mean_layer)
    dictio["std_layer"].append(std_layer)
    dictio["median_layer"].append(median_layer)
    dictio["mode_layer"].append(mode_layer)
    dictio["touch_boundary"].append(touch_boundary)
    dictio["polarity_mean_field"].append(mean_field)
    dictio["polarity_mean_field_size"].append(np.linalg.norm(mean_field))
    dictio["min_mean_curvature"].append(min_mean_curvature)
    dictio["max_mean_curvature"].append(max_mean_curvature)
    dictio["min_median_curvature"].append(min_median_curvature)
    dictio["max_median_curvature"].append(max_median_curvature)
    dictio["mean_curvature_mean"].append(mean_curvature_mean)
    dictio["median_curvature_mean"].append(median_curvature_mean)
    dictio["std_mean_curvature"].append(std_mean_curvature)
    dictio["std_median_curvature"].append(std_median_curvature)
    dictio["negative_mean_curvature_ratio"].append(negative_mean_curvature_ratio)
    dictio["under005_mean_curvature_ratio"].append(under005_mean_curvature_ratio)
    dictio["under01_mean_curvature_ratio"].append(under01_mean_curvature_ratio)
    dictio["under02_mean_curvature_ratio"].append(under02_mean_curvature_ratio)
    dictio["under03_mean_curvature_ratio"].append(under03_mean_curvature_ratio)
    dictio["notnan_ratio_mean"].append(notnan_ratio_mean)


df = pd.DataFrame(dictio)
df.to_csv(
    "../feature_analysis/interval=" + str(interval) + "_" + id_ + type_ + ".txt",
    index=False,
)
