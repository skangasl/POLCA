import torch
import numpy as np
import sklearn
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import MDS
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd
from joblib import parallel_backend
from scipy import spatial, stats
import networkx as nx

import seaborn as sns

sns.set_context("talk")

plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'font.size': 25}
plt.rcParams.update(params)

def compute_aic_bic(X, y, aic=True):
    k = len(set(y))
    A = np.nan_to_num(-1 * spatial.distance.cdist(X, X, metric="cosine") + 2., nan=0.0, posinf=1.0, neginf=-1.0)
    G = nx.from_numpy_array(np.matrix(A), create_using=nx.DiGraph)
    y = np.squeeze(y)
    clusters_list = [np.where(y == i)[0].tolist() for i in range(k)]
    score = min(nx.conductance(G, cluster_i, weight='weight')
            for cluster_i in clusters_list)
    return score

def concat_array_from_files(file_list, save_str = None):
    final_lst = []
    for index_range in file_list:
        curr_lst = []
        for checkpoint_file in index_range:
            curr_arr = np.load(checkpoint_file)
            curr_arr = curr_arr.reshape((len(curr_arr), 1))
            curr_lst.append(curr_arr)
        final_lst.append(np.concatenate(curr_lst, axis=1))
    new_arr = np.concatenate(final_lst, axis=0)
    if save_str:
        np.save(save_str, new_arr)
    return new_arr

def get_window_arr(traj_arr, window_size=20, normalized=True, include_values=True):
    if include_values:
        to_return = np.zeros((traj_arr.shape[0], traj_arr.shape[1] + max(traj_arr.shape[1] + 1 - window_size, 1)))
        to_return[:, :traj_arr.shape[1]] = traj_arr
        start_idx = traj_arr.shape[1]
    else:
        to_return = np.zeros((traj_arr.shape[0], max(traj_arr.shape[1] + 1 - window_size, 1)))
        start_idx = 0

    if traj_arr.shape[1] >= window_size:
        norm_arr = np.std(traj_arr, axis=1) + 1e-8
    else:
        norm_arr = np.ones((1, 1))

    for x in range(max(traj_arr.shape[1] + 1 - window_size, 1)):
        to_return[:, start_idx + x] = np.std(traj_arr[:, x:x+window_size], axis=1)
        if normalized:
            to_return[:, start_idx + x] = to_return[:, traj_arr.shape[1] + x]/norm_arr
    return to_return

def get_vel_acc(traj_arr):
    to_return = np.zeros((traj_arr.shape[0], traj_arr.shape[1] * 2 - 1))
    to_return[:, :traj_arr.shape[1]] = traj_arr
    for x in range(0, traj_arr.shape[1]-1):
        to_return[:, traj_arr.shape[1] + x] = traj_arr[:, x + 1] - traj_arr[:, x]
    return to_return


def get_spectral_clusters(traj_arr, k=10, correlation=False, save_str = None, cluster_type="spectral", first_pts = False, inst_timing=-1, normalized=True, include_values=True, min_n_pts = -1, min_samples=-1, metric="euclidean"):
    if first_pts:
        traj_arr = traj_arr[:, :1000]
    if inst_timing > 0:
        traj_arr = get_window_arr(traj_arr, window_size=inst_timing, normalized=normalized, include_values=include_values)
    

    if normalized:
        traj_arr = (traj_arr - np.mean(traj_arr, axis=0)) / (np.std(traj_arr, axis=0))

    with parallel_backend('threading', n_jobs=-1):
        if cluster_type == "spectral":
            if correlation:
                corr_mat = None
                # print(metric)
                if metric == "cosine":
                    corr_mat = np.nan_to_num(1 + (1-spatial.distance.cdist(traj_arr, traj_arr, metric="cosine")), nan=0.0, posinf=1.0, neginf=-1.0) / 2.
                elif metric == "correlation":
                    corr_mat = np.abs(stats.spearmanr(traj_arr, axis=1).statistic)
                # print(corr_mat)
                print(corr_mat.shape)
                clusters = SpectralClustering(n_clusters=k, assign_labels="kmeans", affinity="precomputed", eigen_solver="amg").fit(corr_mat).labels_ 
            else:
                clusters = SpectralClustering(n_clusters=k, n_init=100, assign_labels="kmeans", eigen_solver="amg", affinity="nearest_neighbors").fit(traj_arr).labels_
        elif cluster_type == "spectral_qr":
            if correlation:
                clusters = SpectralClustering(n_clusters=k,  n_components=30, assign_labels="cluster_qr", affinity="precomputed").fit(np.nan_to_num(np.abs(1-spatial.distance.cdist(traj_arr, traj_arr, metric="cosine")), nan=0.0, posinf=1.0, neginf=-1.0)).labels_
            else:
                clusters = SpectralClustering(n_clusters=k,  n_components=30, assign_labels="cluster_qr").fit(traj_arr).labels_
        elif cluster_type == "kmeans":
            # if k < 5:
            if correlation:
                clusters = KMeans(n_clusters=k, n_init='auto').fit(np.nan_to_num(np.abs(1-spatial.distance.cdist(traj_arr, traj_arr, metric="cosine")))).labels_
            else:
                clusters = KMeans(n_clusters=k, n_init='auto').fit(traj_arr).labels_
        elif cluster_type == "kmeans_scaled":
            scaler = StandardScaler()
            clusters = KMeans(n_clusters=k, n_init="auto").fit(scaler.fit_transform(traj_arr)).labels_
        elif cluster_type == "em":
            clusters = GaussianMixture(n_components=k).fit_predict(traj_arr)
        elif cluster_type == "agg":
            clusters = AgglomerativeClustering(n_clusters=k).fit(traj_arr).labels_
        elif "HDBSCAN" in cluster_type:
            min_clust_size = int(traj_arr.shape[0]//15)
            if min_n_pts > 0:
                min_clust_size = min_n_pts

            min_samp = min_clust_size
            if min_samples > 0:
                min_samp = min_samples
            if normalized:
                traj_arr = traj_arr - np.mean(traj_arr, axis=1, keepdims=True)

            if cluster_type == "HDBSCAN":
                clusters = HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samp, metric=metric).fit(traj_arr).labels_ + 1
            else:
                clusters = HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samp, cluster_selection_method="leaf", metric=metric).fit(traj_arr).labels_ + 1
    if save_str:
        np.save(save_str, clusters)
    return clusters

def compute_means_by_cluster(traj_arr, clusters, k=10, save_str=None, magnitude=False):
    means = {}
    stds = {}
    for i in range(k):
        tmp = traj_arr[np.where(clusters == i)]
        if len(tmp) == 0:
            means[i] = None
            stds[i] = None
        if magnitude:
            means[i] = np.mean(np.abs(tmp), axis=0)
            stds[i] = np.std(np.abs(tmp), axis=0)
        else:
            means[i] = np.mean(tmp, axis=0)
            stds[i] = np.std(tmp, axis=0)
    if save_str:
        np.save(save_str, means)
    return means, stds

def plot_clusters(traj_arr, clusters, model_name, dim, out_file, k=10, plot_type="png", shuffled=None, log_scale=False, min_max=None, alpha=0.2, xlabs=None, title_app = "", all_alphas = None, ignore_idx=-10, sample_idx=-1):
    if xlabs is None:
        checkpoint_idxs = list(range(traj_arr.shape[1]))
    else:
        checkpoint_idxs = xlabs
    if k <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif k <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, k))

    seen_clusters = set()
    plt.figure()
    for i in range(traj_arr.shape[0]):
        if all_alphas is not None:
            alpha = all_alphas[i]
        if clusters[i] != ignore_idx:
            if clusters[i] in seen_clusters:
                plt.plot(checkpoint_idxs, traj_arr[i], color=colors[clusters[i]], alpha=alpha)
            else:
                plt.plot(checkpoint_idxs, traj_arr[i], color=colors[clusters[i]], alpha=alpha, label=clusters[i])
                seen_clusters.add(clusters[i])

    if sample_idx >= 0:
        plt.axvline(x=sample_idx)
    plt.xlabel("Iteration")
    y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
    plt.ylabel(f"{y_str}")
    if log_scale:
        plt.yscale("log")
    if min_max:
        plt.ylim(min_max)
    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.savefig(out_file, format=plot_type)
    return plt.gca().get_ylim()

def plot_mean_clusters(mean_clusters, std_clusters, model_name, dim, out_file, k=10, plot_type="png", log_scale=False, min_max=None, xlabs=None, ignore_idx=-10, sample_idx=-1):
    checkpoint_idxs = list(range(len(mean_clusters[0])))
    if xlabs is not None:
        checkpoint_idxs = xlabs
    if k <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif k <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, k))
    plt.figure()
    for c in mean_clusters:
        if mean_clusters[c] is not None and c != ignore_idx:
            plt.plot(checkpoint_idxs, mean_clusters[c], color=colors[c], label=c, alpha=0.7)
            if std_clusters is not None:
                plt.fill_between(checkpoint_idxs, mean_clusters[c]-std_clusters[c], mean_clusters[c]+std_clusters[c], color=colors[c], alpha=0.2)
    if sample_idx >= 0:
        plt.axvline(x=sample_idx, color='black', alpha=0.5)
    plt.legend(title="Cluster")
    # plt.title(f"{dim} Trajectories")
    if log_scale:
        plt.yscale("log")
    if min_max:
        plt.ylim(min_max[0], min_max[1])
    plt.xlabel("Iteration")
    y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
    plt.ylabel(f"{y_str}")
    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.savefig(out_file, format=plot_type)
    return plt.gca().get_ylim()

def plot_with_stds(traj_arr, model_name, dim, out_file, xlabs=None, mean_traj_arr=None, plot_type="png", log_scale=False):
    checkpoint_idxs = list(range(traj_arr.shape[1]))
    if xlabs is not None:
        checkpoint_idxs = xlabs
    k = traj_arr.shape[2]
    if k <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif k <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, k))
    for c in range(k):
        plt.figure()
        for x in range(traj_arr.shape[0]):
            plt.plot(checkpoint_idxs, traj_arr[x, :, c], color=colors[c], alpha=0.2)

        # plt.title(f"{dim} Trajectories for 3 Digit Addition Cluster {c}")
        if log_scale:
            plt.yscale("log")
        plt.xlabel("Iteration")
        y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
        plt.ylabel(f"{y_str}")
        plt.tight_layout()
        plt.locator_params(nbins=3)
        plt.savefig(f"{out_file}_{c}.png", format=plot_type)
    return

def plot_cluster_histogram(clusters, model_name, dim, out_file, log_scale = False, plot_type="png", bar_labels=None, ignore_idx = -1):
    plt.figure()
    labels, counts = np.unique(clusters, return_counts=True)
    for label in labels:
        if label == ignore_idx:
            counts = np.delete(counts, np.where(labels == label))
            labels = np.delete(labels, np.where(labels == label))

    if len(bar_labels) > len(labels):
        labels = [0] + list(labels)
        counts = [0] + list(counts)
    bars = plt.bar(labels, counts, align='center')
    if log_scale:
        plt.yscale('log')
    plt.gca().set_xticks(labels)

    plt.xlabel("Cluster")
    plt.ylabel(f"Number of Points in Cluster")
    if bar_labels is not None:
        plt.bar_label(bars, labels=bar_labels, label_type="center") 
    plt.tight_layout()
    plt.locator_params(nbins=3, axis='y')
    plt.savefig(out_file, format=plot_type)
    return

def plot_carries_histogram(carries_dcts, model_name, dim, out_file, plot_type="png"):
    d = {}
    for k in carries_dcts[0].keys():
        if k != "Mean carries":
            d[k] = [d[k] for d in carries_dcts]
    
    width = 1.0/(len(d.keys()) + 1)  # the width of the bars
    multiplier = 0

    x = np.arange(len(carries_dcts))
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in d.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)

        multiplier += 1

    ax.set_xticks(x)

    ax.set_xlabel("Cluster")
    ax.legend(ncols=2)
    ax.set_ylim(0, 1)
    plt.locator_params(nbins=3, axis='y')
    plt.savefig(out_file, format=plot_type)
    return

def plot_accs_histogram(accs_dcts, model_name, dim, out_file, plot_type="png"):

    d = {}
    for k in range(len(accs_dcts[0])):
        d[k] = [accs_dcts[d][k] for d in accs_dcts]
    
    width = 1.0/(len(d.keys()) + 1)  # the width of the bars
    multiplier = 0

    x = np.arange(len(accs_dcts.keys()))
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in d.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)

        multiplier += 1

    ax.set_xticks(x)

    ax.set_xlabel("Cluster")
    ax.legend(ncols=2)
    ax.set_ylim(0, 1)
    plt.locator_params(nbins=3, axis='y')
    plt.savefig(out_file, format=plot_type)
    return

def plot_cat_histogram(clusters, test_arr, model_name, dim, out_file, cats = None, plot_type="png", ignore_idx=-1):
    labels = np.unique(clusters, return_counts=False)
    test_categories = np.unique(test_arr, return_counts=False)
    labels = np.array([l for l in labels if l != ignore_idx])

    width = 0.2

    x = labels
    fig, ax = plt.subplots(layout='constrained')

    attribute_dct = {"Carry": {"1000s": [], "100s": [], "10s": [], "1s": [0.] * len(labels)}, "No carry": {"1000s": [], "100s": [], "10s": [], "1s": []}}
    if cats is None:
        cats = {d:d for d in test_categories}
    homs = {l: [0.0, 0.0] for l in labels}
    for l in labels:
        curr_test = test_arr[clusters == l]
        for c in cats:
            split_idx = cats[c].rfind(' ')
            curr_carry = cats[c][:split_idx]
            curr_place = cats[c][split_idx + 1:]
            curr_arr = curr_test == c
            attribute_dct[curr_carry][curr_place].append(sum(curr_arr))
            if curr_carry == "Carry":
                homs[l][1] += sum(curr_arr)
            else:
                homs[l][0] += sum(curr_arr)

    bottom = {}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    alphas = [1.0, 0.6]

    for i, carry_str in enumerate(["Carry", "No carry"]):
        for j, digit_str in enumerate(attribute_dct[carry_str]):
            if digit_str not in bottom:
                bottom[digit_str] = np.zeros(len(labels))
            
            offset = width * j - 0.3
            p = ax.bar(x + offset, attribute_dct[carry_str][digit_str], width, color=colors[j], alpha=alphas[i], edgecolor="black", bottom=bottom[digit_str])
            bottom[digit_str] += attribute_dct[carry_str][digit_str]


    ax.set_xticks(x)
    ax.set_xlabel("Cluster")
    digits = [k for k in attribute_dct["No carry"]]
    color_patches = [mpatches.Patch(color=c, label=digits[i]) for i, c in enumerate(colors)]
    legend1 = ax.legend(handles=color_patches, title="Digit skill", loc=(1.04, 0.5))

    # Legend for transparency
    carries = ["Carry", "No carry"]
    alpha_patches = [
        mpatches.Patch(color='gray', alpha=a, label=carries[i]) for i, a in enumerate(alphas)
    ]
    legend2 = ax.legend(handles=alpha_patches, title="Carrying skill", loc=(1.04, 0))

    # Add the first legend back manually
    ax.add_artist(legend1)
    # ax.set_ylim(0, 1)
    plt.locator_params(nbins=3, axis='y')
    plt.savefig(out_file, format=plot_type)
    return

def plot_scatter(x, y, save_str):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("POLCA Vector Number")
    plt.ylabel(f"Target Number")
    plt.locator_params(nbins=3)
    plt.savefig(save_str, format="png")
    return

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def unshuffle(l, perm):
    res = [None] * len(l)
    for i, j in enumerate(perm):
        res[j] = l[i]
    return res 

def plot_closest_centroid(trajectories, centroids, model_name, dim, out_file, xlabs = None, k=5, plot_type="png", ignore_idx=-10, sample_idx=-1, metric="minkowski"):
    out_dct = {}
    for key in centroids:
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric=metric)
        neigh.fit(trajectories)
        loc_arr = neigh.kneighbors(centroids[key].reshape(1, -1), return_distance=False)[0]
        out_dct[key] = trajectories[loc_arr, :]

    if xlabs is not None:
        checkpoint_idxs = xlabs
    else:
        checkpoint_idxs = list(range(len(centroids[0])))
    n_c = len(centroids.keys())
    if n_c <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif n_c <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, n_c))

    plt.figure()
    for c in centroids:
        i = 0
        if c != ignore_idx:
            for vec in out_dct[c]:
                if i == 0:
                    plt.plot(checkpoint_idxs, vec, color=colors[c], label=c, alpha=0.5)
                else:
                    plt.plot(checkpoint_idxs, vec, color=colors[c], alpha=0.5)
                i += 1
    if sample_idx >= 0:
        plt.axvline(x=sample_idx)
    plt.legend(title="Cluster")
    plt.xlabel("Iteration")
    y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
    plt.ylabel(f"{y_str}")
    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.savefig(out_file, format=plot_type)
    return

def get_cluster_centers(documents, labels):
    centers = {}
    for label in np.unique(labels):
        centers[label] = np.mean(documents[labels == label, :], axis=0)
    return centers

def to_int(a):
    return int("".join(map(str, a)))

def compute_homogeneity_completeness(clusters, cat_df, cat_lst, save_str=None):
    out_dct = {"metric": ["homogeneity", "completeness", "v_measure"]}
    for cat in cat_lst:
        hom_cat_score = sklearn.metrics.homogeneity_score(np.array(cat_df[cat]), clusters)
        comp_cat_score = sklearn.metrics.completeness_score(np.array(cat_df[cat]), clusters)
        v_measure_score = sklearn.metrics.v_measure_score(np.array(cat_df[cat]), clusters)
        out_dct[cat] = [hom_cat_score, comp_cat_score, v_measure_score]
    out_df = pd.DataFrame(out_dct)
    if save_str is not None:
        out_df.to_csv(save_str)
    return out_df

def compute_label_homogeneity(trajectories, clusters, cat_df, cat_lst, save_str=None, ignore_idx=-1):
    label_lst = []
    hom_lst = []
    clust_lst = []
    max_means = []
    mean_maxs = []
    maxs = []
    contains_carry = False
    max_carry_hom = 0.0


    for label in np.unique(clusters):
        if label != ignore_idx:
            curr_label = ""
            curr_hom = 0.0
            curr_idxs = clusters == label
            curr_df = cat_df.iloc[curr_idxs]
            for cat in cat_lst:
                if cat == "carry_new":
                    cat_labels = {0: "No carry", 1: "Carry to digit"}
                else:
                    cat_labels = {0: "1000s place", 1: "100s place", 2: "10s place", 3: "1s place"}

                cat_vals = np.array(curr_df[cat])
                unique, counts = np.unique(cat_vals, return_counts=True)
                counts = counts / len(cat_vals)

                print(cat, label, counts)

                for i, count in enumerate(counts):
                    if cat_labels[unique[i]] == "Carry to digit":
                        max_carry_hom = max(max_carry_hom, count)
                    if count > 0.85:
                        if cat_labels[unique[i]] == "Carry to digit":
                            contains_carry = True
                        if curr_label != "":
                            curr_label = f"{curr_label}\n{cat_labels[unique[i]]}"
                        else:
                            curr_label = cat_labels[unique[i]]
                    curr_hom = max(curr_hom, count)
                if cat == "digit" and "place" not in curr_label:
                    for i, count in enumerate(counts):
                        for j, count2 in enumerate(counts[i+1:]):
                            if count + count2 > 0.85:
                                curr_label = f"{cat_labels[unique[i]]} \&\n{cat_labels[unique[i+1+j]]}"
                        if "place" not in curr_label:
                            if (count < 0.15 and len(counts) > 3) or len(counts) == 3:
                                labeled = False
                                for k, count3 in enumerate(counts):
                                    if k != i or len(counts) == 3:
                                        if not labeled:
                                            curr_label = f"{cat_labels[unique[k]]} \&\n"
                                            labeled = True
                                        else:
                                            curr_label += f"{cat_labels[unique[k]]} \&\n"

                                curr_label = curr_label[:-3]                                
                            else:
                                curr_label = "All digits"
                if curr_label == "":
                    print(unique, counts)
            
            clust_lst.append(label)
            label_lst.append(curr_label)
            hom_lst.append(curr_hom)

            curr_trajs = trajectories[curr_idxs]
            mean_maxs.append(np.mean(np.max(np.abs(curr_trajs), axis=1)))
            max_means.append(np.max(np.mean(curr_trajs, axis=1)))
            maxs.append(np.max(np.abs(curr_trajs)))
        elif len(np.unique(clusters)) == 1:
            cat_vals = np.array(cat_df["carry_new"])
            unique, counts = np.unique(cat_vals, return_counts=True)
            counts = counts / len(cat_vals)
            max_carry_hom = counts[1]
            return sum(counts)/2, ["Outliers"], False, max_carry_hom
    
    save_dict = {"Cluster": clust_lst, "Label": label_lst, "Homogeneity": hom_lst,
                "Mean maximum": mean_maxs, "Maximum mean": max_means, "Maximum": maxs}
    if save_str is not None:
        pd.DataFrame(save_dict).to_csv(save_str)
    if ignore_idx == 0:
        return sum(hom_lst)/len(hom_lst), ["Outliers"] + label_lst, contains_carry, max_carry_hom
    else:
        return sum(hom_lst)/len(hom_lst), label_lst, contains_carry, max_carry_hom

def main():

    vec_lim = 5
    n_eigs=1
    second_order = True
    step_size = 20
    n_layer = 3
    next_topk = 0
    seed=42
    use_change=True
    in_str = ""
    if n_layer > 2:
        in_str += f"_{n_layer}_layer"
    in_str += "_full_hessian"
    vec_comp = "svd_" if "svd" in in_str else ""
    second_order_str = "_second_order" if second_order else ""
    if next_topk is not None:
        in_str += f"_next_topk_{next_topk}"
    if step_size != 5:
        in_str += f"_step_size_{step_size}"
    seed_str = f"_seed_{seed}" 
    out_seed_str = f"_{seed}" if not (seed == 42 and second_order and not "hessian" in in_str) else ""
    if use_change:
        out_seed_str += "_change"
    in_dir = f"addition_task_lca{in_str}{seed_str}_{n_eigs}"
    out_dir = f"results/add_task_lca_test{in_str}{second_order_str}{out_seed_str}_{n_eigs}"
    cluster_type = "HDBSCAN_leaf"
    ignore_idx = 0 
    start_min_n_pts = 300 
    min_samples = 1
    plot_type = "png"
    model_name=f"{n_layer}-layer"
    correlation=True
    metric="cosine"
    first_1000=False
    filter_top10=False
    window_size=-1
    inst_timing=window_size > 0
    permute = False
    normalized=True
    include_values=True 
    absolute_value=False
    
    supp_str = "_first" if first_1000 else ""
    supp_str = supp_str + "_corr" if correlation else ""
    supp_str = supp_str + f"_window{window_size}" if inst_timing else ""
    supp_str = supp_str + ("_no_normalized" if not normalized else "")
    supp_str = supp_str + ("_no_values" if not include_values else "")
    supp_str = supp_str + f"_min_size_{start_min_n_pts}" if start_min_n_pts > 0 and "HDBSCAN" in cluster_type else ""
    supp_str = supp_str + f"_min_samples_{min_samples}" if min_samples > 0 and "HDBSCAN" in cluster_type else ""
    supp_str = supp_str + f"_{metric}_distance" if metric != "euclidean" else ""
    supp_str = supp_str + f"_abs" if absolute_value else ""
    n_epoch = 10000
    n_ex = 1250
    n_digits = 4
    batch_size = 64
    n_iter = n_epoch // step_size
    lca_step_size = step_size
    to_clust = [3] 
    dims = ["lca"] 
    print("Starting...")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # digit categories: 0, 1, 2, 3
    # carry categories: 0: no carries, 1: carry to not from (not possible for digit 3), 2: carry from not to (not possible for digit 0), 3: carry to and from
    # number: ground truth number
    test_df_file = f"{out_dir}/test_df.csv"
    print(test_df_file)
    test_lst = ["digit", "carry_new"]#, "number"]
    if not os.path.exists(test_df_file):
        test_dct = {n: [] for n in test_lst}
        test_dct["number"] = []
        test_tokens = np.load(f'{in_dir}/test_tokens.npy')
        for t in test_tokens:
            ans = t[-n_digits:]
            carries = [0]
            carry_cats = []
            for i, tok in enumerate(ans):
                test_dct["digit"].append(i)
                test_dct["number"].append(tok)
                
                prev_carry = carries[-1]
                if i < n_digits-1:
                    digit_sum = (t[n_digits-2-i]+t[2*(n_digits-1)-i]+prev_carry)
                    carry = int(digit_sum>=10)
                else:
                    carry = 0
                carries.append(carry)

                carry_cat = 0
                if prev_carry == 1:
                    carry_cat = 1
                    if carry == 1:
                        carry_cat = 3
                elif carry == 1:
                    carry_cat = 2
                carry_cats.append(carry_cat)

            test_dct["carry_new"].extend([1 if x == 1 or x == 3 else 0 for x in carry_cats[::-1]])
        
        test_df = pd.DataFrame(test_dct)
        test_df.to_csv(test_df_file)
    else:
        test_df = pd.read_csv(test_df_file)
    test_lst.append("carry_digit")
    test_df["carry_digit"] = np.array(list(test_df["carry_new"])) * 4 + np.array(list(test_df["digit"]))
    
    hom_sum = 0
    tot_n_hom = 0
    for dim in dims:
        if dim == "loss":
            traj_out_file = f"{out_dir}/all_loss_trajectories.npy"
            if not os.path.exists(traj_out_file):
                print("computing loss trajectory array")
                losses = np.zeros((n_ex, n_epoch))
                per_digit_losses = np.zeros((n_ex, n_epoch, n_digits))
                for i in range(n_epoch):
                    losses[:, i] = np.load(f'{in_dir}/{dim}_epoch_{i}.npy')
                    per_digit_losses[:, i, :] = np.load(f'{in_dir}/per_digit_{dim}_epoch_{i}.npy')
                np.save(traj_out_file, per_digit_losses)
            else:
                per_digit_losses = np.load(traj_out_file)

            plot_with_stds(per_digit_losses, model_name, dim, f"{out_dir}/{dim}", np.mean(per_digit_losses, axis=2), plot_type)

            loss_diffs = np.zeros((per_digit_losses.shape[0], per_digit_losses.shape[1]-1, per_digit_losses.shape[2]))
            for i in range(0, n_epoch-1):
                loss_diffs[:, i, :] = per_digit_losses[:, i+1, :] - per_digit_losses[:, i, :]
            
            plot_with_stds(loss_diffs, model_name, dim, f"{out_dir}/{dim}_diffs", np.mean(loss_diffs, axis=2), plot_type)

            loss_accs = np.zeros((per_digit_losses.shape[0], per_digit_losses.shape[1]-2, per_digit_losses.shape[2]))
            for i in range(0, n_epoch-2):
                loss_accs[:, i, :] = loss_diffs[:, i+1, :] - loss_diffs[:, i, :]
            
            plot_with_stds(loss_accs, model_name, dim, f"{out_dir}/{dim}_accs", np.mean(loss_accs, axis=2), plot_type)

            to_cluster = {"Loss": per_digit_losses, "Loss Differences": loss_diffs, "Loss Accelerations": loss_accs}
        elif dim == "logits":
            traj_out_file_mag = f"{out_dir}/all_{dim}_norm_trajectories.npy"
            traj_out_file_ang = f"{out_dir}/all_{dim}_cossim_trajectories.npy"
            if not os.path.exists(traj_out_file_ang):
                print(f"computing {dim} trajectory array")
                if dim == "logits":
                    norm_arr = np.zeros((n_ex, n_epoch, n_digits))
                    ang_arr = np.zeros((n_ex, n_epoch, n_digits))
                else:
                    norm_arr = np.zeros((n_ex, n_epoch))
                    ang_arr = np.zeros((n_ex, n_epoch))
                for i in range(1, n_epoch):
                    norm_arr[:, i] = np.load(f'{in_dir}/{dim}_diff_norm_epoch_{i}.npy')
                    ang_arr[:, i] = np.load(f'{in_dir}/{dim}_diff_cossim_epoch_{i}.npy')
                
                np.save(traj_out_file_mag, norm_arr)
                np.save(traj_out_file_ang, ang_arr)
            else:
                norm_arr = np.load(traj_out_file_mag)
                ang_arr = np.load(traj_out_file_ang)
            
            if dim == "logits":
                plot_with_stds(norm_arr, model_name, f"{dim} Norm", f"{out_dir}/{dim}_norm", plot_type=plot_type)
                plot_with_stds(ang_arr, model_name, f"{dim} Cosine Similarity", f"{out_dir}/{dim}_cossim", plot_type=plot_type, log_scale=True)
            
            to_cluster = {f"{dim} Norm": norm_arr, f"{dim} Cosine Similarity": ang_arr}

        else:
            traj_out_file = f"{out_dir}/all_loss_trajectories.npy"
            if not os.path.exists(traj_out_file):
                print("computing loss trajectory array")
                losses = np.zeros((n_ex, n_iter, n_digits))
                last_idx = 0
                for idx in range(0, n_ex//batch_size + 1):
                    for i in range(n_iter):
                        curr_fname = f'{in_dir}/{vec_comp}losses_{i}_index_{idx}.npy'
                        if os.path.exists(curr_fname):
                            try:
                                curr_losses = np.load(curr_fname)
                            except EOFError:
                                print(f"File does not exist for index {idx} iteration {i}")
                                n_iter = i
                                losses = losses[:, :i, :]
                                break
                            for d in range(curr_losses.shape[0]):
                                losses[last_idx + d//n_digits, i, d % n_digits] = curr_losses[d]
                        else:
                            print(f"File does not exist for index {idx} iteration {i}")
                            n_iter = i
                            losses = losses[:, :i, :]
                            break

                    last_idx += curr_losses.shape[0]//n_digits
                np.save(traj_out_file, losses)
            else:
                losses = np.load(traj_out_file)
                n_iter = losses.shape[1]

            plot_with_stds(losses, model_name, dim, f"{out_dir}/loss", mean_traj_arr=np.mean(losses, axis=2), plot_type=plot_type)

            loss_diffs = np.zeros((n_ex, n_iter-1, n_digits))
            for i in range(0, n_iter-1):
                loss_diffs[:, i, :] = losses[:, i+1, :] - losses[:, i, :]

            plot_with_stds(loss_diffs, model_name, dim, f"{out_dir}/loss_diffs", mean_traj_arr=np.mean(loss_diffs, axis=2), plot_type=plot_type)

            to_cluster = {
                "Loss": ([x * lca_step_size for x in list(range(n_iter))], losses) } #, 

            ## load LCA arrays
            ckpt_strs = ["ckpt"]
            lca_out_files = [f"{out_dir}/all_lca_trajectories{second_order_str}.obj"]
            if second_order:
                first_order_lca_out_files = [f"{out_dir}/all_lca_trajectories.obj"]
                if seed == 521 and n_eigs == 1:
                    ckpt_strs = ["ckpt2"]
                    lca_out_files = [f"{out_dir}/all_lca_ckpt2_trajectories{second_order_str}.obj"]
                    first_order_lca_out_files = [f"{out_dir}/all_lca_ckpt2_trajectories.obj"]
                
            if not os.path.exists(lca_out_files[0]):
                print("computing lca trajectory array")
                for ckpt_idx, ckpt_str in enumerate(ckpt_strs):
                    lca_out_file = lca_out_files[ckpt_idx]
                    lca_dct = {}
                    if second_order:
                        first_order_lca_out_file = first_order_lca_out_files[ckpt_idx]
                        first_order_lca_dct = {}
                    last_idx = 0
                    for idx in range(0, n_ex//batch_size + 1):
                        for i in range(n_iter-1): 
                            if second_order:
                                curr_lca_fname = f'{in_dir}/{vec_comp}lca_{ckpt_str}_{i}_index_{idx}.npy'
                            curr_fname = f'{in_dir}/{vec_comp}lca{second_order_str}_{ckpt_str}_{i}_index_{idx}.npy'
                            if os.path.exists(curr_fname):
                                # n x k
                                curr_lcas = np.load(curr_fname)
                                if second_order:
                                    curr_lca_first_order = np.load(curr_lca_fname)
                                for j in range(curr_lcas.shape[1]):
                                    if j not in lca_dct:
                                        if not second_order:
                                            lca_dct[j] = (np.zeros((n_ex, n_iter - 1 - i, n_digits)), i)
                                        else:
                                            lca_dct[j] = (np.zeros((n_ex, n_iter - 2 - i, n_digits)), i)
                                        if second_order:
                                            first_order_lca_dct[j] = (np.zeros((n_ex, n_iter - 1 - i, n_digits)), i)
                                    for d in range(curr_lcas.shape[0]):
                                        if not second_order:
                                            lca_dct[j][0][last_idx + d//n_digits, i - lca_dct[j][1], d % n_digits] = curr_lcas[d, j]
                                        else:
                                            first_order_lca_dct[j][0][last_idx + d//n_digits, i - first_order_lca_dct[j][1], d % n_digits] = curr_lca_first_order[d, j]
                                            lca_dct[j][0][last_idx + d//n_digits, i - lca_dct[j][1], d % n_digits] = curr_lcas[d, j] 
                                        if use_change and i - lca_dct[j][1] > 0:
                                            lca_dct[j][0][last_idx + d//n_digits, i - lca_dct[j][1], d % n_digits] += lca_dct[j][0][last_idx + d//n_digits, i - lca_dct[j][1] - 1, d % n_digits]

                        last_idx += curr_lcas.shape[0] // n_digits
                    pickle.dump(lca_dct, open(lca_out_file, "wb"))
                    if second_order:
                        pickle.dump(first_order_lca_dct, open(first_order_lca_out_file, "wb"))
            else:
                lca_dct = pickle.load(open(lca_out_files[0], "rb"))
                if second_order:
                    first_order_lca_dct = pickle.load(open(first_order_lca_out_files[0], "rb"))
            
            for j in lca_dct:
                if j < vec_lim:
                    xlabs = [x * lca_step_size for x in list(range(lca_dct[j][1], n_iter-1))]
                    if second_order:
                        xlabs = xlabs[:-1]
                    to_cluster[f"POLCA Vector {j}"] = (xlabs, np.abs(lca_dct[j][0]) if absolute_value else lca_dct[j][0])
            
            grad_proj_out_file = f"{out_dir}/all_grad_proj_trajectories.obj"
            if not os.path.exists(grad_proj_out_file):
                print("computing grad proj trajectory array")
                grad_dct = {}
                last_idx = 0
                for idx in range(0, n_ex//batch_size + 1):
                    for i in range(n_iter-1):
                        curr_fname = f'{in_dir}/{vec_comp}grad_proj_ckpt_{i}_index_{idx}.npy'
                        if seed == 521 and n_eigs == 1:
                            curr_fname = f'{in_dir}/{vec_comp}grad_proj2_ckpt_{i}_index_{idx}.npy'
                        if os.path.exists(curr_fname):
                            # n x k
                            curr_grads = np.load(curr_fname)
                            for j in range(curr_grads.shape[1]):
                                if j not in grad_dct:
                                    grad_dct[j] = (np.zeros((n_ex, n_iter - 1 - i, n_digits)), i)
                                for d in range(curr_grads.shape[0]):
                                    grad_dct[j][0][last_idx + d//n_digits, i - grad_dct[j][1], d % n_digits] = curr_grads[d, j]

                    last_idx += curr_grads.shape[0] // n_digits
                pickle.dump(grad_dct, open(grad_proj_out_file, "wb"))
            else:
                grad_dct = pickle.load(open(grad_proj_out_file, "rb"))
            
            for j in grad_dct:
                if j < vec_lim:
                    xlabs = [x * lca_step_size for x in list(range(grad_dct[j][1], n_iter-1))]
                    plot_clusters(np.expand_dims(np.mean(grad_dct[j][0] ** 2, axis=(0, 2)), axis=0), [0], model_name, f"fisher direction {j}", f"{out_dir}/fisher_{j}.png", k=1, alpha=1.0, xlabs=xlabs)

        
        loss_vals = {}
        carry_locs = []
        carry_nums = []
        loss_max_hom = {}

        loss_carry_trajectories = []
        loss_carry_numbers = []

        lca_carry_trajectories = []
        lca_carry_numbers = []
        lca_carry_alphas = []
        out_df = None
        for c in to_cluster:
            print(c)

            plt.close()
            full_to_analyze = []
            full_real_clusters = []
            xlabs = to_cluster[c][0]
            for i in range(to_cluster[c][1].shape[0]):
                # n_ex, n_epoch, n_digits
                for j in range(to_cluster[c][1].shape[2]):
                    full_to_analyze.append(to_cluster[c][1][i, :, j])
                    full_real_clusters.append(j)
            
            full_to_analyze = np.array(full_to_analyze)
            full_real_clusters = np.array(full_real_clusters)
            if c == "Loss":
                full_loss_vals = np.copy(full_to_analyze)
                loss_xlabs = np.copy(xlabs)

            for output_val in [-1]: #range(10):
                if output_val != -1:
                    analyze_idxs = np.array(test_df["number"]) == output_val
                else:
                    analyze_idxs = np.array(test_df["number"]) != -1
                if "POLCA" in c:
                    analyze_idxs = analyze_idxs * full_to_analyze[:, -1] < 0.
                if sum(analyze_idxs) < 50:
                    continue

                to_analyze = full_to_analyze[analyze_idxs]
                real_clusters = full_real_clusters[analyze_idxs]
                new_test_df = test_df.iloc[analyze_idxs]
                if output_val == 0 and c == "Loss":
                    supp_str = f"{supp_str}_{output_val}"
                elif output_val == -1 and c != "Loss":
                    supp_str = f"{supp_str[:-2]}{output_val}"
                else:
                    supp_str = f"{supp_str[:-1]}{output_val}"
                
                if c == "Loss":
                    loss_vals = np.copy(full_to_analyze) 

                top10_idxs = None
                if filter_top10 and not c == "Loss":
                    top10 = int(0.1*n_digits*to_analyze.shape[0])
                    top10_idxs = np.argpartition(np.max(to_analyze, axis=1), -top10)[-top10:]
                    to_analyze = to_analyze[top10_idxs]
                    real_clusters = real_clusters[top10_idxs]
                    new_test_df = new_test_df.iloc[top10_idxs]

                
                if "HDBSCAN" in cluster_type:
                    min_n_pts = start_min_n_pts

                    cluster_traj_out_file = f"{out_dir}/{c}_digits_{n_digits}_{cluster_type}{supp_str}.npy"
                    clusters = get_spectral_clusters(to_analyze, correlation = correlation, save_str=cluster_traj_out_file, cluster_type=cluster_type, first_pts=first_1000, inst_timing=window_size, normalized=normalized, include_values=include_values, min_n_pts=min_n_pts, min_samples=curr_min_samples, metric=metric)
                    clust_lst = [np.max(clusters)+1]
                else:
                    aics = []
                    # bics = []
                    for n_c in to_clust:
                        cluster_traj_out_file = f"{out_dir}/{c}_digits_{n_digits}_clusters_{n_c}_{cluster_type}{supp_str}.npy"
                        if not os.path.exists(cluster_traj_out_file):
                            print("computing clusters")
                            clusters = get_spectral_clusters(to_analyze, k=n_c, correlation = correlation, save_str=cluster_traj_out_file, cluster_type=cluster_type, first_pts=first_1000, inst_timing=window_size, normalized=normalized, include_values=include_values, metric=metric)
                        else:
                            clusters = np.load(cluster_traj_out_file)

                    if not cluster_type == "spectral":
                        clust_lst = to_clust
                    else:
                        clust_lst = to_clust

                iter_sampled=-1
                if "POLCA" in c or "Fisher" in c:
                    vec_idx = int(c[c.rfind(' ')+1:])
                    if vec_idx == 0:
                        iter_sampled = lca_step_size
                    else:
                        iter_sampled = (10000/50 * vec_idx) #/lca_step_size
                idx = 0
                for n_clusters in clust_lst:
                    curr_supp_str = supp_str
                    if "HDBSCAN" in cluster_type and idx == 0:
                        cluster_traj_out_file = f"{out_dir}/{c}_digits_{n_digits}_{cluster_type}{supp_str}.npy"
                    elif "HDBSCAN" in cluster_type:
                        cluster_traj_out_file = f"{out_dir}/{c}_digits_{n_digits}_{cluster_type}{supp_str}{idx+1}.npy"
                        curr_supp_str = f"top10{idx}"
                    else:
                        cluster_traj_out_file = f"{out_dir}/{c}_digits_{n_digits}_clusters_{n_clusters}_{cluster_type}{supp_str}.npy"
                    clusters = np.load(cluster_traj_out_file)

                    if n_clusters > 1:
                        plot_closest_centroid(to_analyze, get_cluster_centers(to_analyze, clusters), model_name, c, f"{out_dir}/{c}_plot_closest_centroid_{n_clusters}_{cluster_type}{curr_supp_str}.{plot_type}", xlabs=xlabs, k=10, plot_type=plot_type, ignore_idx=ignore_idx, sample_idx=iter_sampled, metric=metric)

                    print("got clusters")
                    for cat in ['carry_digit']:
                        cats = {0: "No carry 1000s", 1: "No carry 100s", 2: "No carry 10s", 3: "No carry 1s", 4: "Carry 1000s", 5: "Carry 100s", 6: "Carry 10s"} if cat == "carry_digit" else cats
                        plot_cat_histogram(clusters, np.array(list(new_test_df[cat])), f"{cat}", c, 
                                        f"{out_dir}/{c}_{cat}_histogram_{n_clusters}_clusters_{cluster_type}{curr_supp_str}.{plot_type}", 
                                        cats = cats, plot_type=plot_type, ignore_idx=ignore_idx)
                    homogeneity_score, cluster_labels, contains_carry, max_carry_hom = compute_label_homogeneity(to_analyze, clusters, new_test_df, test_lst[:-1], save_str=f"{out_dir}/{c}_homogeneity_{n_clusters}_clusters_{cluster_type}{curr_supp_str}.csv", ignore_idx=ignore_idx)
                    curr_out_df = compute_homogeneity_completeness(clusters, new_test_df, test_lst[:-1], save_str=f"{out_dir}/{c}_homogeneity_completeness_{n_clusters}_clusters_{cluster_type}{curr_supp_str}.csv")
                    n_rows = curr_out_df.shape[0]
                    curr_out_df["target_number"] = [output_val] * n_rows
                    curr_out_df["cluster_dimension"] = [c] * n_rows
                    if out_df is None:
                        out_df = curr_out_df
                    else:
                        out_df = pd.concat([out_df, curr_out_df])
                        
                    print(f"{c} {output_val} hom score: {max_carry_hom}")
                    if c == "Loss":
                        loss_max_hom[output_val] = max_carry_hom

                    if n_clusters > 1:
                        log_scale = False
                        plot_clusters(to_analyze, clusters, model_name, c, f"{out_dir}/{c}_plot_{n_clusters}_{cluster_type}{curr_supp_str}.{plot_type}", k=n_clusters, log_scale=log_scale, xlabs=xlabs, ignore_idx=ignore_idx, sample_idx=iter_sampled)
                        plot_cluster_histogram(clusters, model_name, c, f"{out_dir}/{c}_histogram_{n_clusters}_clusters_{cluster_type}{curr_supp_str}.{plot_type}", bar_labels=cluster_labels, ignore_idx=ignore_idx)

                        mean_clusters, std_clusters = compute_means_by_cluster(to_analyze, clusters, k=n_clusters, save_str=f"{out_dir}/{c}_cluster_means_{n_clusters}_{cluster_type}{curr_supp_str}.npy")
                        y_lim = plot_mean_clusters(mean_clusters, std_clusters, model_name, c, f"{out_dir}/{c}_mean_plot_{n_clusters}_{cluster_type}_clusters{curr_supp_str}.{plot_type}", k=n_clusters, log_scale=log_scale, xlabs=xlabs, ignore_idx=ignore_idx, sample_idx=iter_sampled)

                       
                        if c != "Loss":
                            loss_temp = loss_vals[analyze_idxs] 
                            if top10_idxs is not None:
                                loss_temp = loss_temp[top10_idxs]
                            plot_clusters(loss_temp, clusters, model_name, "Loss", f"{out_dir}/{c}_cluster_Loss_plot_{n_clusters}_{cluster_type}{curr_supp_str}.{plot_type}", k=n_clusters, xlabs=loss_xlabs, sample_idx=iter_sampled)
                            mean_clusters, std_clusters = compute_means_by_cluster(loss_temp, clusters, k=n_clusters, save_str=f"{out_dir}/{c}_cluster_means_{n_clusters}_{cluster_type}{curr_supp_str}.npy")
                            plot_mean_clusters(mean_clusters, std_clusters, model_name, "Loss", f"{out_dir}/{c}_cluster_Loss_mean_plot_{n_clusters}_{cluster_type}_clusters{curr_supp_str}.{plot_type}", k=n_clusters, xlabs=loss_xlabs, ignore_idx=ignore_idx, sample_idx=iter_sampled)
                            hom_sum += homogeneity_score
                            tot_n_hom += 1
                            if contains_carry:
                                vec_int = int(c[c.rfind(" ")+1:])
                                carry_locs.append(vec_int)
                                carry_nums.append(output_val)

                        for c_idx, clust in enumerate(cluster_labels):
                            if "Carry to digit" in clust:
                                if "Loss" not in c:
                                    lca_carry_trajectories.append(mean_clusters[c_idx])
                                    lca_carry_numbers.append(output_val)
                                    vec_int = int(c[c.rfind(" ")+1:])
                                    lca_carry_alphas.append(vec_int)
                                else:
                                    loss_carry_trajectories.append(mean_clusters[c_idx])
                                    loss_carry_numbers.append(output_val)

                    plt.close()
                    idx += 1
                
            for cluster_filter in test_lst:
                k = 4 if (cluster_filter == "digit" or cluster_filter == "carry") else 2
                k = 10 if cluster_filter == "number" else k
                k = 7 if (cluster_filter == "carry_digit") else k
                plot_clusters(full_to_analyze, list(test_df[cluster_filter]), model_name, c, f"{out_dir}/{c}_plot_{n_digits}_{cluster_filter}_clusters.{plot_type}", k=k, log_scale=log_scale, xlabs=xlabs)
                real_mean_clusters, real_std_clusters = compute_means_by_cluster(full_to_analyze,  np.array(list(test_df[cluster_filter])), k=k, save_str=f"{out_dir}/{c}_{cluster_filter}_cluster_means_{n_digits}.npy")
                plot_mean_clusters(real_mean_clusters, real_std_clusters, model_name, c, f"{out_dir}/{c}_mean_plot_{n_digits}_{cluster_filter}_clusters.{plot_type}", k=k, log_scale=log_scale, xlabs=xlabs)
        
        out_df.to_csv(f"{out_dir}/all_cluster_scores.csv")

        max_a = max(lca_carry_alphas)
        lca_carry_alphas = [(max_a+a+1)/(2*(max_a+1)) for a in lca_carry_alphas]
        ylim = plot_clusters(np.array(lca_carry_trajectories), np.array(lca_carry_numbers), "", "Loss", f"{out_dir}/all_lca_carry{curr_supp_str}.{plot_type}", k=10, title_app=" for Carry POLCA Clusters", xlabs=loss_xlabs, all_alphas = lca_carry_alphas)
        
        plot_clusters(np.array(loss_carry_trajectories), np.array(loss_carry_numbers), "", "Loss", f"{out_dir}/all_loss_carry{curr_supp_str}.{plot_type}", k=10, alpha=1.0, xlabs=loss_xlabs, min_max=ylim)

        mean_locs = np.array(list(test_df["carry_new"])) == 1
        filtered_losses = full_loss_vals[mean_locs]
        filtered_numbers = np.array(list(test_df["number"]))[mean_locs]
        mean_clusters, std_clusters = compute_means_by_cluster(filtered_losses, filtered_numbers, k=10)
        plot_mean_clusters(mean_clusters, None, "", "Loss", f"{out_dir}/all_real_loss_carry{curr_supp_str}.{plot_type}", k=10, xlabs=loss_xlabs)

        print(hom_sum/tot_n_hom)

        plot_scatter(carry_locs, carry_nums, f"{out_dir}/carry_locs_{cluster_type}{curr_supp_str}.{plot_type}")

    return

if __name__ == '__main__':
    main()