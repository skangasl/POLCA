import torch
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans, AgglomerativeClustering, HDBSCAN
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from joblib import parallel_backend
import pandas as pd
from transformers import PreTrainedTokenizerFast
from scipy import spatial
import networkx as nx
import random
import spacy
import seaborn as sns

sns.set_context("talk")

plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'font.size': 25}
plt.rcParams.update(params)

# We exclude the following components as we do not need them. 
nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
# Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
# Adds the English PyMUSAS rule-based tagger to the main spaCy pipeline
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)

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

def get_spectral_clusters(traj_arr, k=10, correlation=False, save_str = None, cluster_type="spectral", first_pts = False, inst_timing=-1, normalized=True, include_values=True, min_n_pts = -1, min_samples=-1, metric="cosine"):
    if first_pts:
        traj_arr = traj_arr[:, :1000]
    if inst_timing > 0:
        traj_arr = get_window_arr(traj_arr, window_size=inst_timing, normalized=normalized, include_values=include_values)

    centroids = None

    with parallel_backend('threading', n_jobs=-1):
        if cluster_type == "spectral":
            if correlation:
                clusters = SpectralClustering(n_clusters=k, assign_labels="kmeans", affinity="precomputed", eigen_solver="amg").fit(np.nan_to_num(-1 * spatial.distance.cdist(traj_arr, traj_arr, metric="cosine") + 2., nan=0.0, posinf=1.0, neginf=-1.0)).labels_ #np.nan_to_num(np.corrcoef(traj_arr))
            else:
                clusters = SpectralClustering(n_clusters=k, assign_labels="kmeans", eigen_solver="amg").fit(traj_arr).labels_
        elif cluster_type == "spectral_qr":
            if correlation:
                clusters = SpectralClustering(n_clusters=k, assign_labels="cluster_qr", affinity="precomputed").fit(np.nan_to_num(np.corrcoef(traj_arr))).labels_
            else:
                clusters = SpectralClustering(n_clusters=k, assign_labels="cluster_qr").fit(traj_arr).labels_
        elif cluster_type == "kmeans":
            if correlation:
                clusters = KMeans(n_clusters=k, n_init='auto').fit(np.nan_to_num(np.corrcoef(traj_arr))).labels_
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
            min_clust_size = int(traj_arr.shape[0]//(10)) # approximate from arithmetic setting
            if min_n_pts > 0:
                min_clust_size = min_n_pts

            min_samp = min_clust_size
            if min_samples > 0:
                min_samp = min_samples
            if normalized:
                traj_arr = traj_arr - np.mean(traj_arr, axis=1, keepdims=True)

            if cluster_type == "HDBSCAN":
                hdbscan = HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samp, metric=metric, store_centers="medoid").fit(traj_arr)#.labels_ + 1
            else:
                hdbscan = HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samp, cluster_selection_method="leaf", metric=metric, store_centers="medoid").fit(traj_arr)#.labels_ + 1
            
            clusters = hdbscan.labels_ + 1
            centroids = {i+1: hdbscan.medoids_[i, :] for i in range(len(hdbscan.medoids_))}
    if save_str:
        np.save(save_str, clusters)
    return clusters, centroids

def compute_means_by_cluster(traj_arr, clusters, k=10, save_str=None, magnitude=False, cossim=True):
    means = {}
    stds = {}
    for i in clusters:
        # if i > 0:
        tmp = traj_arr[np.where(clusters == i)]
        if magnitude:
            means[i] = np.mean(np.abs(tmp), axis=0)
            stds[i] = np.std(np.abs(tmp), axis=0)
        else:
            means[i] = np.mean(tmp, axis=0)
            stds[i] = np.std(tmp, axis=0)
    if save_str:
        np.save(save_str, means)
    return means, stds

def plot_clusters(traj_arr, clusters, model_name, dim, out_file, k=10, plot_type="png", shuffled=None, log_scale=False, min_max=None, alpha=0.2, xlabs=None, ignore_idx = -1):
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

    plt.figure()

    for i in range(traj_arr.shape[0]):
        if clusters[i] >= 0 and clusters[i] != ignore_idx:
            plt.plot(checkpoint_idxs, traj_arr[i], color=colors[clusters[i]], alpha=alpha)
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

def plot_mean_clusters(mean_clusters, std_clusters, model_name, dim, out_file, k=10, plot_type="png", log_scale=False, min_max=None, xlabs=None, curr_vec="", ignore_idx=-1):
    if xlabs is not None:
        checkpoint_idxs = xlabs
    else:
        checkpoint_idxs = list(range(len(mean_clusters[0])))
    if k <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif k <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, k))

    plt.figure()
    for c in sorted(mean_clusters):
        if c != ignore_idx:
            plt.plot(checkpoint_idxs, mean_clusters[c], color=colors[c], label=c, alpha=0.7)
            plt.fill_between(checkpoint_idxs, mean_clusters[c]-std_clusters[c], mean_clusters[c]+std_clusters[c], color=colors[c], alpha=0.2)
    plt.legend(title="Cluster", loc="upper right")
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

def plot_with_stds(traj_arr, model_name, dim, out_file, cluster_filtering, domains = None, xlabs = None, mean_traj_arr=None, plot_type="png", log_scale=False):
    plt.figure()
    if xlabs is None:
        checkpoint_idxs = list(range(traj_arr.shape[1]))
    else:
        checkpoint_idxs = xlabs
    k = cluster_filtering.shape[0]
    if k <= 10:
        colors = [plt.cm.tab10(i) for i in range(10)]
    elif k <= 20:
        colors = [plt.cm.tab20(i) for i in range(20)]
    else:
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, k))
    for c in range(k):
        for x in range(traj_arr.shape[0]):
            plt.plot(checkpoint_idxs, traj_arr[x, :], color=colors[c], alpha=0.2)

    if mean_traj_arr is None:
        curr_arr = np.mean(traj_arr, axis=0)
        curr_std = np.std(traj_arr, axis=0)
        plt.plot(checkpoint_idxs, curr_arr, color=colors[k+1], label=f"Mean Loss") 
        plt.fill_between(checkpoint_idxs, curr_arr-curr_std, curr_arr+curr_std, color=colors[k+1], alpha=0.2)
    
    plt.legend()
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Iteration")
    y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
    plt.ylabel(f"{y_str}")
    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.savefig(f"{out_file}.png", format=plot_type)
    return

def plot_cluster_histogram(clusters, model_name, dim, out_file, log_scale = False, plot_type="png", bar_labels=None):
    plt.figure()
    labels, counts = np.unique(clusters, return_counts=True)
    bars = plt.bar(labels, counts, align='center')
    if log_scale:
        plt.yscale('log')
    plt.gca().set_xticks(labels)
    plt.xlabel("Cluster")
    plt.ylabel(f"Number of Points in Cluster")
    if bar_labels is not None:
        plt.bar_label(bars, labels=[bar_labels[l] if l != 0 else "Outliers" for l in labels], rotation=90, label_type="center")
    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.savefig(out_file, format=plot_type)
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

def to_int(a):
    return int("".join(map(str, a)))

def get_tokens_to_analyze(trajectories, test_data, topk=10, criterion = "max_mean"):
    test_tokens = test_data[0]
    test_tokenizer = test_data[1]

    if criterion == "pos":
        token_lst = ['NOUN']
        top_tokens = token_lst
        top_token_names = token_lst
    else:
        if criterion == "fixed":
            token_lst = ["."]
            top_tokens = np.array([test_tokenizer.encode(t)[0] for t in token_lst])

        else:
            unique_tokens = np.unique(test_tokens)
            print(unique_tokens.shape)
            token_criteria = np.zeros(unique_tokens.shape)
            tot_valid = 0
            
            for i, token in enumerate(unique_tokens):
                curr_trajs = trajectories[test_tokens == token, :]
                if curr_trajs.shape[0] >= 30: 
                    if criterion == "mean_max":
                        token_criteria[i] = np.mean(np.max(np.abs(curr_trajs), axis=1))
                    elif criterion == "max_mean":
                        token_criteria[i] = np.max(np.mean(np.abs(curr_trajs), axis=1))
                    elif criterion == "random":
                        token_criteria[i] = random.random()
                    else:
                        # not implemented
                        pass
                    tot_valid += 1

            print(f"Total valid tokens: {tot_valid}")
            top_ind = np.argpartition(token_criteria, -topk)[-topk:]
            print(top_ind)
            top_ind = top_ind[np.argsort(token_criteria[top_ind])]
            print(top_ind)
            top_tokens = unique_tokens[top_ind]
            print(top_tokens)

        top_token_names = [test_tokenizer.decode(t) for t in top_tokens]
        print(top_token_names)
    
    return top_tokens, top_token_names

def plot_closest_centroid(trajectories, clusters, centroids, model_name, dim, out_file, test_data, k=5, plot_type="png", xlabs=None, loss_trajectories=None, loss_xlabs=None, metric='minkowski', normalized=False, ignore_idx=-1):
    out_dcts = [{}]
    if loss_trajectories is not None:
        out_dcts.append({})
    test_tokens = test_data[0]
    test_token_idxs = test_data[1][0]
    test_tokenizer = test_data[2]
    return_dct = {}
    for i, key in enumerate(centroids):
        curr_idxs = np.nonzero(clusters.flatten() == key)

        for j in range(3):
            cluster_toks = test_tokens[test_token_idxs[clusters.flatten() == key] - j]
            counts = np.bincount(cluster_toks)
            ind = np.argpartition(counts, -k)[-k:]
            top_tokens = ind[np.argsort(counts[ind])]

            print(f"Top {j} prev tokens for cluster {key}:")
            print(top_tokens)
            print(np.sort(counts[ind]))
            try:
                print([test_tokenizer.decode(t) for t in top_tokens])
            except IndexError:
                pass

        if curr_idxs[0].shape[0] > k:
            norm_trajectories = trajectories
            neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric=metric)
            neigh.fit(norm_trajectories[clusters == key, :])
            loc_arr = neigh.kneighbors(centroids[key].reshape(1, -1), return_distance=False)[0]

            locs = [curr_idxs[0][l] for l in loc_arr]
            out_dcts[0][key] = trajectories[locs, :]
            out_dcts[1][key] = loss_trajectories[locs, :]
        else:
            locs = curr_idxs[0].flatten()
            out_dcts[0][key] = trajectories[locs, :]
            out_dcts[1][key] = loss_trajectories[locs, :]



        print(f"Closest documents to centroid for cluster {key}:")
        print(locs)
        try:
            print([test_tokenizer.decode(test_tokens[test_token_idxs[l]-1]) for l in locs])
            print([test_tokenizer.decode(test_tokens[test_token_idxs[l]]) for l in locs])
            curr_centroids = [test_tokenizer.decode(test_tokens[test_token_idxs[l]-20:test_token_idxs[l]+1]) for l in locs]
            print(curr_centroids)
            return_dct[key] = curr_centroids
        except IndexError:
            print("Index Error")
            pass
    
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

    inum = 0
    for out_dct in out_dcts:
        plt.figure()
        for c in sorted(centroids):
            if c != ignore_idx:
                for i, vec in enumerate(out_dct[c]):
                    if i == 0:
                        plt.plot(checkpoint_idxs, vec, color=colors[c], label=c, alpha=0.7)
                    else:
                        plt.plot(checkpoint_idxs, vec, color=colors[c], alpha=0.7)

        plt.legend(title="Cluster")
        plt.xlabel("Iteration")
        y_str = "Projected loss" if "POLCA" in dim else "Full-rank loss"
        plt.ylabel(f"{y_str}")
        if inum > 0:
            out_file = f"{out_file[:-4]}_loss_trajectories.{plot_type}"
        plt.tight_layout()
        plt.locator_params(nbins=3)
        plt.savefig(out_file, format=plot_type)

        if loss_trajectories is not None:
            if loss_xlabs is not None:
                checkpoint_idxs = loss_xlabs
            else:
                checkpoint_idxs = list(range(len(loss_trajectories[0])))
        inum += 1
    return return_dct

def get_cluster_pos_homogeneities(trajectories, clusters, centroids, test_data, curr_token, save_str = "", threshold = 0.75, ignore_idx=0):
    test_tokens = test_data[0]
    test_token_idxs = test_data[1][0]
    test_tokenizer = test_data[2]
    
    cluster_labels = {}
    for i in np.unique(clusters):
        if i != ignore_idx:
            cluster_labels[i] = {"token": {}, "word": {}}
            init_locs = test_token_idxs[clusters.flatten() == i]

            locs = init_locs
            
            context_before = [test_tokenizer.decode(test_tokens[l-30:l+1], clean_up_tokenization_spaces=True) for l in locs]
            context_before = [s.replace('[PAD]', '[') for s in context_before]
            context_before_pos = [list(nlp(t))[-4:][::-1] for t in context_before]

            for doc in context_before_pos:
                for k2, token in enumerate(doc):
                    if -k2 not in cluster_labels[i]['word']:
                        cluster_labels[i]['word'][-k2] = {}

                    analyze_lst = [token.pos_]
                    if not token.text[0].isupper() and any(x.isupper() for x in token.text[1:]):
                        analyze_lst[0] = f"MIDCAP_{analyze_lst[0]}"

                    for tag in analyze_lst:
                        if tag not in cluster_labels[i]['word'][-k2]:
                            cluster_labels[i]['word'][-k2][tag] = 1/len(locs)
                        else:
                            cluster_labels[i]['word'][-k2][tag] += 1/len(locs)
            

        
    print(cluster_labels)
    df = pd.DataFrame(cluster_labels)
    if save_str != "":
        df.to_csv(save_str)

    return cluster_labels

def pos_homogeneities(trajectories, clusters, centroids, test_pos, curr_token, save_str = "", threshold = 0.75, ignore_idx=0):
    
    cluster_labels = {}
    for i in np.unique(clusters):
        if i != ignore_idx:
            cluster_labels[i] = {"token": {}, "word": {}}

            locs = np.where(clusters.flatten() == i)[0].tolist()

            for l in locs:
                for k2 in range(3):
                    if -k2 not in cluster_labels[i]['word']:
                        cluster_labels[i]['word'][-k2] = {}

                    analyze_lst = [test_pos[l-k2]]

                    for tag in analyze_lst:
                        if tag not in cluster_labels[i]['word'][-k2]:
                            cluster_labels[i]['word'][-k2][tag] = 1/len(locs)
                        else:
                            cluster_labels[i]['word'][-k2][tag] += 1/len(locs)
            

        
    print(cluster_labels)
    df = pd.DataFrame(cluster_labels)
    if save_str != "":
        df.to_csv(save_str)

    return cluster_labels

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
    ax.set_title(f"{dim} Cluster Membership Histogram")
    ax.set_xlabel("Cluster")
    ax.legend(ncols=2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.locator_params(axis='y', nbins=3)
    plt.savefig(out_file, format=plot_type)
    return

def get_cluster_centers(documents, labels):
    centers = {}
    for label in np.unique(labels):
        if label >-1: #!= 0:
            centers[label] = np.mean(documents[labels == label, :], axis=0)
            
    return centers

def main():
    to_clust = [2] 
    n_domains = 1
    n_eigs = 1
    tok_len = 128 
    num_docs = 100
    min_n_pts = 100 
    min_samples = 2 
    seed = 42 #975
    seed_str = f"_{seed}" if seed > 0 else ""
    # n_ex = num_docs * tok_len
    ex_str = f"_{num_docs}_hessian_step_500" #if n_ex != 5000 else ""
    second_order = True
    second_order_str = "_second_order" if second_order else ""
    lca_step_size=200
    step_size_str = f"_step_size_{lca_step_size}"
    n_subset = -1
    subset_str = f"_subset_{n_subset}" if n_subset > 0 else ""
    basis_str = "_3_layer_full_hessian"
    use_change = True


    in_dir = f"english_wikipedia_lca{basis_str}{step_size_str}{ex_str}_{n_eigs}{seed_str}"
    out_dir = f"results/wikipedia_task{basis_str}{step_size_str}{ex_str}_{n_eigs}{seed_str}"
    if use_change:
        out_dir += "_change2"
    token_path = f"wikipedia/tokenizer"
    domains = {0: "Wikipedia"}
    cluster_type = "HDBSCAN" 
    plot_type = "png"
    model_name="3-layer"
    correlation=True
    first_1000=False
    window_size=-1
    inst_timing=True if window_size > 0 else False
    permute = False
    normalized=True
    include_values= True
    absolute_value=False 
    metric="cosine"
    ignore_idx=0
    
    supp_str = "_first" if first_1000 else ""
    supp_str = supp_str + "_corr" if correlation else ""
    supp_str = supp_str + f"_window{window_size}" if inst_timing else ""
    supp_str = supp_str + f"_min_size_{min_n_pts}" if min_n_pts > 0 else ""
    supp_str = supp_str + f"_min_samples_{min_samples}" if min_samples > 0 else ""
    supp_str = supp_str + f"_abs" if absolute_value else ""
    init_supp_str = supp_str + "2"

    n_iter = 80
    batch_size = 64


    dims = ["lca"] 
    print("Starting...")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(f"{out_dir}/test_tokens.npy"):
        save_lca = False
        if os.path.exists(f'{in_dir}/lca_docs.csv'):
            test_data = pd.read_csv(f'{in_dir}/lca_docs.csv')
            save_lca = True
        else:
            test_data = pd.read_csv(f'{out_dir}/lca_docs.csv')
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=token_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        all_test_docs = list(test_data["text"])
        test_tokens = tokenizer(all_test_docs, padding = True, truncation=True, max_length = 128, return_tensors="pt").to("cpu")
        filt_idxs = list(range(tok_len))
        filt_idxs = filt_idxs[1:batch_size] + filt_idxs[batch_size+1:]
        print(filt_idxs)
        print(test_tokens.input_ids.shape)
        test_tokens = np.array(test_tokens.input_ids)[:, filt_idxs].flatten()
        print(test_tokens.shape[0])
        np.save(f"{out_dir}/test_tokens.npy", np.array(test_tokens))
        if save_lca:
            test_data.to_csv(f"{out_dir}/lca_docs.csv")
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=token_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        test_tokens = np.load(f"{out_dir}/test_tokens.npy")
        test_data = pd.read_csv(f"{out_dir}/lca_docs.csv")
    
    if n_subset > 0:
        test_data = test_data[:n_subset]
    n_ex = test_tokens.shape[0]

    labels = np.ones((n_ex, 1))
    real_clusterings = labels.T


    for dim in dims:
        
        traj_out_file = f"{out_dir}/all_loss_trajectories.npy"
        to_break = False
        if not os.path.exists(traj_out_file):
            print("computing loss trajectory array")
            losses = np.zeros((n_ex, n_iter))
            for i in range(n_iter):
                last_idx = 0
                for idx in range(0, (tok_len//batch_size) * num_docs):
                    curr_fname = f'{in_dir}/losses_{i}_index_{idx}.npy'
                    if os.path.exists(curr_fname):
                        curr_losses = np.load(curr_fname)

                        losses[last_idx:last_idx+curr_losses.shape[0], i] = curr_losses
                    else:
                        print(f"File does not exist for index {idx} iteration {i}")
                        n_iter = i
                        losses = losses[:, :i]
                        to_break=True
                        break
                    
                    last_idx += curr_losses.shape[0]

                if to_break:
                    break
            np.save(traj_out_file, losses)
        else:
            losses = np.load(traj_out_file)
            n_iter = losses.shape[1]

        plot_with_stds(losses, model_name, dim, f"{out_dir}/loss", real_clusterings, domains=domains, mean_traj_arr=None, plot_type=plot_type)

        to_cluster = {
            "Loss": ([lca_step_size * x for x in range(n_iter)], losses) } 
        ## load LCA arrays
        print(n_iter)

        lca_out_file = f"{out_dir}/all_lca_trajectories{second_order_str}.obj"
        if second_order:
            first_order_lca_out_file = f"{out_dir}/all_lca_trajectories.obj"
        if not os.path.exists(lca_out_file):
            print("computing lca trajectory array")
            lca_dct = {}
            if second_order:
                first_order_lca_dct = {}
            last_idx = 0
            for idx in range(0, (tok_len//batch_size) * num_docs):
                for i in range(n_iter-1): 
                    if second_order:
                        curr_lca_fname = f'{in_dir}/lca_ckpt_{i}_index_{idx}.npy'
                    curr_fname = f'{in_dir}/lca{second_order_str}_ckpt_{i}_index_{idx}.npy'
                    if os.path.exists(curr_fname):
                        # n x k
                        curr_lcas = np.load(curr_fname)
                        if second_order:
                            curr_lca_first_order = np.load(curr_lca_fname)
                        for j in range(curr_lcas.shape[1]):
                            if j not in lca_dct:
                                if not second_order:
                                    lca_dct[j] = (np.zeros((n_ex, n_iter - 1 - i)), i)
                                else:
                                    lca_dct[j] = (np.zeros((n_ex, n_iter - 2 - i)), i)
                                if second_order:
                                    first_order_lca_dct[j] = (np.zeros((n_ex, n_iter - 1 - i)), i)
                            for d in range(curr_lcas.shape[0]):
                                if not second_order:
                                    lca_dct[j][0][last_idx:last_idx + curr_lcas.shape[0], i - lca_dct[j][1]] = curr_lcas[d, j]
                                    if use_change and i - lca_dct[j][1] > 0:
                                        lca_dct[j][0][last_idx:last_idx + curr_lcas.shape[0], i - lca_dct[j][1]] += lca_dct[j][0][last_idx:last_idx + curr_lcas.shape[0], i - lca_dct[j][1] - 1]
                                else:
                                    first_order_lca_dct[j][0][last_idx + d, i - first_order_lca_dct[j][1]] = curr_lca_first_order[d, j]
                                    lca_dct[j][0][last_idx + d, i - lca_dct[j][1]] = curr_lcas[d, j]
                                    if use_change and i - lca_dct[j][1] > 0:
                                        lca_dct[j][0][last_idx + d, i - lca_dct[j][1]] += lca_dct[j][0][last_idx + d, i - lca_dct[j][1] - 1]
                    else:
                        print(f"No file {curr_fname}")

                last_idx += curr_lcas.shape[0]
            pickle.dump(lca_dct, open(lca_out_file, "wb"))
            pickle.dump(first_order_lca_dct, open(first_order_lca_out_file, "wb"))
        else:
            lca_dct = pickle.load(open(lca_out_file, "rb"))
            if second_order:
                first_order_lca_dct = pickle.load(open(first_order_lca_out_file, "rb"))
        
        for j in lca_dct:
            xlabs = [lca_step_size * x for x in range(lca_dct[j][1], n_iter-1)]
            if second_order:
                xlabs = xlabs[:-1]
            lca_dct[j][0][:, 0] = 0.
            to_cluster[f"POLCA Vector {j}"] = (xlabs, np.abs(lca_dct[j][0]) if absolute_value else np.array(lca_dct[j][0]))
        
        grad_proj_out_file = f"{out_dir}/all_grad_proj_trajectories.obj"
        if not os.path.exists(grad_proj_out_file):
            print("computing grad proj trajectory array")
            grad_dct = {}
            last_idx = 0
            for idx in range(0, (tok_len//batch_size) * num_docs):
                for i in range(n_iter-1):
                    curr_fname = f'{in_dir}/grad_proj_ckpt_{i}_index_{idx}.npy'
                    if os.path.exists(curr_fname):
                        # n x k
                        curr_grads = np.load(curr_fname)
                        for j in range(curr_grads.shape[1]):
                            if j not in grad_dct:
                                grad_dct[j] = (np.zeros((n_ex, n_iter - 1 - i)), i)
                            grad_dct[j][0][last_idx:last_idx + curr_grads.shape[0], i - grad_dct[j][1]] = curr_grads[:, j]

                last_idx += curr_grads.shape[0]
            pickle.dump(grad_dct, open(grad_proj_out_file, "wb"))
        else:
            grad_dct = pickle.load(open(grad_proj_out_file, "rb"))
        
        for j in grad_dct:
            xlabs = [lca_step_size * x for x in range(grad_dct[j][1], n_iter-1)]
        
            plot_clusters(np.expand_dims(np.mean(grad_dct[j][0] ** 2, axis=0), axis=0), [0], model_name, f"fisher direction {j}", f"{out_dir}/fisher_{j}.png", k=1, alpha=1.0, xlabs=xlabs)

    all_cluster_labs = {"Token": [], "POLCA Vector": [], "Cluster": [], "Property": [], "Centroids": []}
    for c in to_cluster:
        all_to_analyze = np.array(to_cluster[c][1])

        xlabs = to_cluster[c][0]
        if c == "Loss":
            all_loss_vals = np.copy(all_to_analyze)
            loss_xlabs = np.copy(np.array(xlabs))

        
        tokens_to_analyze, token_analyze_names = get_tokens_to_analyze(all_to_analyze, (test_tokens, tokenizer), topk=10, criterion = "fixed")

        for token_idx, token in enumerate(tokens_to_analyze):
            test_idxs = np.array([1] * test_tokens.shape[0], dtype=int)
            
            if c != "Loss":
                if np.mean(all_to_analyze[:, -1]) >= 0.:
                    print(f"Not using {c}")
                    break
                test_idxs = test_idxs * all_to_analyze[:, -1] < 0.
            
            if sum(test_idxs) < 50:
                break
            
            to_analyze = all_to_analyze[test_idxs == 1, :]

            loss_vals = all_loss_vals[test_idxs==1, :] 

            test_token_idxs = np.nonzero(test_idxs)
            supp_str = f"{init_supp_str}_{token_analyze_names[token_idx]}"
            centroids = None

            if "HDBSCAN" in cluster_type:
                cluster_traj_out_file = f"{out_dir}/{c}_{n_domains}_domains_{cluster_type}{supp_str}.npy"
                clusters, centroids = get_spectral_clusters(to_analyze, correlation = correlation, save_str=cluster_traj_out_file, cluster_type=cluster_type, first_pts=first_1000, inst_timing=window_size, normalized=normalized, include_values=include_values, min_n_pts=min_n_pts, min_samples=min_samples, metric=metric)
                clust_lst = [np.max(clusters) + 1]
            elif cluster_type == "top10":
                top10_ind = np.argsort(-np.std(to_analyze, axis=1))[:int(0.01*to_analyze.shape[0])]
                clusters = np.zeros(to_analyze.shape[0], dtype=int)
                clusters[top10_ind] = 1
                cluster_traj_out_file =  f"{out_dir}/{c}_{n_domains}_domains_{cluster_type}{supp_str}.npy"
                np.save(cluster_traj_out_file, clusters)
                clust_lst = [np.max(clusters) + 1]
            else:
                aics = []
                bics = []
                for n_c in to_clust:
                    cluster_traj_out_file = f"{out_dir}/{c}_{n_domains}_domains_{cluster_type}{supp_str}.npy"
                    if not os.path.exists(cluster_traj_out_file):
                        print("computing clusters")
                        clusters, _ = get_spectral_clusters(to_analyze, k=n_c, correlation = correlation, save_str=cluster_traj_out_file, cluster_type=cluster_type, first_pts=first_1000, inst_timing=window_size, metric=metric)
                    else:
                        clusters = np.load(cluster_traj_out_file)
                    aics.append(compute_aic_bic(to_analyze, clusters, aic=True))

                plt.figure()
                plt.plot(to_clust, aics, label="Conductance")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Conductance")
                plt.tight_layout()
                plt.savefig(f"{out_dir}/{c}_{n_domains}_domains_conductance_{cluster_type}{supp_str}.{plot_type}", format=plot_type)
                clust_lst = [to_clust[np.argmax(np.array(aics))]]

                clusters = np.load(f"{out_dir}/{c}_{n_domains}_domains_{cluster_type}{supp_str}.npy")
                n_clusters = n_c + 1

                print(f"Top 30 tokens per cluster for {c}:")
                if centroids is None:
                    centroids = get_cluster_centers(to_analyze, clusters)
                elif 0 not in centroids:
                    centroids[0] = np.mean(to_analyze[clusters == 0, :], axis=0)

                cluster_centroids = plot_closest_centroid(to_analyze, clusters, centroids, model_name, c, f"{out_dir}/{c}_plot_closest_centroid_{n_clusters}_{cluster_type}{supp_str}.{plot_type}", 
                    (test_tokens, test_token_idxs, tokenizer), k=10, plot_type=plot_type, xlabs=xlabs, loss_trajectories=loss_vals, loss_xlabs=loss_xlabs, normalized=normalized, ignore_idx=ignore_idx)#, metric=metric) #loss_vals)

                curr_tok = token_analyze_names[token_idx]
                cluster_labels = get_cluster_pos_homogeneities(to_analyze, clusters, centroids, (test_tokens, test_token_idxs, tokenizer), curr_tok, save_str=f"{out_dir}/{c}_homogeneities_{n_clusters}_{cluster_type}{supp_str}.csv", ignore_idx=0 if "HDBSCAN" in cluster_type else -1)
                

                for temp_clust in cluster_labels:
                    all_cluster_labs["Token"] += [curr_tok]
                    all_cluster_labs["POLCA Vector"] += [c]
                    all_cluster_labs["Cluster"] += [temp_clust]
                    all_cluster_labs["Property"] += [cluster_labels[temp_clust]]
                    all_cluster_labs["Centroids"] += [cluster_centroids[temp_clust]]

                print("got clusters")

                
                log_scale = False 
                plot_clusters(to_analyze, clusters, model_name, c, f"{out_dir}/{c}_plot_{n_clusters}_{cluster_type}{supp_str}.{plot_type}", k=n_clusters, log_scale=log_scale, xlabs=xlabs, ignore_idx=ignore_idx)

                plot_cluster_histogram(clusters, model_name, c, f"{out_dir}/{c}_histogram_{n_clusters}_clusters_{cluster_type}{supp_str}.{plot_type}")

                mean_clusters, std_clusters = compute_means_by_cluster(to_analyze, clusters, k=n_clusters, save_str=f"{out_dir}/{c}_cluster_means_{n_clusters}_{cluster_type}{supp_str}.npy")
                y_lim = plot_mean_clusters(mean_clusters, std_clusters, model_name, c, f"{out_dir}/{c}_mean_plot_{n_clusters}_{cluster_type}_clusters{supp_str}.{plot_type}", k=n_clusters, log_scale=log_scale, xlabs=xlabs, ignore_idx=ignore_idx)

                if c != "Loss":
                    plot_clusters(loss_vals, clusters, model_name, "Loss", f"{out_dir}/{c}_cluster_Loss_plot_{n_clusters}_{cluster_type}{supp_str}.{plot_type}", k=n_clusters, xlabs=loss_xlabs, ignore_idx=ignore_idx)
                    mean_clusters, std_clusters = compute_means_by_cluster(loss_vals, clusters, k=n_clusters, save_str=f"{out_dir}/{c}_cluster_means_{n_clusters}_{cluster_type}{supp_str}.npy")
                    plot_mean_clusters(mean_clusters, std_clusters, model_name, "Loss", f"{out_dir}/{c}_cluster_Loss_mean_plot_{n_clusters}_{cluster_type}_clusters{supp_str}.{plot_type}", k=n_clusters, xlabs=loss_xlabs, curr_vec=c, ignore_idx=ignore_idx)

                plt.close()
    return

if __name__ == '__main__':
    main()