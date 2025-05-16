from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import (
    HookPoint,
) 
from transformer_lens.utils import to_numpy

import argparse
import pickle
import random
import os
import gc
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType as TT
import tqdm.auto as tqdm

import datasets
from datasets import load_dataset, load_from_disk
from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import AutoTokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast
from hessian_eigenthings import compute_hessian_eigenthings
from torch.utils.data import TensorDataset, DataLoader
from .incremental_pca import IncrementalPCAonGPU
# from sklearn.decomposition import IncrementalPCA

NUM_DIGITS=3
PLUS_INDEX = 10
EQUALS_INDEX = 11
device = "cuda" if torch.cuda.is_available() else "cpu"
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
def batch_iterator(datasets, split="train+validation", batch_size=1000):
    if len(datasets[0]) == 1:
        dset1 = load_dataset(datasets[0], split=split)
    else:
        dset1 = load_dataset(datasets[0][0], datasets[0][1], split=split)

    for i in range(0, len(dset1), batch_size):
        yield dset1[i : i + batch_size]["text"]
    
def train_tokenizer(domain_datasets, path, split="train+validation"):

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train_from_iterator(batch_iterator(domain_datasets, split=split), vocab_size=30_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save(path)
    return tokenizer

def data_generator(datasets, batch_size, split, seed, from_disk=False):
    torch.manual_seed(seed)
    if not from_disk:
        if len(datasets[0]) == 1:
            simple_data = load_dataset(datasets[0][0], split=split)
        else:
            simple_data = load_dataset(datasets[0][0], datasets[0][1], split=split)
    else:
        simple_data = load_from_disk(datasets[0][0])[split]

    return DataLoader(
            simple_data,
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True)

def get_random_subset(domain_datasets, sample_size=16, split="validation", seed=42, save_file="", loader=False):
    if save_file == "" or not os.path.exists(save_file):
        if not loader:
            sample_ds = data_generator(domain_datasets, sample_size, split=split, seed=seed)
        else:
            sample_ds = domain_datasets

        _, sample_docs = next(enumerate(sample_ds))
        sample_df = pd.DataFrame({"text": list(sample_docs["text"])})
        sample_df = sample_df.sample(frac = 1)
        if save_file != "":
            sample_df.to_csv(save_file)
    else:
        sample_df = pd.read_csv(save_file)

    all_sample_docs = list(sample_df.text)
    return all_sample_docs

def get_pred_log_probs(logits, tokens):
    trunc_logits = logits[:, -(NUM_DIGITS+2):-1]
    ans_tokens = tokens[:, -(NUM_DIGITS+1):]
    log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
    return pred_log_probs

def loss_fn(logits, tokens, per_doc=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    logits = logits[:, :-1]
    tokens = tokens[:, 1:]

    torch.cuda.empty_cache()

    log_probs = F.log_softmax(logits, dim=-1)
    del logits
    torch.cuda.empty_cache()

    correct_log_probs = torch.gather(log_probs, -1, tokens[..., None])[..., 0]
    del log_probs
    torch.cuda.empty_cache()
    
    if per_doc:
        return -correct_log_probs #.mean(axis=1)
    else:
        return -correct_log_probs.mean()

def batch_gradient(network, X, attn_mask, loss_function, retain_graph=False):
    losses = loss_function(network(X, attention_mask = attn_mask), X, per_doc = True).flatten()
    # losses = losses.reshape((losses.shape[0] * losses.shape[1], 1))
    batched_grads = torch.autograd.grad(losses, network.parameters(), grad_outputs=(torch.eye(losses.shape[0], device='cuda'), ), is_grads_batched=True, retain_graph=retain_graph)
    return torch.stack([torch.nn.utils.parameters_to_vector([g[loss_ind].contiguous() for g in batched_grads]) for loss_ind in range(losses.shape[0])]), losses

def batch_hvp(network, X, attn_mask, vecs, loss_function):
    grad, _ = batch_gradient(network, X, attn_mask, loss_function, retain_graph=True)
    batched_hvp = torch.autograd.grad(grad, network.parameters(), grad_outputs=vecs, only_inputs=True)
    return torch.stack([torch.nn.utils.parameters_to_vector([g[loss_ind].contiguous() for g in batched_hvp]) for loss_ind in range(vecs.shape[0])])

def compute_rayleigh(X, AX):
    return torch.squeeze(torch.matmul(X.unsqueeze(1), AX.unsqueeze(3)))

def project_nullspace(V, A):
   # (I - A^T * (A * A^T)^-1 * A) V
   # V = (N x M), A = (N x M) (M >> N)
   # multiplied in order that minimizes the size of all matrices computed
   return (V.T - torch.matmul(A.T, torch.matmul(torch.linalg.lstsq(torch.matmul(A, A.T), A).solution, V.T))).T

def compute_hessian_eigenvectors(model, n_eigs, sample_loader, save_dir, criterion, hessian_num, next_topk=0):
    # (1) compute n_eigs top hessian eigenvectors/eigenvalues
    # criterion = loss_fn
    # using: https://github.com/noahgolmant/pytorch-hessian-eigenthings/tree/master
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model,
        sample_loader,
        criterion,
        n_eigs if next_topk == 0 else next_topk + n_eigs,
        mode='power_iter',
        # power_iter_steps=args.num_steps,
        max_possible_gpu_samples=256,
        # momentum=args.momentum,
        full_dataset=True,
        use_gpu=True,
        # power_iter_steps=30,
    )

    if next_topk > 0:
        top_eigenvecs = eigenvecs[:next_topk, :].copy()
        eigenvecs = eigenvecs[next_topk:, :].copy()
        eigenvals = eigenvals[next_topk:]
        

    gc.collect()
    torch.cuda.empty_cache()

    # (2) add top eigenvectors to basis B
    if hessian_num == 0:
        B = torch.nn.functional.normalize(torch.tensor(eigenvecs.copy()).to("cuda"))
        all_eigvals = eigenvals
    else:
        prev_B = torch.tensor(np.load(f'{save_dir}/hessian_eigenvector_basis_ckpt_{hessian_num-1}.npy')) #.to("cuda")
        all_eigvals = np.concatenate([np.load(f'{save_dir}/all_hessian_eigenvalues_ckpt_{hessian_num-1}.npy'), to_numpy(eigenvals)])

        # (3) project current top k eigenvectors to the nullspace of B
        # note: computing on CPU because there is an issue with the linsolver for pytorch cuda
        if next_topk:
            B = torch.concat((prev_B, 
                              torch.nn.functional.normalize(
                                  project_nullspace(torch.tensor(eigenvecs.copy()), torch.concat((prev_B, torch.tensor(top_eigenvecs))))
                                  )
                            ))
        else:
            B = torch.concat((prev_B, torch.nn.functional.normalize(project_nullspace(torch.tensor(eigenvecs.copy()), prev_B))))

    # print(B.shape)
    np.save(f'{save_dir}/hessian_eigenvector_basis_ckpt_{hessian_num}.npy', to_numpy(B))
    if os.path.exists(f'{save_dir}/hessian_eigenvector_basis_ckpt_{hessian_num-1}.npy'):
        os.remove(f'{save_dir}/hessian_eigenvector_basis_ckpt_{hessian_num-1}.npy')
    np.save(f'{save_dir}/all_hessian_eigenvalues_ckpt_{hessian_num}.npy', all_eigvals)

    del eigenvals
    del all_eigvals
    gc.collect()
    torch.cuda.empty_cache()
    return


def compute_svd_basis(model, n_eigs, sample_loader, save_dir, criterion, svd_num, batch_size=64, normalize_svd=False):
    # ipca = IncrementalPCA(n_components=n_eigs)
    ipca = IncrementalPCAonGPU(n_components=n_eigs)
    # (1) compute n_eigs top svd of gradient
    # criterion = loss_fn
    # using: https://github.com/noahgolmant/pytorch-hessian-eigenthings/tree/master
    # gradients_lst = []
    for i, (curr_test_tokens, curr_attn_mask) in tqdm.tqdm(enumerate(sample_loader)):

        max_batch_size = 1 #// min(curr_test_tokens.shape[1], batch_size)
        for k in range(0, curr_test_tokens.shape[0]): #, max_batch_size):
            # (4.2) compute gradients at each pt in the batch 
            for j in range(0, curr_test_tokens.shape[1], batch_size):
                torch.cuda.empty_cache()
                if not isinstance(model, list):
                    gradients, _ = batch_gradient(model, curr_test_tokens[k:k+max_batch_size, j:j+batch_size].to("cuda"), None, criterion) 
                else:
                    gradients, _ = batch_gradient(model[0], curr_test_tokens[k:k+max_batch_size, j:j+batch_size].to("cuda"), None, criterion)
                    gradients /= len(model)
                    for m in model[1:]:
                        torch.cuda.empty_cache()
                        curr_grad, _ = batch_gradient(m, curr_test_tokens[k:k+max_batch_size, j:j+batch_size].to("cuda"), None, criterion)
                        gradients += curr_grad / len(model)
                        del curr_grad
                        gc.collect()
                        torch.cuda.empty_cache()


                if normalize_svd:
                    gradients = gradients / torch.linalg.norm(gradients, dim=1).unsqueeze(-1)
                with torch.no_grad():
                    ipca.partial_fit(gradients)

                del gradients
                gc.collect()
                torch.cuda.empty_cache()
            
    eigenvecs = ipca.components_.clone().to("cuda")

    del ipca
    gc.collect()
    torch.cuda.empty_cache()

    # (2) add top eigenvectors to basis B
    if svd_num == 0:
        B = torch.nn.functional.normalize(eigenvecs)
        # all_eigvals = eigenvals
    else:
        prev_B = torch.tensor(np.load(f'{save_dir}/svd_eigenvector_basis_ckpt_{svd_num-1}.npy')).to("cuda")

        # (3) project current top k eigenvectors to the nullspace of B
        # note: computing on CPU because there is an issue with the linsolver for pytorch cuda
        B = torch.concat((prev_B, torch.nn.functional.normalize(project_nullspace(eigenvecs, prev_B))))

    np.save(f'{save_dir}/svd_eigenvector_basis_ckpt_{svd_num}.npy', to_numpy(B))
    if os.path.exists(f'{save_dir}/svd_eigenvector_basis_ckpt_{svd_num-1}.npy'):
        os.remove(f'{save_dir}/svd_eigenvector_basis_ckpt_{svd_num-1}.npy')

    gc.collect()
    torch.cuda.empty_cache()
    return



def run_lca_analysis(model, prev_model, lca_loader, save_dir, iter_num, hessian_num, loss_function, second_order=False, batch_size = 64, vec_comp = "hessian"):
    append_str = f"{vec_comp}_" if vec_comp != "hessian" else ""
    # (4) LCA of hessian eigenvalues/eigenvecs: what is contribution of direction to performance improvement on that sample + save LCA result
    # (4.1) compute projection of difference in model parameters to B
    if iter_num > 0:
        with torch.inference_mode():
            prev_B = torch.tensor(np.load(f'{save_dir}/{vec_comp}_eigenvector_basis_ckpt_{hessian_num}.npy')).to("cuda")

            weight_diff = torch.nn.utils.parameters_to_vector(model.parameters()) - torch.nn.utils.parameters_to_vector(prev_model.parameters())
            
            # use previous basis
            proj_weight_diff = torch.matmul(weight_diff, prev_B.T)

            del weight_diff
            del prev_model
            gc.collect()
            torch.cuda.empty_cache()

    tot_i = 0
    curr_idx = 0
    for i, (curr_test_tokens, curr_attn_mask) in tqdm.tqdm(enumerate(lca_loader)):
        tot_i += curr_test_tokens.shape[0] * curr_test_tokens.shape[1]
        for j in range(0, curr_test_tokens.shape[1], batch_size):
            # if i == 0:
            #     j_iter += 1
            torch.cuda.empty_cache()
            gradients, losses = batch_gradient(model, curr_test_tokens[:, j:j+batch_size].to("cuda"), curr_attn_mask[:, j:j+batch_size].to("cuda"), loss_function)
            np.save(f'{save_dir}/{append_str}losses_{iter_num}_index_{curr_idx}.npy', to_numpy(losses))

            if second_order:
                if curr_idx == 0:
                    tot_grad = torch.sum(gradients, dim=0)
                else:
                    tot_grad = tot_grad + torch.sum(gradients, dim=0)

            if iter_num > 0:
                with torch.inference_mode():
                    # (4.3) compute contribution of each eigenvector direction to loss (LCA)
                    # project gradient at x^t-1 onto eigenvectors (multiply by eigenvectors)
                    prev_grad = torch.load(f'{save_dir}/{append_str}prev_batch_gradients_index_{curr_idx}.pt')
                    grad_proj = torch.matmul(prev_grad, prev_B.T)
                    # (4.3.1) save projected gradient for alternative clustering
                    np.save(f'{save_dir}/{append_str}grad_proj_ckpt_{iter_num-1}_index_{curr_idx}.npy', to_numpy(grad_proj))
                    
                    # (4.4) save LCA results: LCA_t-1,i = proj(grad(x^t-1))_i * proj(theta^t - theta^t-1)_i
                    lca = grad_proj * proj_weight_diff
                np.save(f'{save_dir}/{append_str}lca_ckpt_{iter_num-1}_index_{curr_idx}.npy', to_numpy(lca))

            # np.save(f'{save_dir}/prev_batch_gradients_index_{i}.npy', to_numpy(gradients))
            torch.save(gradients, f'{save_dir}/{append_str}prev_batch_gradients_index_{curr_idx}.pt')
            gc.collect()
            torch.cuda.empty_cache()

            curr_idx += 1

    if second_order:
        torch.save(tot_grad, f'{save_dir}/total_grad_ckpt_{iter_num}.pt')
        del tot_grad
        gc.collect()
        torch.cuda.empty_cache()

        if iter_num > 1:
            # (4.5) second order approximation
            eigvals = np.load(f'{save_dir}/all_hessian_eigenvalues_ckpt_{hessian_num}.npy')
            with torch.inference_mode():
                prev_tot_grad = torch.load(f'{save_dir}/total_grad_ckpt_{iter_num-2}.pt')
                os.remove(f'{save_dir}/total_grad_ckpt_{iter_num-2}.pt')
                
                tot_grad = torch.load(f'{save_dir}/total_grad_ckpt_{iter_num-1}.pt')
                tot_grad_proj_diff = to_numpy(torch.matmul(tot_grad - prev_tot_grad, prev_B.T))

                for i in tqdm.tqdm(range(curr_idx)):
                    lca = np.load(f'{save_dir}/lca_ckpt_{iter_num-2}_index_{i}.npy')

                    grad_proj = np.load(f'{save_dir}/grad_proj_ckpt_{iter_num-1}_index_{i}.npy')[:, :lca.shape[1]]
                    prev_grad_proj = np.load(f'{save_dir}/grad_proj_ckpt_{iter_num-2}_index_{i}.npy')[:, :lca.shape[1]]
                    lca = lca + 1/2 * eigvals[:lca.shape[1]] * (grad_proj - prev_grad_proj / tot_grad_proj_diff[:lca.shape[1]]) * to_numpy(proj_weight_diff.cpu() ** 2)[:lca.shape[1]]

                    np.save(f'{save_dir}/lca_second_order_ckpt_{iter_num-2}_index_{i}.npy', to_numpy(lca))

    gc.collect()
    torch.cuda.empty_cache()
    return
    
def main():
    print("Running")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", help='directory to save results', type=str)
    parser.add_argument("-t", "--analysis_type", help='type of analysis', type=str)
    parser.add_argument("-k", "--n_eigs", help='number of hessian eigenvectors to compute at each epoch', type=int)
    parser.add_argument("-n", "--n_examples", help='number of test examples to evaluate LCA on', type=int)
    parser.add_argument("-e", "--n_hessian_examples", help='number of examples for hessian', nargs='?', type=int, const=-1, default=-1)
    parser.add_argument("--lca_step_size", help="lca step size", type=int)
    parser.add_argument("--seed", help="random seed", type=int)
    parser.add_argument("--hessian_step_size", help="hessian step size", type=int)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--second_order', dest='feature', action='store_true')
    feature_parser.add_argument('--no_second_order', dest='feature', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()
    print("Parsed Arguments")

    save_dir = args.save_dir
    analysis_type = args.analysis_type
    n_eigs = args.n_eigs
    second_order = args.feature
    domain_datasets = [["wikipedia", "20220301.en"]]
    token_path = f"{save_dir}/tokenizer"
    full_split="train[:1000000]"
    split_str = "train" 
    max_len = 128

    print("creating model")
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=512,
        d_head=512//4,
        n_heads=4,
        d_mlp=4*512,
        d_vocab= 30001,
        d_vocab_out= 30001,
        n_ctx=max_len,
        act_fn="relu",
        normalization_type="LN",
    )

    model = HookedTransformer(cfg)
    if not os.path.exists(token_path):
        tokenizer = train_tokenizer(domain_datasets, token_path, split=full_split)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=token_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    gc.collect()
    torch.cuda.empty_cache()

    batch_size = 64 #256
    split_batch_size = 64
    test_size = args.n_examples
    hessian_size = args.n_hessian_examples
    if hessian_size < 0:
        hessian_size = test_size
    hessian_step_size = 256
    if "svd" in analysis_type:
        hessian_step_size = 1
    test_step_size = 1
    rand_seed = args.seed
    rand_seed2 = 297 #54 #297 # 321
    num_epochs = 1 #30
    eval_window = args.lca_step_size
    hessian_indices = args.hessian_step_size
    tot_n_hessians = 30
    lr = 1e-4
    betas = (0.9, 0.95)
    max_grad_norm = 1.0
    wd = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 100, 1.0))

    print("generating data")
    if analysis_type == "loss_logits_activation":
        split="validation"
        save_str=f'{save_dir}/test_docs.csv'
    else:
        # split=split_str
        split="test"
        save_str=f'{save_dir}/lca_docs.csv'
    
    dataset=None
    if domain_datasets[0][0] == "wikipedia":
        data_dir = os.path.join(os.path.dirname(save_dir), "wikipedia")
        # if not os.path.exists(data_dir):
        dataset = load_dataset(domain_datasets[0][0], domain_datasets[0][1], split=full_split)
        dataset = dataset.train_test_split(test_size=0.1, load_from_cache_file = True,
                                           train_indices_cache_file_name = f"{data_dir}/train", test_indices_cache_file_name=f"{data_dir}/test")

    if dataset is not None:
        torch.manual_seed(rand_seed)
        test_loader = DataLoader(
                dataset["test"],
                batch_size=test_size, shuffle=True,
                num_workers=0, pin_memory=True)
        
        all_test_docs = get_random_subset(test_loader, save_file=save_str, loader=True)
    else:
        all_test_docs = get_random_subset(domain_datasets, test_size, split=split, seed=rand_seed, save_file=save_str)

    test_tokens = tokenizer(all_test_docs, padding = True, truncation=True, max_length = max_len, return_tensors="pt").to("cpu")
    test_dataset = TensorDataset(test_tokens.input_ids, test_tokens.attention_mask)
    test_loader = DataLoader(test_dataset, test_step_size, shuffle=False, num_workers=0)
    
    del test_tokens
    gc.collect()
    torch.cuda.empty_cache()

    if dataset is not None:
        torch.manual_seed(rand_seed)
        ds = DataLoader(
                dataset["train"],
                batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True)
    else:
        ds = data_generator(domain_datasets, batch_size, split=split_str, seed=rand_seed)
    
    if "eigenvector_lca_analysis" in analysis_type:
        if dataset is not None:
            torch.manual_seed(rand_seed2)
            hessian_loader = DataLoader(
                dataset["train"],
                batch_size=hessian_size, shuffle=True,
                num_workers=0, pin_memory=True)
        else:
            hessian_loader = data_generator(domain_datasets, hessian_size, split_str, seed=n_hessian_eval)


    inum = 0
    n_evals = 0
    n_hessian_eval = 0
    loss = 0.0
    print(f"Training model with evaluation window {eval_window}")

    for docs in tqdm.tqdm(ds):

        tokens = tokenizer(docs["text"], padding = True, truncation=True,  max_length = max_len, return_tensors="pt").to(device)
        logits = model(tokens.input_ids, attention_mask=tokens.attention_mask) #, 

        loss = loss_fn(logits, tokens.input_ids)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        del tokens
        del logits
        del docs
        gc.collect()
        torch.cuda.empty_cache()

        if inum % eval_window == 0:
            pickle.dump(model, open(f"{save_dir}/model_n_evals_{n_evals}.obj", "wb"))
            n_evals += 1

        if "eigenvector_lca_analysis" in analysis_type and (inum == eval_window or inum % hessian_indices == 0) and n_hessian_eval < tot_n_hessians:
            # create sample loader with random train documents to compute hessian eigenvectors on
            print("Creating Sample Loader for Hessian Eigenvectors")
            _, docs = next(enumerate(hessian_loader))
            tokens = tokenizer(docs["text"], padding = True, truncation=True,  max_length = max_len, return_tensors="pt").input_ids
            sample_dataset = TensorDataset(tokens, tokens)
            sample_loader = DataLoader(sample_dataset, hessian_step_size, shuffle=True, num_workers=0)

            del tokens
            del docs
            gc.collect()
            torch.cuda.empty_cache()

            print("Running Hessian Computation")
            if "svd" in analysis_type:
                # compute hessian eigenvectors at the beginning of each epoch (except wait n_evals iterations at the first epoch for stability)
                compute_svd_basis(model, n_eigs, sample_loader, save_dir, loss_fn, n_hessian_eval, 8)
                # epoch_run = True
            else:
                compute_hessian_eigenvectors(model, n_eigs, sample_loader, save_dir, loss_fn, n_hessian_eval)
                
            n_hessian_eval += 1

            del sample_dataset
            del sample_loader
            gc.collect()
            torch.cuda.empty_cache()

        inum += 1
        if inum % 100 == 1:
            print(f"Iteration: {inum} Train Loss: {loss.item()}")
    # n_hessian_eval = tot_n_hessians

    n_evals = 0
    for docs in tqdm.tqdm(ds):
        losses_test = None
        if analysis_type == "loss_logits_activation":
            losses_test = [] #run_loss_logits_activation_analysis(model, test_tokens, save_dir, test_step_size, inum)
        elif "eigenvector_lca_analysis" in analysis_type and inum % eval_window == 0:
            model = pickle.load(open(f"{save_dir}/model_n_evals_{n_evals}.obj", "rb"))

            # compute LCA every n_evals iterations
            prev_model = None
            if n_evals > 0:
                prev_model = pickle.load(open(f"{save_dir}/model_n_evals_{n_evals-1}.obj", "rb"))
            
            print("Running Hessian Eigenvector LCA Analysis")
            vec_comp = "hessian"
            if "svd" in analysis_type:
                vec_comp = "svd"
            run_lca_analysis(model, prev_model, test_loader, save_dir, n_evals, n_hessian_eval-1, loss_fn, second_order, vec_comp = vec_comp, batch_size = split_batch_size)

            n_evals += 1
            
        if inum % 100 == 0:
            test_str = ""
            if losses_test:
                test_str=f"Test Loss: {losses_test.mean().item()}"
                del losses_test

            print(f"Iteration: {inum} Train Loss: {loss.item()} {test_str}")
        
        gc.collect()
        torch.cuda.empty_cache()
        inum += 1

if __name__ == '__main__':
    main()
