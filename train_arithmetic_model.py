from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import (
    HookPoint,
) 
import torch
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType as TT

import tqdm.auto as tqdm
from transformer_lens.utils import to_numpy
import argparse
import gc
from hessian_eigenthings import compute_hessian_eigenthings
import pickle
from torch.utils.data import TensorDataset, DataLoader
from .train_model_eng_domains import compute_hessian_eigenvectors, compute_svd_basis, compute_rayleigh


NUM_DIGITS=3
PLUS_INDEX = 10
EQUALS_INDEX = 11
device = "cuda" if torch.cuda.is_available() else "cpu"

def data_generator(batch_size, seed):
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, 3*NUM_DIGITS+3)).to(torch.int64)
        x = torch.randint(0, 10, (batch_size, NUM_DIGITS))
        y = torch.randint(0, 10, (batch_size, NUM_DIGITS))
        batch[:, :NUM_DIGITS] = x
        batch[:, NUM_DIGITS] = PLUS_INDEX
        batch[:, 1+NUM_DIGITS:1+NUM_DIGITS*2] = y
        batch[:, 1+NUM_DIGITS*2] = EQUALS_INDEX
        carries = [torch.zeros((batch_size,)).to(torch.int64)]
        for i in range(NUM_DIGITS):
            carry = carries[-1]
            digit_sum = (batch[:, NUM_DIGITS-1-i]+batch[:, 2*NUM_DIGITS-i]+carry)
            batch[:, -1-i] = (digit_sum % 10)
            carry = (digit_sum>=10).to(torch.int64)
            carries.append(carry)
        batch[:, -1-NUM_DIGITS] = carries[-1]
        carries = torch.stack(carries, axis=1)
        yield batch.cuda(), carries.cuda()

def get_pred_log_probs(logits, tokens):
    trunc_logits = logits[:, -(NUM_DIGITS+2):-1]
    ans_tokens = tokens[:, -(NUM_DIGITS+1):]
    log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
    return pred_log_probs

def loss_fn(logits, tokens, per_doc=False):
    if per_doc:
        return -get_pred_log_probs(logits, tokens) #.mean(axis=1)
    return -get_pred_log_probs(logits, tokens).mean()

def batch_gradient(network, X, loss_function, retain_graph=False):
    losses = loss_function(network(X), X, per_doc = True).flatten()
    batched_grads = torch.autograd.grad(losses, network.parameters(), grad_outputs=(torch.eye(losses.shape[0], device='cuda'), ), is_grads_batched=True, retain_graph=retain_graph)
    return torch.stack([torch.nn.utils.parameters_to_vector([g[loss_ind].contiguous() for g in batched_grads]) for loss_ind in range(losses.shape[0])]), losses

def single_hvp(sample, vec, params):
    return torch.autograd.grad(sample, params, grad_outputs=vec)

def batch_hvp(network, X, vecs, loss_function, grad=None):
    if grad is None:
        losses = loss_function(network(X), X, per_doc = True)
        grad = torch.autograd.grad(losses, network.parameters(), grad_outputs=(torch.eye(X.shape[0], device='cuda'), ), is_grads_batched=True, create_graph=True)
        grad = torch.stack([torch.cat([g[loss_ind].flatten() for g in grad]) for loss_ind in range(X.shape[0])])
    gradlist = []
    for g in grad:
        batched_hvp = torch.autograd.grad(g, network.parameters(), grad_outputs=(vecs, ), is_grads_batched=True, retain_graph=True)
        gradlist.append(torch.stack([torch.nn.utils.parameters_to_vector([g[i].contiguous() for g in batched_hvp]) for i in range(vecs.shape[0])]))
    return torch.stack(gradlist)

def run_lca_analysis(model, prev_model, lca_loader, save_dir, iter_num, hessian_num, loss_function, second_order=False, vec_comp="hessian"):
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
            if not second_order:
                del prev_model
            gc.collect()
            torch.cuda.empty_cache()

    tot_i = 0
    for i, curr_test_tokens in tqdm.tqdm(enumerate(lca_loader)):

        # (4.2) compute gradients at each pt in the batch
        gradients, losses = batch_gradient(model, curr_test_tokens[0].to("cuda"), loss_function)
        np.save(f'{save_dir}/{append_str}losses_{iter_num}_index_{i}.npy', to_numpy(losses))

        if second_order:
            if i == 0:
                tot_grad = torch.sum(gradients, dim=0)
            else:
                tot_grad = tot_grad + torch.sum(gradients, dim=0) 

        if iter_num > 0:
            # (4.3) compute contribution of each eigenvector direction to loss (LCA)
            # project gradient at x^t-1 onto eigenvectors (multiply by eigenvectors)
            # prev_grad = torch.tensor(np.load(f'{save_dir}/prev_batch_gradients_index_{i}.npy')).to("cuda")
            with torch.inference_mode():
                prev_grad = torch.load(f'{save_dir}/{append_str}prev_batch_gradients_index_{i}.pt')
                grad_proj = torch.matmul(prev_grad, prev_B.T)
                # (4.3.1) save projected gradient for alternative clustering
                np.save(f'{save_dir}/{append_str}grad_proj_ckpt_{iter_num-1}_index_{i}.npy', to_numpy(grad_proj))
                
                # (4.4) save LCA results: LCA_t-1,i = proj(grad(x^t-1))_i * proj(theta^t - theta^t-1)_i
                lca = grad_proj * proj_weight_diff
            np.save(f'{save_dir}/{append_str}lca_ckpt_{iter_num-1}_index_{i}.npy', to_numpy(lca))

        # np.save(f'{save_dir}/prev_batch_gradients_index_{i}.npy', to_numpy(gradients))
        torch.save(gradients, f'{save_dir}/{append_str}prev_batch_gradients_index_{i}.pt')
        gc.collect()
        torch.cuda.empty_cache()

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
                tot_grad = torch.load(f'{save_dir}/total_grad_ckpt_{iter_num-1}.pt')
                tot_grad_proj_diff = to_numpy(torch.matmul(tot_grad - prev_tot_grad, prev_B.T))

                for i, curr_test_tokens in tqdm.tqdm(enumerate(lca_loader)):
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
    parser.add_argument("-k", "--n_eigs", help='number of hessian eigenvectors to compute at each epoch', type=int)
    parser.add_argument("--lca_step_size", help="lca step size", type=int)
    parser.add_argument("--rand_seed", help="random seed used for data", type=int)
    parser.add_argument('--basis_type', nargs='?', const="hessian", type=str, default="hessian")
    parser.add_argument("--next_topk", help="whether to use 2nd top k eigenvecs/values", type=bool)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--second_order', dest='feature', action='store_true')
    feature_parser.add_argument('--no_second_order', dest='feature', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()
    print("Parsed Arguments")

    save_dir = args.save_dir
    n_eigs = args.n_eigs
    second_order = args.feature
    lca_step_size = args.lca_step_size
    rand_seed = args.rand_seed
    basis_type = args.basis_type
    next_topk = args.next_topk

    print("creating model")
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=512,
        d_head=512//4,
        n_heads=4,
        d_mlp=4*512,
        d_vocab= 12,
        d_vocab_out= 10,
        n_ctx=3*NUM_DIGITS+3,
        act_fn="relu",
        normalization_type="LN",
    )
    model = HookedTransformer(cfg)

    batch_size = 64
    test_size = 1250
    # rand_seed = 42
    num_epochs = 10000
    n_hessian_ckpts = 30
    lr = 1e-5
    betas = (0.9, 0.95)
    wd = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 100, 1.0))

    print("generating data")
    test_ds = data_generator(test_size, seed=rand_seed)
    test_tokens, _ = next(test_ds)#[:test_subset]
    test_dataset = TensorDataset(test_tokens)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    np.save(f'{save_dir}/test_tokens.npy', to_numpy(test_tokens))

    ds = data_generator(batch_size, seed=rand_seed)

    n_hessians = 0
    prev_lca_step = 0
    for epoch in tqdm.tqdm(range(num_epochs)):
        tokens, _ = next(ds)
        logits = model(tokens)
        per_token_losses = -get_pred_log_probs(logits, tokens).mean(0)
        loss = per_token_losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % (num_epochs//n_hessian_ckpts) == 0:
            sample_tokens, _ = next(test_ds)
            sample_dataset = TensorDataset(sample_tokens, sample_tokens)
            sample_loader = DataLoader(sample_dataset, batch_size, shuffle=True, num_workers=0)

            if basis_type == "hessian":
                compute_hessian_eigenvectors(model, n_eigs, sample_loader, save_dir, loss_fn, n_hessians, next_topk=next_topk)
            else:
                compute_svd_basis(model, n_eigs, sample_loader, save_dir, loss_fn, n_hessians)
            n_hessians += 1
        
        pickle.dump(model, open(f"{save_dir}/model_n_evals_{epoch}.obj", "wb"))

        # compute LCA every n_evals iterations
        prev_model = None
        if epoch > 0:
            prev_model = pickle.load(open(f"{save_dir}/model_n_evals_{prev_lca_step}.obj", "rb"))
        
        print("Running Hessian Eigenvector LCA Analysis")
        if epoch % lca_step_size == 0:
            run_lca_analysis(model, prev_model, test_loader, save_dir, epoch//lca_step_size, n_hessians-1, loss_fn, second_order, vec_comp=basis_type)
            prev_lca_step = epoch

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} Train Loss: {loss.item()}")
            print(f"Per Token Train Loss: {per_token_losses}")

if __name__ == '__main__':
    main()
