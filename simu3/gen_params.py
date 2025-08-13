# %%
import pickle

import torch

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(3)

n_cont = 5
n_bin = 5
n_items = n_cont + n_bin
K = 1

## set a sparse network structure of S
S_bin = torch.zeros((n_bin, n_bin))
S_nonzero_prob_bin = 0.5

## randomly select some elements in the lower triangular matrix to be non-zero
mask = torch.tril(torch.ones(n_bin, n_bin), diagonal=-1).bool()
S_bin[mask] = (
    (torch.rand(int(n_bin * (n_bin - 1) / 2)) < S_nonzero_prob_bin)
    * (torch.rand(int(n_bin * (n_bin - 1) / 2)) * 0.6 + 0.4)
    * (torch.randint(2, (int(n_bin * (n_bin - 1) / 2),)) * 2 - 1)
)

## make S symmetric
S_bin = S_bin + S_bin.T

## The diagonal of S is s_{jj} + 2*b_j, we set to be 0 here
S_bin.fill_diagonal_(0)

# %%
torch.manual_seed(4)

## set a sparse network structure of S_cont
S_nonzero_prob_cont = 0.5
# Start with random positive definite matrix
A = torch.randn(n_cont, n_cont)
S_cont = torch.mm(A, A.t())

# Sparsify the S_cont
mask = torch.rand(n_cont, n_cont) < S_nonzero_prob_cont
mask = mask.triu(diagonal=1)  # Upper triangular part
S_cont = S_cont * (mask + mask.t())  # Symmetric mask

# Diagonal dominance to ensure positive definiteness
S_cont += torch.diag(S_cont.abs().sum(dim=1) + 0.1)
S_cont

# %%
with open("params.pkl", "wb") as f:
    pickle.dump([S_cont, S_bin], f)

# %%
