## Simulation I-2
## True model is ignorable, estimated imputation model is non-ignorable.

# %% import libraries
import pickle
import sys
import time

import torch

sys.path.append("../")

from depend_funcs_ignorable import (
    generate_non_ignorable_mixed_data,
    mean_analysis_model,
)  # noqa: E402
from depend_funcs_nonignorable import (  # noqa: E402
    NonIgnorableImputerInfer,
    NonIgnorableImputerInfer_stand_alone,
)

# Specifiy the number of responses, items, factors, and covariates
n_responses = 5000
n_cont = 10
n_bin = 10
n_items = n_cont + n_bin
n_eta = 4
n_xi = 1
n_covs = 1
torch.set_default_tensor_type(torch.DoubleTensor)

torch.manual_seed(0)  # fix the model parameters, across replications
measurement_sigma = 0.5
W_alpha = torch.randn(n_items, n_covs)
W_beta = (torch.rand(n_items, n_eta) + 0.5).tril()
w_beta_mask = torch.ones(W_beta.shape).tril().bool()
W_gamma = torch.randn(n_items, n_covs) * 0.5 - 3.0
W_zeta = (torch.rand(n_items, n_xi) + 0.2).tril()
w_zeta_mask = torch.ones(W_zeta.shape).tril().bool()
W_kappa = torch.zeros(n_xi, n_eta)
w_kappa_mask = torch.ones(W_kappa.shape).bool()

x_covs = torch.randn(n_responses, n_covs)
x_covs[:, 0] = 1


# use for slurm job of large-scale replications
# reps = int(os.getenv("SLURM_ARRAY_TASK_ID"))
reps = 1
torch.manual_seed(reps)
_, responses_masked = generate_non_ignorable_mixed_data(
    x_covs, measurement_sigma, W_alpha, W_beta, W_gamma, W_zeta, W_kappa, n_cont, n_bin
)
# remove rows that all variables are missing
valid_rows = (~responses_masked.isnan()).sum(dim=1) > 0
responses_masked = responses_masked[valid_rows]
x_covs = x_covs[valid_rows]

n_responses = responses_masked.shape[0]
print("missing rate: ", responses_masked.isnan().float().mean())


# %% Model Fitting (Algorithm1)
initial_values = {
    "log_sigma": torch.zeros(n_cont),
    "W_alpha": torch.zeros(n_items, n_covs),
    "W_beta": torch.zeros(n_items, n_eta).tril(),
    "W_gamma": torch.zeros(n_items, n_covs) - 3.0,
    "W_zeta": torch.zeros(n_items, n_xi).tril(),
    "W_kappa": torch.zeros(n_xi, n_eta),
}

model = NonIgnorableImputerInfer(
    responses_masked.clone(), n_eta, n_xi, x_covs.clone(), initial_values
)
start_time = time.time()
model.fit(optimizer_choice="Adam", max_iter=3000, lr=0.01)
model.fit(optimizer_choice="MySA_Ruppert", max_iter=3000, lr=0.01)
end_time = time.time()
print(f"Time taken to execute the code: {end_time - start_time} seconds")

params_hat = torch.cat(
    [
        model.log_sigma.detach().clone(),
        model.W_alpha.detach().clone().flatten(),
        model.W_beta.detach()[w_beta_mask].clone().flatten(),
        model.W_gamma.detach().clone().flatten(),
        model.W_zeta.detach()[w_zeta_mask].clone().flatten(),
        model.W_kappa.detach()[w_kappa_mask].clone().flatten(),
    ],
    dim=0,
)

# %% Perform Multiple Imputation and Inference (Algorithm 2)
params_hat_all = {
    "log_sigma": model.log_sigma.detach().clone(),
    "W_alpha": model.W_alpha.detach().clone(),
    "W_beta": model.W_beta.detach().clone(),
    "W_gamma": model.W_gamma.detach().clone(),
    "W_zeta": model.W_zeta.detach().clone(),
    "W_kappa": model.W_kappa.detach().clone(),
    "w_kappa_mask": w_kappa_mask.clone(),
}
model_infer = NonIgnorableImputerInfer_stand_alone(
    responses_masked.clone(), n_eta, n_xi, x_covs.clone(), params_hat_all
)
start_time = time.time()
responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, I_obs_inv = (
    model_infer.infer(mis_copies=20, M=2000, burn_in=1000)
)
end_time = time.time()
print(f"Time taken to execute the code: {end_time - start_time} seconds")
params_vars = Lambda_hat.diag() / n_responses
params_vars1 = I_obs_inv.diag()

## get analysis model results
beta_hat, beta_vars1 = mean_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat
)

# %% save results
with open("../res_folder/simu1-2/res_" + str(reps) + ".pkl", "wb") as f:
    pickle.dump(
        [
            responses_all,
            S_ij_all1,
            S_i_obs,
            D_hat_i_all,
            Lambda_hat,
            params_hat,
            params_vars,
            params_vars1,
            beta_hat,
            beta_vars1,
            responses_masked,
        ],
        f,
    )
