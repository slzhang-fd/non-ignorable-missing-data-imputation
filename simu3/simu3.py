## Simulation III
## This simulation examines the robustness of our imputation approach when the true
## joint distribution of the data does not conform to the latent variable structure
## assumed by our imputation model. We generate data from a Graphical Model containing
## 5 continuous and 5 binary variables to test model misspecification scenarios.

# %%
import pickle
import sys
import time

import torch
from simu_graphical_depends import (
    MAR_mask,
    generate_graphical_mixed_y,
    mean_analysis_model,
)

sys.path.append("../")

from depend_funcs_nonignorable import (  # noqa: E402
    NonIgnorableImputerInfer,
    NonIgnorableImputerInfer_stand_alone,
)

torch.set_default_tensor_type(torch.DoubleTensor)

# %% generate data
with open("params.pkl", "rb") as f:
    S_cont, S_bin = pickle.load(f)

n_cont = S_cont.shape[0]
n_bin = S_bin.shape[0]
n_items = n_cont + n_bin

n_responses = 5000
measurement_sigma = 0.5

# use for slurm job of large-scale replications
# reps = int(os.getenv("SLURM_ARRAY_TASK_ID"))
reps = 1
torch.manual_seed(reps)

responses = generate_graphical_mixed_y(S_cont, measurement_sigma, S_bin, n_responses)
p_rate = torch.ones(n_items - 1) * 0.1
q_rate = torch.ones(n_items - 1) * 0.4
responses_masked = MAR_mask(responses, p_rate, q_rate)
responses_masked.isnan().float().mean()

# %%
## set dimensions for the latent factor imputation model
n_covs = 1
x_covs = torch.randn(n_responses, n_covs)
x_covs[:, 0] = 1

n_eta = 1
n_xi = 1
initial_values = {
    "log_sigma": torch.zeros(n_cont),
    "W_alpha": torch.zeros(n_items, n_covs),
    "W_beta": torch.zeros(n_items, n_eta).tril(),
    "W_gamma": torch.zeros(n_items, n_covs) - 1.0,
    "W_zeta": torch.zeros(n_items, n_xi).tril(),
    "W_kappa": torch.zeros(n_xi, n_eta),
}
w_beta_mask = torch.ones(initial_values["W_beta"].shape).tril().bool()
w_zeta_mask = torch.ones(initial_values["W_zeta"].shape).tril().bool()


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
        model.W_kappa.detach().clone().flatten(),
    ],
    dim=0,
)

# %% Check infer results
params_hat_all = {
    "log_sigma": model.log_sigma.detach().clone(),
    "W_alpha": model.W_alpha.detach().clone(),
    "W_beta": model.W_beta.detach().clone(),
    "W_gamma": model.W_gamma.detach().clone(),
    "W_zeta": model.W_zeta.detach().clone(),
    "W_kappa": model.W_kappa.detach().clone(),
    "w_kappa_mask": torch.ones(initial_values["W_kappa"].shape).bool(),
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
with open("../res_folder/simu3/res_" + str(reps) + ".pkl", "wb") as f:
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
