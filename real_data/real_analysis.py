## Real data analysis using European Social Survey (ESS) round 9 data from 2018.
# We analyze data from the "Justice and fairness in Europe" rotating module,
# which measures perceptions of justice regarding income, education and job chances
# across 29 European countries.
# %%
import os  # noqa: F401
import pickle
import sys

import pandas
import torch

sys.path.append("../")
from depend_funcs_ig_poly import (  # noqa: E402
    NonIgnorableImputerInfer_poly_reg,
    NonIgnorableImputerInfer_poly_reg_stand_alone,
    mean_analysis_model,
)

## Load clearned data
with open("response_xcov_data.csv") as f:
    responses = pandas.read_csv(f, header=0)

torch.set_default_tensor_type(torch.DoubleTensor)

# use for slurm job of large-scale replications
# reps = int(os.getenv("SLURM_ARRAY_TASK_ID"))
reps = 1
torch.manual_seed(reps + 1)
responses_subset = responses[responses["cntry"] == reps].iloc[:, 1:]
responses_subset_ts = torch.tensor(responses_subset.values[:, :20], dtype=torch.double)

n_responses = responses_subset_ts.shape[0]
n_cont = 10
n_bin = 0
n_ord = 10
n_items = n_cont + n_bin + n_ord
n_eta = 3
Mj = 4
n_xi = 1

x_covs = torch.randn(n_responses, 1)
x_covs[:, 0] = 1
## remove the last one due to co-linearty
x_covs1 = torch.tensor(responses_subset.values[:, 20:-1], dtype=torch.double)
n_covs1 = x_covs1.shape[1]


# %%
## set intercepts
B = torch.zeros(n_ord, Mj)
B[:, 0] = torch.logit(torch.tensor(1.0 / (Mj + 1))) + torch.randn(n_ord) * 0.1
for k in range(1, Mj):
    B[:, k] = torch.logit(torch.tensor((k + 1) / (Mj + 1)))
B[:, 1:] = B.diff(dim=1).log()
assert B.isfinite().all()

initial_values = {
    "log_sigma": torch.zeros(n_cont),
    "W_alpha": torch.zeros(n_items, 1),
    "W_beta": torch.randn(n_items, n_eta).tril(),
    "B": B.clone(),
    "W_gamma": torch.zeros(n_items, 1) - 2.0,
    "W_zeta": torch.randn(n_items, n_xi).tril(),
    "W_kappa": torch.zeros(n_xi, n_eta),
    "W_coeff_eta": torch.zeros(n_eta, n_covs1),
    "W_coeff_xi": torch.zeros(n_xi, n_covs1),
    "continuous_cols": range(n_cont),
    "binary_cols": range(n_cont, n_cont + n_bin),
    "ordinal_cols": range(n_cont + n_bin, n_cont + n_bin + n_ord),
}
model = NonIgnorableImputerInfer_poly_reg(
    responses_subset_ts.clone(), x_covs.clone(), x_covs1.clone(), initial_values
)
## find initial values for parameters
model.fit(optimizer_choice="Adam", max_iter=3000, lr=0.01, fix_kappa=False)
model.fit(optimizer_choice="MySA_Ruppert", max_iter=3000, lr=0.01, fix_kappa=False)

w_alpha_mask = torch.ones(n_items, 1).bool()
w_alpha_mask[-n_ord:, 0] = False
w_beta_mask = torch.ones(n_items, n_eta).tril().bool()
w_zeta_mask = torch.ones(n_items, n_xi).tril().bool()
params_hat = torch.cat(
    [
        model.log_sigma.detach().clone(),
        model.W_alpha.detach()[w_alpha_mask].clone().flatten(),
        model.W_beta.detach()[w_beta_mask].clone().flatten(),
        model.B.detach().clone().flatten(),
        model.W_gamma.detach().clone().flatten(),
        model.W_zeta.detach()[w_zeta_mask].clone().flatten(),
        model.W_kappa.detach().clone().flatten(),
        model.W_coeff_eta.detach().clone().flatten(),
        model.W_coeff_xi.detach().clone().flatten(),
    ],
    dim=0,
)
# %% Check infer results
params_hat_all = {
    "log_sigma": model.log_sigma.detach().clone(),
    "W_alpha": model.W_alpha.detach().clone(),
    "W_beta": model.W_beta.detach().clone(),
    "B": model.B.detach().clone(),
    "W_gamma": model.W_gamma.detach().clone(),
    "W_zeta": model.W_zeta.detach().clone(),
    "W_kappa": model.W_kappa.detach().clone(),
    "W_coeff_eta": model.W_coeff_eta.detach().clone(),
    "W_coeff_xi": model.W_coeff_xi.detach().clone(),
    "continuous_cols": range(n_cont),
    "binary_cols": range(n_cont, n_cont + n_bin),
    "ordinal_cols": range(n_cont + n_bin, n_cont + n_bin + n_ord),
}

model_infer = NonIgnorableImputerInfer_poly_reg_stand_alone(
    responses_subset_ts.clone(), x_covs.clone(), x_covs1.clone(), params_hat_all
)

(
    responses_all,
    S_ij_all1,
    S_i_obs,
    D_hat_i_all,
    Lambda_hat,
    I_obs_inv,
) = model_infer.infer(mis_copies=20, M=10000, burn_in=1000)
params_vars = Lambda_hat.diag() / n_responses
params_vars1 = I_obs_inv.diag()

## get analysis model results
beta_hat, beta_vars1 = mean_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat
)

# %% save results
with open("../res_folder/real_data/res_" + str(reps) + ".pkl", "wb") as f:
    pickle.dump(
        [
            responses_all,
            S_ij_all1,
            S_i_obs,
            D_hat_i_all,
            Lambda_hat,
            params_vars,
            params_vars1,
            beta_hat,
            beta_vars1,
            params_hat,
            responses_subset_ts,
        ],
        f,
    )
