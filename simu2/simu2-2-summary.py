# %%
import pickle
import sys

import torch

sys.path.append("../")

from depend_funcs import (
    plot_analysis_boxplot,
    plot_analysis_coverage,
    plot_analysis_point_estimation,
    plot_analysis_sd_estimation,
)
from depend_funcs_ignorable import (
    cond_mean_analysis_model,
    generate_non_ignorable_mixed_data,
)

params_hat_all = []
params_vars_all = []
params_vars_all1 = []
beta_hat_all = []
beta_sd_all1 = []

for reps in range(100):
    with open("../res_folder/simu2-2/res_" + str(reps + 1) + ".pkl", "rb") as f:
        (
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
        ) = pickle.load(f)
        params_hat_all.append(params_hat.detach())
        params_vars_all.append(params_vars.detach())
        params_vars_all1.append(params_vars1.detach())
        beta_hat, beta_vars1 = cond_mean_analysis_model(
            responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat
        )
        beta_hat_all.append(beta_hat.detach())
        beta_sd_all1.append(beta_vars1.detach().sqrt())

analysis_index = 2

# %%
n_responses = responses_masked.shape[0]
n_cont = 10
n_bin = 10
n_items = n_cont + n_bin
n_eta = 4
n_xi = 1
n_covs = 1
torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
measurement_sigma = 0.5
W_alpha = torch.randn(n_items, n_covs)
W_beta = (torch.rand(n_items, n_eta) + 0.5).tril()
w_beta_mask = torch.ones(W_beta.shape).tril().bool()
W_gamma = torch.randn(n_items, n_covs) * 0.5 - 3.0
W_zeta = (torch.rand(n_items, n_xi) + 0.2).tril()
w_zeta_mask = torch.ones(W_zeta.shape).tril().bool()
W_kappa = torch.rand(n_xi, n_eta) + 1.0
w_kappa_mask = W_kappa > 0


params_true = torch.cat(
    [
        torch.full((n_cont,), measurement_sigma).log(),
        W_alpha.flatten(),
        W_beta[w_beta_mask].flatten(),
        W_gamma.flatten(),
        W_zeta[w_zeta_mask].flatten(),
        W_kappa[w_kappa_mask].flatten(),
    ],
    dim=0,
)
# Create sub-ranges
ranges = [
    (0, n_cont),
    (n_cont, n_cont + n_items * n_covs),
    (n_cont + n_items * n_covs, n_cont + n_items * n_covs + w_beta_mask.sum()),
    (
        n_cont + n_items * n_covs + w_beta_mask.sum(),
        n_cont + n_items * n_covs + w_beta_mask.sum() + n_items * n_covs,
    ),
    (
        n_cont + n_items * n_covs + w_beta_mask.sum() + n_items * n_covs,
        n_cont
        + n_items * n_covs
        + w_beta_mask.sum()
        + n_items * n_covs
        + w_zeta_mask.sum(),
    ),
    (
        n_cont
        + n_items * n_covs
        + w_beta_mask.sum()
        + n_items * n_covs
        + w_zeta_mask.sum(),
        n_cont
        + n_items * n_covs
        + w_beta_mask.sum()
        + n_items * n_covs
        + w_zeta_mask.sum()
        + w_kappa_mask.sum(),
    ),
]
estimated_mat = torch.stack(params_hat_all)
estimated_mat[:, ranges[2][0] : ranges[2][1]] = torch.abs(
    estimated_mat[:, ranges[2][0] : ranges[2][1]]
)
estimated_mat[:, ranges[4][0] : ranges[4][1]] = torch.abs(
    estimated_mat[:, ranges[4][0] : ranges[4][1]]
)
estimated_mat[:, ranges[5][0] : ranges[5][1]] = torch.abs(
    estimated_mat[:, ranges[5][0] : ranges[5][1]]
)

# %% point estimation for the analysis model
##########################################################################################
NN = int(1e6)
responses_ps, _ = generate_non_ignorable_mixed_data(
    torch.ones(NN, 1),
    measurement_sigma,
    W_alpha,
    W_beta,
    W_gamma,
    W_zeta,
    W_kappa,
    n_cont,
    n_bin,
)
## mean analysis model
if analysis_index == 1:
    analysis_params_true = responses_ps.mean(0)
elif analysis_index == 2:
    # conditional mean analysis model
    analysis_params_true = torch.zeros(n_items * 2 - 2)
    analysis_params_true[: (n_items - 1)] = responses_ps[
        responses_ps[:, -1] == 0, :-1
    ].mean(0)
    analysis_params_true[(n_items - 1) :] = (
        responses_ps[responses_ps[:, -1] == 1, :-1].mean(0)
        - analysis_params_true[: (n_items - 1)]
    )

estimated_mean = torch.stack(beta_hat_all).mean(0)

plot_analysis_point_estimation(
    analysis_params_true,
    estimated_mean,
    "../res_folder/simu2-2/analysis" + str(analysis_index) + "_point.png",
)

# %% boxplot for analysis point estimation
##########################################################################################
difference = torch.stack(beta_hat_all) - analysis_params_true

plot_analysis_boxplot(
    difference,
    "../res_folder/simu2-2/analysis" + str(analysis_index) + "_box.png",
    xaxis_ticks=analysis_index == 1,
)

# %% empirical and estimated SD of analysis model parameters
##########################################################################################
empirical_sd = torch.stack(beta_hat_all).var(0).sqrt()
estimated_sd = torch.stack(beta_sd_all1).median(0).values

plot_analysis_sd_estimation(
    empirical_sd,
    estimated_sd,
    "../res_folder/simu2-2/analysis" + str(analysis_index) + "_sd.png",
)

# %% coverage for the analysis model parameters
##########################################################################################
z_value = 1.96  # for 95% CI with a normal distribution
# z_value = 1.645  # for 90% CI with a normal distribution
margin_of_error = z_value * torch.stack(beta_sd_all1).median(0).values
lower_bound = torch.stack(beta_hat_all) - margin_of_error
upper_bound = torch.stack(beta_hat_all) + margin_of_error

coverage_count = (lower_bound <= analysis_params_true) * (
    analysis_params_true <= upper_bound
)
coverage_ratios = coverage_count.float().mean(0)

plot_analysis_coverage(
    coverage_ratios,
    "../res_folder/simu2-2/analysis" + str(analysis_index) + "_cover.png",
    xaxis_tick_num=analysis_index,
)
