# %%
import pickle
import sys

import torch

sys.path.append("../")

from simu_graphical_depends import generate_graphical_mixed_y

from depend_funcs import (
    plot_analysis_boxplot,
    plot_analysis_coverage,
)
from depend_funcs_ignorable import (
    cond_mean_analysis_model,
)

params_hat_all = []
params_vars_all = []
params_vars_all1 = []
beta_hat_all = []
beta_sd_all1 = []

for reps in range(100):
    with open("../res_folder/simu3/res_" + str(reps + 1) + ".pkl", "rb") as f:
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

analysis_index = 1

# %% point estimation for the analysis model
##########################################################################################

NN = int(1e6)
with open("params.pkl", "rb") as f:
    S_cont, S_bin = pickle.load(f)

n_cont = S_cont.shape[0]
n_bin = S_bin.shape[0]
n_items = n_cont + n_bin

measurement_sigma = 0.5

responses_ps = generate_graphical_mixed_y(
    S_cont.float(), measurement_sigma, S_bin.float(), NN
)

if analysis_index == 1:
    analysis_params_true = responses_ps.mean(0)
else:
    ## conditional mean
    analysis_params_true = torch.zeros(n_items * 2 - 2)
    analysis_params_true[: (n_items - 1)] = responses_ps[
        responses_ps[:, -1] == 0, :-1
    ].mean(0)
    analysis_params_true[(n_items - 1) :] = (
        responses_ps[responses_ps[:, -1] == 1, :-1].mean(0)
        - analysis_params_true[: (n_items - 1)]
    )

estimated_mean = torch.stack(beta_hat_all).mean(0)

# %% boxplot for analysis point estimation
##########################################################################################
difference = torch.stack(beta_hat_all) - analysis_params_true

plot_analysis_boxplot(
    difference,
    "../res_folder/simu3/analysis" + str(analysis_index) + "_boxplot.png",
)

# %% coverage for the analysis model parameters
##########################################################################################
z_value = 1.96  # for 95% CI with a normal distribution
margin_of_error = z_value * torch.stack(beta_sd_all1).median(0).values
lower_bound = torch.stack(beta_hat_all) - margin_of_error
upper_bound = torch.stack(beta_hat_all) + margin_of_error

coverage_count = (lower_bound <= analysis_params_true) * (
    analysis_params_true <= upper_bound
)
coverage_ratios = coverage_count.float().mean(0)

plot_analysis_coverage(
    coverage_ratios,
    "../res_folder/simu3/analysis" + str(analysis_index) + "_cover.png",
    xaxis_tick_num=analysis_index,
)
