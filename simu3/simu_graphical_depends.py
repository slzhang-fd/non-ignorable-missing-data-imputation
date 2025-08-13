# %%
import torch
from torch.distributions import MultivariateNormal


def MAR_mask(Y, p, q):
    Y_masked = Y.clone()
    for j in range(Y.shape[1] - 1):
        # mask = np.random.rand(Y.shape[0]) < (p[j] * Y[:,-1] + q[j] * (1-Y[:,-1]))
        mask = torch.rand(Y.shape[0]) < (p[j] * Y[:, -1] + q[j] * (1 - Y[:, -1]))
        Y_masked[mask, j] = float("nan")

    return Y_masked


def mean_analysis_model(responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat):
    n_responses = responses_all[0].shape[0]
    responses_ave = sum(responses_all) / len(responses_all)
    beta_hat = responses_ave.mean(0)
    U_i_bar_all = responses_ave - beta_hat
    Omega_c_hat = (responses_ave.T @ responses_ave) / n_responses - torch.outer(beta_hat, beta_hat)
    kappa_hat1 = [(r_ij - beta_hat).T @ (s_ij - S_i_obs) for s_ij, r_ij in zip(S_ij_all1, responses_all)]
    kappa_hat1 = sum(kappa_hat1) / len(kappa_hat1) / n_responses
    temp21 = kappa_hat1 @ D_hat_i_all.T @ U_i_bar_all
    term31 = (temp21 + temp21.T) / n_responses
    Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
    Sigma_hat1 = Omega_hat1
    beta_vars1 = Sigma_hat1.diag() / n_responses

    return beta_hat, beta_vars1


def generate_graphical_mixed_y(S_cont, sigma, S_bin, N, mcmc_len=1000, silent=True):
    ## Because of the non-standard form of prior theta in the model,
    ## we generate data y from a mcmc process that iterative
    ## between sampling y and theta given each other
    ## use Theta_0 from standard normal
    J = S_bin.shape[0]
    ## first generate y_cont from Gaussian graphical model with precision matrix S_cont and measurmement error sigma
    mean = torch.zeros(S_cont.shape[0])
    cov = torch.inverse(S_cont)
    y_cont = MultivariateNormal(mean, cov).sample([N])
    y_cont += torch.randn(N, S_cont.shape[0]) * sigma

    ## generate y_bin from Bernoulli graphical model
    y_bin = (torch.rand((N, J)) < 0.5).int()
    for i in range(mcmc_len):
        if i % 100 == 0 and not silent:
            print("\rIteration: {}".format(i))
        ## sample y_j given y_{-j} and theta
        for j in range(J):
            y_bin_j = y_bin.clone()
            y_bin_j[:, j] = 0.5
            c_j = y_bin_j.to(torch.get_default_dtype()) @ S_bin[:, j]
            y_prob_j = torch.sigmoid(c_j)
            y_bin[:, j] = (torch.rand(N) < y_prob_j).int()

    return torch.cat([y_cont, y_bin], dim=1)
