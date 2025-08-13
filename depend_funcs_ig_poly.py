import math

import numpy as np
import torch
from polyagamma import random_polyagamma
from pypolyagamma import PyPolyaGamma
from torch.distributions import MultivariateNormal, Normal
from torch.func import jacfwd
from torch.nn import Parameter

from depend_funcs import MySA_Ruppert1


def MAR_mask(Y, p, q, ref_index=-1):
    Y_masked = Y.clone()
    for j in range(Y.shape[1]):
        if j != ref_index:
            mask = torch.rand(Y.shape[0]) < (
                p[j] * Y[:, ref_index] + q[j] * (1 - Y[:, ref_index])
            )
            Y_masked[mask, j] = float("nan")

    return Y_masked


def my_normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def my_g_func(x_upper, x_lower, mu, omega):
    # assert (omega > 0).all()
    temp_upper = omega.sqrt() * (x_upper - mu)
    temp_lower = omega.sqrt() * (x_lower - mu)
    return my_normal_cdf(temp_upper), my_normal_cdf(temp_lower)


def polya_gamma_sample(h, z, pg=PyPolyaGamma()):
    """Sample from a Polya-Gamma distribution PG(h, z)."""
    z_flat = z.flatten()
    h_flat = np.ones_like(z_flat) * h
    omega = np.empty_like(z_flat)
    pg.pgdrawv(h_flat, z_flat, omega)

    if (omega <= 0).any():
        print("omega min:", omega.min())
        print(omega.shape)
        print(omega[omega <= 0])
        omega = omega.clip(min=1e-2)

    assert (omega > 0).all(), "Omega_ord is not positive"
    return omega.reshape(z.shape)


def trans_B_to_D(B):
    n_ord = B.shape[0]
    D_temp = torch.cat(
        (
            torch.full((n_ord, 1), float("-inf")),
            B[:, 0].unsqueeze(-1),
            B[:, 0].unsqueeze(-1) + B[:, 1:].exp().cumsum(dim=1),
            torch.full((n_ord, 1), float("inf")),
        ),
        dim=1,
    )
    return D_temp


def generate_poly_data(
    x_covs, measurement_sigma, W_alpha, W_beta, B, n_cont, n_bin, n_ord
):
    n_responses, _ = x_covs.shape
    n_items = n_cont + n_bin + n_ord
    n_eta = W_beta.shape[1]
    Mj = B.shape[1]

    continuous_cols = torch.arange(n_cont)
    binary_cols = torch.arange(n_cont, n_cont + n_bin)
    ordinal_cols = torch.arange(n_cont + n_bin, n_items)

    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    ## generate continuous data
    temp_cont = (
        x_covs @ W_alpha[continuous_cols, :].T + eta @ W_beta[continuous_cols, :].T
    )
    responses_cont = temp_cont + torch.randn(n_responses, n_cont) * measurement_sigma

    ## generate binary data
    temp_bin = x_covs @ W_alpha[binary_cols, :].T + eta @ W_beta[binary_cols, :].T
    responses_bin = torch.bernoulli(torch.sigmoid(temp_bin))

    ## generate ordinal data
    D_temp = trans_B_to_D(B)
    # print(D_temp)
    temp_ord = (
        x_covs[:, 1:] @ W_alpha[ordinal_cols, 1:].T + eta @ W_beta[ordinal_cols, :].T
    )
    logits_temp = torch.zeros(n_responses, n_ord, Mj + 2)
    # print(temp_ord.shape)
    # print(D_temp[:, 0].squeeze().shape)
    for k in range(Mj + 2):
        logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

    prob_temp = torch.sigmoid(logits_temp)
    responses_ord = (torch.rand(n_responses, n_ord).unsqueeze(-1) < prob_temp).sum(
        dim=2
    ).type(torch.get_default_dtype()) - 1

    return torch.cat((responses_cont, responses_bin, responses_ord), dim=1)


def generate_nonig_poly_data(
    x_covs,
    measurement_sigma,
    W_alpha,
    W_beta,
    B,
    W_gamma,
    W_zeta,
    W_kappa,
    n_cont,
    n_bin,
    n_ord,
):
    n_responses, _ = x_covs.shape
    n_items = n_cont + n_bin + n_ord
    n_eta = W_beta.shape[1]
    Mj = B.shape[1]

    continuous_cols = torch.arange(n_cont)
    binary_cols = torch.arange(n_cont, n_cont + n_bin)
    ordinal_cols = torch.arange(n_cont + n_bin, n_items)

    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    ## generate continuous data
    temp_cont = (
        x_covs @ W_alpha[continuous_cols, :].T + eta @ W_beta[continuous_cols, :].T
    )
    responses_cont = temp_cont + torch.randn(n_responses, n_cont) * measurement_sigma

    ## generate binary data
    temp_bin = x_covs @ W_alpha[binary_cols, :].T + eta @ W_beta[binary_cols, :].T
    responses_bin = torch.bernoulli(torch.sigmoid(temp_bin))

    ## generate ordinal data
    D_temp = trans_B_to_D(B)
    # print(D_temp)
    temp_ord = (
        x_covs[:, 1:] @ W_alpha[ordinal_cols, 1:].T + eta @ W_beta[ordinal_cols, :].T
    )
    logits_temp = torch.zeros(n_responses, n_ord, Mj + 2)
    # print(temp_ord.shape)
    # print(D_temp[:, 0].squeeze().shape)
    for k in range(Mj + 2):
        logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

    prob_temp = torch.sigmoid(logits_temp)
    responses_ord = (torch.rand(n_responses, n_ord).unsqueeze(-1) < prob_temp).sum(
        dim=2
    ).type(torch.get_default_dtype()) - 1
    responses = torch.cat((responses_cont, responses_bin, responses_ord), dim=1)

    ## generate mask pattern
    _, n_xi = W_zeta.shape
    xi = eta @ W_kappa.T + MultivariateNormal(
        torch.zeros(n_xi), torch.eye(n_xi)
    ).sample((n_responses,))
    temp = x_covs @ W_gamma.T + xi @ W_zeta.T
    mask = torch.rand(n_responses, n_items) < torch.sigmoid(temp)
    responses_masked = torch.where(mask, torch.tensor(float("nan")), responses)
    return responses, responses_masked


def generate_latent_reg_poly_data(
    x_covs,
    x_covs1,
    measurement_sigma,
    W_alpha,
    W_beta,
    B,
    W_gamma,
    W_zeta,
    W_kappa,
    W_coeff_eta,
    W_coeff_xi,
    n_cont,
    n_bin,
    n_ord,
):
    n_responses, _ = x_covs.shape
    n_items = n_cont + n_bin + n_ord
    n_eta = W_beta.shape[1]
    Mj = B.shape[1]

    continuous_cols = torch.arange(n_cont)
    binary_cols = torch.arange(n_cont, n_cont + n_bin)
    ordinal_cols = torch.arange(n_cont + n_bin, n_items)

    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    eta += x_covs1 @ W_coeff_eta.T
    ## generate continuous data
    temp_cont = (
        x_covs @ W_alpha[continuous_cols, :].T + eta @ W_beta[continuous_cols, :].T
    )
    responses_cont = temp_cont + torch.randn(n_responses, n_cont) * measurement_sigma

    ## generate binary data
    temp_bin = x_covs @ W_alpha[binary_cols, :].T + eta @ W_beta[binary_cols, :].T
    responses_bin = torch.bernoulli(torch.sigmoid(temp_bin))

    ## generate ordinal data
    D_temp = trans_B_to_D(B)
    # print(D_temp)
    temp_ord = (
        x_covs[:, 1:] @ W_alpha[ordinal_cols, 1:].T + eta @ W_beta[ordinal_cols, :].T
    )
    logits_temp = torch.zeros(n_responses, n_ord, Mj + 2)
    # print(temp_ord.shape)
    # print(D_temp[:, 0].squeeze().shape)
    for k in range(Mj + 2):
        logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

    prob_temp = torch.sigmoid(logits_temp)
    responses_ord = (torch.rand(n_responses, n_ord).unsqueeze(-1) < prob_temp).sum(
        dim=2
    ).type(torch.get_default_dtype()) - 1
    responses = torch.cat((responses_cont, responses_bin, responses_ord), dim=1)

    ## generate mask pattern
    _, n_xi = W_zeta.shape
    xi = eta @ W_kappa.T + MultivariateNormal(
        torch.zeros(n_xi), torch.eye(n_xi)
    ).sample((n_responses,))
    xi += x_covs1 @ W_coeff_xi.T
    temp = x_covs @ W_gamma.T + xi @ W_zeta.T
    mask = torch.rand(n_responses, n_items) < torch.sigmoid(temp)
    responses_masked = torch.where(mask, torch.tensor(float("nan")), responses)
    return responses, responses_masked


@torch.jit.script
def sample_eta_mixed_poly_jit(
    response_cont,
    response_bin,
    z_ord,
    W_beta_cont,
    W_beta_bin,
    W_beta_ord,
    x_cov_alpha_c,
    x_cov_alpha_b,
    x_cov_alpha_o,
    sigma_sq_inv,
    Omega,
    Omega_ord,
    StandardNormals,
):
    n_responses, _ = response_cont.shape
    _, n_eta = W_beta_cont.shape
    eta = torch.zeros(n_responses, n_eta)
    Sigma_temp = W_beta_cont.T @ torch.diag(sigma_sq_inv) @ W_beta_cont + torch.eye(
        n_eta
    )
    for i in range(n_responses):
        matrix_to_invert = (
            Sigma_temp
            + W_beta_bin.T @ torch.diag(Omega[i, :]) @ W_beta_bin
            + W_beta_ord.T @ torch.diag(Omega_ord[i, :]) @ W_beta_ord
        )
        u = torch.linalg.cholesky(matrix_to_invert)
        Sigma_eta_pos = torch.cholesky_inverse(u)
        L = torch.linalg.cholesky(Sigma_eta_pos)
        mu_eta_pos = Sigma_eta_pos @ (
            W_beta_cont.T
            @ torch.diag(sigma_sq_inv)
            @ (response_cont[i, :].unsqueeze(-1) - x_cov_alpha_c[i, :].unsqueeze(-1))
            - W_beta_bin.T @ torch.diag(Omega[i, :]) @ x_cov_alpha_b[i, :].unsqueeze(-1)
            + W_beta_bin.T @ (response_bin[i, :].unsqueeze(-1) - 0.5)
            + W_beta_ord.T
            @ torch.diag(Omega_ord[i, :])
            @ (z_ord[i, :].unsqueeze(-1) - x_cov_alpha_o[i, :].unsqueeze(-1))
        )
        eta[i, :] = mu_eta_pos.squeeze() + StandardNormals[i, :] @ L.T

    return eta.nan_to_num(posinf=0.0, neginf=0.0)


@torch.jit.script
def sample_eta_nonig_poly_jit(
    response_cont,
    response_bin,
    z_ord,
    W_beta_cont,
    W_beta_bin,
    W_beta_ord,
    x_cov_alpha_c,
    x_cov_alpha_b,
    x_cov_alpha_o,
    sigma_sq_inv,
    W_kappa,
    xi,
    Omega,
    Omega_ord,
    StandardNormals,
):
    n_responses, _ = response_cont.shape
    _, n_eta = W_beta_cont.shape
    eta = torch.zeros(n_responses, n_eta)
    Sigma_temp = (
        W_beta_cont.T @ torch.diag(sigma_sq_inv) @ W_beta_cont
        + torch.eye(n_eta)
        + W_kappa.T @ W_kappa
    )
    for i in range(n_responses):
        matrix_to_invert = (
            Sigma_temp
            + W_beta_bin.T @ torch.diag(Omega[i, :]) @ W_beta_bin
            + W_beta_ord.T @ torch.diag(Omega_ord[i, :]) @ W_beta_ord
        )
        u = torch.linalg.cholesky(matrix_to_invert)
        Sigma_eta_pos = torch.cholesky_inverse(u)
        L = torch.linalg.cholesky(Sigma_eta_pos)
        mu_eta_pos = Sigma_eta_pos @ (
            W_beta_cont.T
            @ torch.diag(sigma_sq_inv)
            @ (response_cont[i, :].unsqueeze(-1) - x_cov_alpha_c[i, :].unsqueeze(-1))
            - W_beta_bin.T @ torch.diag(Omega[i, :]) @ x_cov_alpha_b[i, :].unsqueeze(-1)
            + W_beta_bin.T @ (response_bin[i, :].unsqueeze(-1) - 0.5)
            + W_beta_ord.T
            @ torch.diag(Omega_ord[i, :])
            @ (z_ord[i, :].unsqueeze(-1) - x_cov_alpha_o[i, :].unsqueeze(-1))
            + W_kappa.T @ xi[i, :].unsqueeze(-1)
        )
        eta[i, :] = mu_eta_pos.squeeze() + StandardNormals[i, :] @ L.T

    # return eta.clamp(-4, 4).nan_to_num(posinf=0.0, neginf=0.0)
    return eta.nan_to_num(posinf=0.0, neginf=0.0)


@torch.jit.script
def sample_eta_nonig_poly_reg_jit(
    response_cont,
    response_bin,
    z_ord,
    W_beta_cont,
    W_beta_bin,
    W_beta_ord,
    x_cov_alpha_c,
    x_cov_alpha_b,
    x_cov_alpha_o,
    sigma_sq_inv,
    W_kappa,
    xi_centered,
    fixed_effects,
    Omega,
    Omega_ord,
    StandardNormals,
):
    n_responses, _ = response_cont.shape
    _, n_eta = W_beta_cont.shape
    eta = torch.zeros(n_responses, n_eta)
    Sigma_temp = (
        W_beta_cont.T @ torch.diag(sigma_sq_inv) @ W_beta_cont
        + torch.eye(n_eta)
        + W_kappa.T @ W_kappa
    )
    for i in range(n_responses):
        matrix_to_invert = (
            Sigma_temp
            + W_beta_bin.T @ torch.diag(Omega[i, :]) @ W_beta_bin
            + W_beta_ord.T @ torch.diag(Omega_ord[i, :]) @ W_beta_ord
        )
        u = torch.linalg.cholesky(matrix_to_invert)
        Sigma_eta_pos = torch.cholesky_inverse(u)
        L = torch.linalg.cholesky(Sigma_eta_pos)
        mu_eta_pos = Sigma_eta_pos @ (
            W_beta_cont.T
            @ torch.diag(sigma_sq_inv)
            @ (response_cont[i, :].unsqueeze(-1) - x_cov_alpha_c[i, :].unsqueeze(-1))
            - W_beta_bin.T @ torch.diag(Omega[i, :]) @ x_cov_alpha_b[i, :].unsqueeze(-1)
            + W_beta_bin.T @ (response_bin[i, :].unsqueeze(-1) - 0.5)
            + W_beta_ord.T
            @ torch.diag(Omega_ord[i, :])
            @ (z_ord[i, :].unsqueeze(-1) - x_cov_alpha_o[i, :].unsqueeze(-1))
            + W_kappa.T @ xi_centered[i, :].unsqueeze(-1)
            + fixed_effects[i, :].unsqueeze(-1)
        )
        eta[i, :] = mu_eta_pos.squeeze() + StandardNormals[i, :] @ L.T

    # return eta.clamp(-4, 4).nan_to_num(posinf=0.0, neginf=0.0)
    return eta.nan_to_num(posinf=0.0, neginf=0.0)


@torch.jit.script
def sample_xi_combined_jit(
    z_variables, W_zeta, x_covs_gamma, eta_kappa, Omega, StandardNormals
):
    n_responses, _ = z_variables.shape
    _, n_xi = W_zeta.shape
    xi = torch.zeros(n_responses, n_xi)
    mu_temp = eta_kappa.T + W_zeta.T @ (z_variables - 0.5).T
    for i in range(n_responses):
        matrix_to_invert = W_zeta.T @ torch.diag(Omega[i, :]) @ W_zeta + torch.eye(n_xi)
        u = torch.linalg.cholesky(matrix_to_invert)
        Sigma_xi_pos = torch.cholesky_inverse(u)
        L = torch.linalg.cholesky(Sigma_xi_pos)
        mu_xi_pos = Sigma_xi_pos @ (
            # eta_kappa[i, :].unsqueeze(-1)
            # + W_zeta.T @ (z_variables[i, :].unsqueeze(-1) - 0.5)
            mu_temp[:, i].unsqueeze(-1)
            - W_zeta.T @ torch.diag(Omega[i, :]) @ x_covs_gamma[i, :].unsqueeze(-1)
        )
        xi[i, :] = mu_xi_pos.squeeze() + StandardNormals[i, :] @ L.T

    # return xi.clamp(-4, 4).nan_to_num(posinf=0.0, neginf=0.0)
    return xi.nan_to_num(posinf=0.0, neginf=0.0)


@torch.jit.script
def sample_xi_combined_reg_jit(
    z_variables, W_zeta, x_covs_gamma, eta_kappa, fixed_effects, Omega, StandardNormals
):
    n_responses, _ = z_variables.shape
    _, n_xi = W_zeta.shape
    xi = torch.zeros(n_responses, n_xi)
    mu_temp = eta_kappa.T + W_zeta.T @ (z_variables - 0.5).T
    for i in range(n_responses):
        matrix_to_invert = W_zeta.T @ torch.diag(Omega[i, :]) @ W_zeta + torch.eye(n_xi)
        u = torch.linalg.cholesky(matrix_to_invert)
        Sigma_xi_pos = torch.cholesky_inverse(u)
        L = torch.linalg.cholesky(Sigma_xi_pos)
        mu_xi_pos = Sigma_xi_pos @ (
            # eta_kappa[i, :].unsqueeze(-1)
            # + W_zeta.T @ (z_variables[i, :].unsqueeze(-1) - 0.5)
            mu_temp[:, i].unsqueeze(-1)
            - W_zeta.T @ torch.diag(Omega[i, :]) @ x_covs_gamma[i, :].unsqueeze(-1)
            + fixed_effects[i, :].unsqueeze(-1)
        )
        xi[i, :] = mu_xi_pos.squeeze() + StandardNormals[i, :] @ L.T

    # return xi.clamp(-4, 4).nan_to_num(posinf=0.0, neginf=0.0)
    return xi.nan_to_num(posinf=0.0, neginf=0.0)


## partial credit model to handle ordinal data
class IgnorableImputerInfer_poly:
    def __init__(self, responses, x_covs, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = initial_values["continuous_cols"]
        self.binary_cols = initial_values["binary_cols"]
        self.ordinal_cols = initial_values["ordinal_cols"]
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if col in self.binary_cols:
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif col in self.ordinal_cols:
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.log_sigma = Parameter(initial_values["log_sigma"])
        self.B = Parameter(initial_values["B"])
        self.Mj = self.B.shape[1]
        self.n_eta = self.W_beta.shape[1]
        ## split intercept parameters out
        # self.alpha0 = Parameter(torch.zeros(self.n_items))

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)
        self.pg = PyPolyaGamma(seed=1)
        self.losses = []

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            # random_polyagamma(torch.ones(temp.shape), temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            # random_polyagamma(torch.full_like(temp_ord, 2), temp_ord - self.z_ord),
            polya_gamma_sample(h=2, z=(temp_ord - self.z_ord).numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_mixed_poly_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            Omega,
            Omega_ord,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y_cont | eta)
        loglik = 0.0
        temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        if self.n_cont > 0:
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
                )
                .log_prob(
                    self.responses[:, self.continuous_cols]
                    - temp[:, self.continuous_cols]
                )
                .sum()
            )
        # log p(y_bin | eta)
        if self.n_bin > 0:
            loglik += (
                self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
                - torch.log1p(torch.exp(temp[:, self.binary_cols]))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(self.B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik / self.n_responses

    def fit(self, optimizer_choice="SGD", max_iter=200, lr=0.1, alpha=0.51):
        if optimizer_choice == "SGD":
            optimizer = torch.optim.SGD(
                [self.log_sigma, self.W_alpha, self.W_beta, self.B], lr=lr
            )
        elif optimizer_choice == "Adam":
            optimizer = torch.optim.Adam(
                [self.log_sigma, self.W_alpha, self.W_beta, self.B], lr=lr
            )
        elif optimizer_choice == "MySA_Ruppert":
            optimizer = MySA_Ruppert1(
                [self.log_sigma, self.W_alpha, self.W_beta, self.B],
                lr=lr,
                alpha=alpha,
                init_steps=(int)(max_iter / 2),
            )
        else:
            raise ValueError("optimizer_choice must be SGD or MySA_Ruppert")

        for i in range(max_iter):
            # Sample x_factors from the posterior distribution
            with torch.no_grad():
                self.sample_eta()
                self.impute()

            # Forward pass: compute the mean and log variance of the factors
            log_lik = self.calcu_loglik()

            # Compute the loss: negative log likelihood
            loss = -log_lik

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

            # Print the loss
            self.losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{max_iter}, Loss: {loss.item()}")


class IgnorableImputerInfer_poly_stand_alone:
    def __init__(self, responses, n_eta, x_covs, params_hat_all):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = []
        self.binary_cols = []
        self.ordinal_cols = []
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if (
                len(unique_values) == 2
                and (0 in unique_values)
                and (1 in unique_values)
            ):
                self.binary_cols.append(col)
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif len(unique_values) > 2 and len(unique_values) <= 10:
                self.ordinal_cols.append(col)
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                self.continuous_cols.append(col)
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs

        self.W_alpha = params_hat_all["W_alpha"]
        self.w_alpha_mask = torch.ones(self.W_alpha.shape).bool()
        self.w_alpha_mask[-self.n_ord :, 0] = False
        self.W_beta = params_hat_all["W_beta"]
        self.w_beta_mask = torch.ones(self.W_beta.shape).tril().bool()
        self.log_sigma = params_hat_all["log_sigma"]
        self.B = params_hat_all["B"]
        self.Mj = self.B.shape[1]

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            random_polyagamma(torch.full_like(temp_ord, 2), temp_ord - self.z_ord),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_mixed_poly_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            Omega,
            Omega_ord,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self, log_sigma, W_alpha_reduced, W_beta_reduced, B):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        # log p(y_cont | eta)
        loglik = 0.0
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
        )
        with torch.no_grad():
            jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(self, log_sigma, W_alpha_reduced, W_beta_reduced, B):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        # log p(y_cont | eta)
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        loglik_vec = torch.zeros(self.n_responses)
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik_vec += MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            ).log_prob(self.responses[:, self.continuous_cols] - temp_cont)
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik_vec += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum(1)
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik_vec += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum(1)
        # log p(eta)
        loglik_vec += Normal(0, 1).log_prob(self.eta).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        with torch.no_grad():
            jaco_efficient = jacfwd(self.calcu_loglik_vec, argnums=(0, 1, 2, 3))(
                self.log_sigma,
                self.W_alpha[self.w_alpha_mask],
                self.W_beta[self.w_beta_mask],
                self.B,
            )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        log_sigma, W_alpha_reduced, W_beta_reduced, B = inputs_vec.split(
            [
                self.n_cont,
                self.w_alpha_mask.sum(),
                self.w_beta_mask.sum(),
                self.n_ord * self.Mj,
            ]
        )
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced

        # log p(y_cont | eta)
        loglik = 0.0
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B.view(self.n_ord, self.Mj))
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
        )
        with torch.no_grad():
            hessian_tuples = torch.autograd.functional.hessian(
                self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
            )
        return hessian_tuples

    def infer(self, mis_copies=3, M=200, thinning=1, thinning2=5):
        params_num = (
            self.log_sigma.numel()
            + self.w_alpha_mask.sum()
            + self.w_beta_mask.sum()
            + self.n_ord * self.Mj
        )
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_i_obs = torch.zeros(self.n_responses, params_num)

        k_count = 1.0
        for i in range(M):
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{M}")

            for j in range(thinning):
                with torch.no_grad():
                    self.sample_eta()
                    self.impute()

            factor = 1.0
            jacobian_temp = self.calcu_jacobian()
            hessian_temp = self.calcu_hessian()
            res1 += factor * (
                -hessian_temp - torch.outer(jacobian_temp, jacobian_temp) - res1
            )
            res2 += factor * (jacobian_temp - res2)
            mis_info1_ave = mis_info1_ave + (res1 - mis_info1_ave) / k_count
            mis_info2_ave = mis_info2_ave + (res2 - mis_info2_ave) / k_count

            S_i_temp = self.calcu_jacobian_per_sample()
            S_per_sample += factor * (S_i_temp - S_per_sample)
            S_i_obs = S_i_obs + (S_per_sample - S_i_obs) / k_count
            k_count += 1.0

        responses_all = []
        S_ij_all1 = []

        for i in range(mis_copies):
            with torch.no_grad():
                self.impute()
            responses_all.append(self.responses.clone())

            # default thinning for each data copy
            for j in range(thinning2):
                with torch.no_grad():
                    self.sample_eta()

            S_ij_latent = self.calcu_jacobian_per_sample()
            S_ij_all1.append(S_ij_latent.clone())

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        I_obs = (
            I_obs + I_obs.T
        ) / 2  ## make sure it is symmetric to avoid numerical issues
        I_obs_inv = I_obs.inverse()
        D_hat_i_all = S_i_obs @ I_obs_inv * self.n_responses
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        return responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, I_obs_inv


class NonIgnorableImputerInfer_poly:
    def __init__(self, responses, x_covs, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.z_variables = self.missing_indices.double()
        self.n_responses, self.n_items = responses.shape
        # self.n_eta = n_eta
        # self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = initial_values["continuous_cols"]
        self.binary_cols = initial_values["binary_cols"]
        self.ordinal_cols = initial_values["ordinal_cols"]
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if col in self.binary_cols:
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif col in self.ordinal_cols:
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.log_sigma = Parameter(initial_values["log_sigma"])
        self.B = Parameter(initial_values["B"])
        self.Mj = self.B.shape[1]
        self.W_gamma = Parameter(initial_values["W_gamma"])
        self.W_zeta = Parameter(initial_values["W_zeta"])
        self.W_kappa = Parameter(initial_values["W_kappa"])
        self.n_xi, self.n_eta = self.W_kappa.shape
        ## split intercept parameters out
        # self.alpha0 = Parameter(torch.zeros(self.n_items))

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)
        self.pg = PyPolyaGamma(seed=1)
        self.losses = []

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            # random_polyagamma(h=2, z=temp_ord - self.z_ord),
            polya_gamma_sample(h=2, z=(temp_ord - self.z_ord).numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_nonig_poly_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            self.W_kappa,
            self.xi,
            Omega,
            Omega_ord,
            StandNormals,
        )

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_jit(
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.W_kappa.T,
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y_cont | eta)
        loglik = 0.0
        temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        # print("xi min max:" + str(self.xi.min()) + str(self.xi.max()) +
        #       " eta min max:" + str(self.eta.min()) + str(self.eta.max()) +
        #       " Y_min:" + str(self.responses[:, self.continuous_cols].min(0))+
        #       " Y_max:" + str(self.responses[:, self.continuous_cols].max(0))+
        #       " W_alpha:" + str(self.W_alpha.data)+
        #       " W_beta:" + str(self.W_beta.data))
        if self.n_cont > 0:
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
                )
                .log_prob(
                    self.responses[:, self.continuous_cols]
                    - temp[:, self.continuous_cols]
                )
                .sum()
            )
            # loglik += Normal(torch.zeros(self.n_cont), self.log_sigma.exp()).log_prob(
            #     self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
            # ).sum()
        # log p(y_bin | eta)
        if self.n_bin > 0:
            loglik += (
                self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
                - torch.log1p(torch.exp(temp[:, self.binary_cols]))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(self.B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += Normal(0, 1).log_prob(self.xi - self.eta @ self.W_kappa.T).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik / self.n_responses

    def fit(self, optimizer_choice="SGD", max_iter=200, lr=0.1, alpha=0.51):
        if optimizer_choice == "SGD":
            optimizer = torch.optim.SGD(
                [
                    self.log_sigma,
                    self.W_alpha,
                    self.W_beta,
                    self.B,
                    self.W_gamma,
                    self.W_zeta,
                    self.W_kappa,
                ],
                lr=lr,
            )
        elif optimizer_choice == "Adam":
            optimizer = torch.optim.Adam(
                [
                    self.log_sigma,
                    self.W_alpha,
                    self.W_beta,
                    self.B,
                    self.W_gamma,
                    self.W_zeta,
                    self.W_kappa,
                ],
                lr=lr,
            )
        elif optimizer_choice == "MySA_Ruppert":
            optimizer = MySA_Ruppert1(
                [
                    self.log_sigma,
                    self.W_alpha,
                    self.W_beta,
                    self.B,
                    self.W_gamma,
                    self.W_zeta,
                    self.W_kappa,
                ],
                lr=lr,
                alpha=alpha,
                init_steps=(int)(max_iter / 2),
            )
        else:
            raise ValueError("optimizer_choice must be SGD or MySA_Ruppert")

        for i in range(max_iter):
            # Sample x_factors from the posterior distribution
            with torch.no_grad():
                self.sample_eta()
                self.sample_xi()
                self.impute()

            # Forward pass: compute the mean and log variance of the factors
            log_lik = self.calcu_loglik()

            # Compute the loss: negative log likelihood
            loss = -log_lik

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

            # Print the loss
            self.losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{max_iter}, Loss: {loss.item()}")


class NonIgnorableImputerInfer_poly_stand_alone:
    def __init__(self, responses, x_covs, params_hat_all):
        self.missing_indices = torch.isnan(responses)
        self.z_variables = self.missing_indices.double()
        self.n_responses, self.n_items = responses.shape
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = params_hat_all["continuous_cols"]
        self.binary_cols = params_hat_all["binary_cols"]
        self.ordinal_cols = params_hat_all["ordinal_cols"]
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if col in self.binary_cols:
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif col in self.ordinal_cols:
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        print(
            "continuous cols:"
            + str(self.continuous_cols)
            + ", binary cols:"
            + str(self.binary_cols)
            + ", ordinal cols:"
            + str(self.ordinal_cols)
        )
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs

        self.W_alpha = params_hat_all["W_alpha"]
        self.w_alpha_mask = torch.ones(self.W_alpha.shape).bool()
        self.w_alpha_mask[-self.n_ord :, 0] = False
        self.B = params_hat_all["B"]
        self.W_beta = params_hat_all["W_beta"]
        self.w_beta_mask = torch.ones(self.W_beta.shape).tril().bool()
        self.log_sigma = params_hat_all["log_sigma"]
        self.Mj = self.B.shape[1]
        self.W_gamma = params_hat_all["W_gamma"]
        self.W_zeta = params_hat_all["W_zeta"]
        self.w_zeta_mask = torch.ones(self.W_zeta.shape).tril().bool()
        self.W_kappa = params_hat_all["W_kappa"]
        self.n_xi, self.n_eta = self.W_kappa.shape

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)
        self.pg = PyPolyaGamma(seed=1)

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp, method="saddle"),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            # random_polyagamma(h=2, z=temp_ord - self.z_ord, method="saddle"),
            polya_gamma_sample(h=2, z=(temp_ord - self.z_ord).numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_nonig_poly_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            self.W_kappa,
            self.xi,
            Omega,
            Omega_ord,
            StandNormals,
        )

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp, method="saddle"),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_jit(
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.W_kappa.T,
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(
        self,
        log_sigma,
        W_alpha_reduced,
        W_beta_reduced,
        B,
        W_gamma,
        W_zeta_reduced,
        W_kappa,
    ):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        loglik = 0.0
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
            # loglik += Normal(torch.zeros(self.n_cont), log_sigma.exp()).log_prob(
            #     self.responses[:, self.continuous_cols] - temp_cont
            # ).sum()
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = self.x_covs @ W_gamma.T + self.xi @ W_zeta.tril().T
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += Normal(0, 1).log_prob(self.xi - self.eta @ W_kappa.T).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.W_kappa,
        )
        with torch.no_grad():
            jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(
        self,
        log_sigma,
        W_alpha_reduced,
        W_beta_reduced,
        B,
        W_gamma,
        W_zeta_reduced,
        W_kappa,
    ):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        loglik_vec = torch.zeros(self.n_responses)
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik_vec += MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            ).log_prob(self.responses[:, self.continuous_cols] - temp_cont)
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik_vec += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum(1)
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik_vec += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum(1)
        # log p(z | xi)
        temp_z = self.x_covs @ W_gamma.T + self.xi @ W_zeta.tril().T
        loglik_vec += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum(
            1
        )
        # log p(xi | eta)
        loglik_vec += Normal(0, 1).log_prob(self.xi - self.eta @ W_kappa.T).sum(1)
        # log p(eta)
        loglik_vec += Normal(0, 1).log_prob(self.eta).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        with torch.no_grad():
            jaco_efficient = jacfwd(
                self.calcu_loglik_vec, argnums=(0, 1, 2, 3, 4, 5, 6)
            )(
                self.log_sigma,
                self.W_alpha[self.w_alpha_mask],
                self.W_beta[self.w_beta_mask],
                self.B,
                self.W_gamma,
                self.W_zeta[self.w_zeta_mask],
                self.W_kappa,
            )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        (
            log_sigma,
            W_alpha_reduced,
            W_beta_reduced,
            B,
            W_gamma,
            W_zeta_reduced,
            W_kappa,
        ) = inputs_vec.split(
            [
                self.n_cont,
                self.w_alpha_mask.sum(),
                self.w_beta_mask.sum(),
                self.n_ord * self.Mj,
                self.n_items * self.n_fixed_effects,
                self.w_zeta_mask.sum(),
                self.n_xi * self.n_eta,
            ]
        )
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        loglik = 0.0
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B.view(self.n_ord, self.Mj))
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = (
            self.x_covs @ W_gamma.view(self.n_items, self.n_fixed_effects).T
            + self.xi @ W_zeta.tril().T
        )
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1)
            .log_prob(self.xi - self.eta @ W_kappa.view(self.n_xi, self.n_eta).T)
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.W_kappa,
        )
        with torch.no_grad():
            hessian_tuples = torch.autograd.functional.hessian(
                self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
            )
        return hessian_tuples

    def infer(self, mis_copies=3, M=200, thinning=1, thinning2=5):
        params_num = (
            self.log_sigma.numel()
            + self.w_alpha_mask.sum()
            + self.w_beta_mask.sum()
            + self.n_ord * self.Mj
            + self.W_gamma.numel()
            + self.w_zeta_mask.sum()
            + self.W_kappa.numel()
        )
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_i_obs = torch.zeros(self.n_responses, params_num)

        k_count = 1.0
        for i in range(M):
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{M}")

            for j in range(thinning):
                with torch.no_grad():
                    self.sample_eta()
                    self.sample_xi()
                    self.impute()

            factor = 1.0
            jacobian_temp = self.calcu_jacobian()
            hessian_temp = self.calcu_hessian()
            res1 += factor * (
                -hessian_temp - torch.outer(jacobian_temp, jacobian_temp) - res1
            )
            res2 += factor * (jacobian_temp - res2)
            mis_info1_ave = mis_info1_ave + (res1 - mis_info1_ave) / k_count
            mis_info2_ave = mis_info2_ave + (res2 - mis_info2_ave) / k_count

            S_i_temp = self.calcu_jacobian_per_sample()
            S_per_sample += factor * (S_i_temp - S_per_sample)
            S_i_obs = S_i_obs + (S_per_sample - S_i_obs) / k_count
            k_count += 1.0

        responses_all = []
        S_ij_all1 = []

        for i in range(mis_copies):
            with torch.no_grad():
                self.impute()
            responses_all.append(self.responses.clone())

            # default thinning for each data copy
            for j in range(thinning2):
                with torch.no_grad():
                    self.sample_eta()
                    self.sample_xi()

            S_ij_latent = self.calcu_jacobian_per_sample()
            S_ij_all1.append(S_ij_latent.clone())

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        I_obs = (
            I_obs + I_obs.T
        ) / 2  ## make sure it is symmetric to avoid numerical issues
        I_obs_inv = I_obs.inverse()
        D_hat_i_all = S_i_obs @ I_obs_inv * self.n_responses
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        return responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, I_obs_inv


class NonIgnorableImputerInfer_poly_reg:
    def __init__(self, responses, x_covs, x_covs1, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.z_variables = self.missing_indices.double()
        self.n_responses, self.n_items = responses.shape
        # self.n_eta = n_eta
        # self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = initial_values["continuous_cols"]
        self.binary_cols = initial_values["binary_cols"]
        self.ordinal_cols = initial_values["ordinal_cols"]
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if col in self.binary_cols:
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif col in self.ordinal_cols:
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs
        self.x_covs1 = x_covs1  # added 0

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.log_sigma = Parameter(initial_values["log_sigma"])
        self.B = Parameter(initial_values["B"])
        self.Mj = self.B.shape[1]
        self.W_gamma = Parameter(initial_values["W_gamma"])
        self.W_zeta = Parameter(initial_values["W_zeta"])
        self.W_kappa = Parameter(initial_values["W_kappa"])
        self.W_coeff_eta = Parameter(
            initial_values["W_coeff_eta"]
        )  # added 1, K_eta * p
        self.W_coeff_xi = Parameter(initial_values["W_coeff_xi"])  # added 2, K_xi * p
        self.n_xi, self.n_eta = self.W_kappa.shape
        ## split intercept parameters out
        # self.alpha0 = Parameter(torch.zeros(self.n_items))

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)
        self.pg = PyPolyaGamma(seed=1)
        self.losses = []

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            # random_polyagamma(h=2, z=temp_ord - self.z_ord),
            polya_gamma_sample(h=2, z=(temp_ord - self.z_ord).numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_nonig_poly_reg_jit(  # modified 1
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            self.W_kappa,
            self.xi - self.x_covs1 @ self.W_coeff_xi.T,  # modified 2
            self.x_covs1 @ self.W_coeff_eta.T,  # added 3
            Omega,
            Omega_ord,
            StandNormals,
        )

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_reg_jit(  # modified
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.W_kappa.T,
            self.x_covs1 @ self.W_coeff_xi.T,  # added 4
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y_cont | eta)
        loglik = 0.0
        temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        # print("xi min max:" + str(self.xi.min()) + str(self.xi.max()) +
        #       " eta min max:" + str(self.eta.min()) + str(self.eta.max()) +
        #       " Y_min:" + str(self.responses[:, self.continuous_cols].min(0))+
        #       " Y_max:" + str(self.responses[:, self.continuous_cols].max(0))+
        #       " W_alpha:" + str(self.W_alpha.data)+
        #       " W_beta:" + str(self.W_beta.data))
        if self.n_cont > 0:
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
                )
                .log_prob(
                    self.responses[:, self.continuous_cols]
                    - temp[:, self.continuous_cols]
                )
                .sum()
            )
            # loglik += Normal(torch.zeros(self.n_cont), self.log_sigma.exp()).log_prob(
            #     self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
            # ).sum()
        # log p(y_bin | eta)
        if self.n_bin > 0:
            loglik += (
                self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
                - torch.log1p(torch.exp(temp[:, self.binary_cols]))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(self.B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1)
            .log_prob(
                self.xi - self.x_covs1 @ self.W_coeff_xi.T - self.eta @ self.W_kappa.T
            )
            .sum()
        )
        # log p(eta)
        loglik += (
            Normal(0, 1).log_prob(self.eta - self.x_covs1 @ self.W_coeff_eta.T).sum()
        )
        return loglik / self.n_responses

    def fit(self, optimizer_choice="Adam", max_iter=3000, lr=0.1, fix_kappa=False):
        if fix_kappa:
            if optimizer_choice == "Adam":
                optimizer = torch.optim.Adam(
                    [
                        self.log_sigma,
                        self.W_alpha,
                        self.W_beta,
                        self.B,
                        self.W_gamma,
                        self.W_zeta,
                        self.W_coeff_eta,
                        self.W_coeff_xi,
                    ],
                    lr=lr,
                )
            elif optimizer_choice == "MySA_Ruppert":
                optimizer = MySA_Ruppert1(
                    [
                        self.log_sigma,
                        self.W_alpha,
                        self.W_beta,
                        self.B,
                        self.W_gamma,
                        self.W_zeta,
                        self.W_coeff_eta,
                        self.W_coeff_xi,
                    ],
                    lr=lr,
                    alpha=0.51,
                    burn_in=1000,
                )
            else:
                raise ValueError("optimizer_choice must be SGD or MySA_Ruppert")
        else:
            if optimizer_choice == "Adam":
                optimizer = torch.optim.Adam(
                    [
                        self.log_sigma,
                        self.W_alpha,
                        self.W_beta,
                        self.B,
                        self.W_gamma,
                        self.W_zeta,
                        self.W_kappa,
                        self.W_coeff_eta,
                        self.W_coeff_xi,
                    ],
                    lr=lr,
                )
            elif optimizer_choice == "MySA_Ruppert":
                optimizer = MySA_Ruppert1(
                    [
                        self.log_sigma,
                        self.W_alpha,
                        self.W_beta,
                        self.B,
                        self.W_gamma,
                        self.W_zeta,
                        self.W_kappa,
                        self.W_coeff_eta,
                        self.W_coeff_xi,
                    ],
                    lr=lr,
                    alpha=0.51,
                    burn_in=1000,
                )
            else:
                raise ValueError("optimizer_choice must be SGD or MySA_Ruppert")

        for i in range(max_iter):
            # Sample x_factors from the posterior distribution
            with torch.no_grad():
                self.sample_eta()
                self.sample_xi()
                self.impute()

            # Forward pass: compute the mean and log variance of the factors
            log_lik = self.calcu_loglik()

            # Compute the loss: negative log likelihood
            loss = -log_lik

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

            # Print the loss
            self.losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{max_iter}, Loss: {loss.item()}")


class NonIgnorableImputerInfer_poly_reg_stand_alone:
    def __init__(self, responses, x_covs, x_covs1, params_hat_all):
        self.missing_indices = torch.isnan(responses)
        self.z_variables = self.missing_indices.double()
        self.n_responses, self.n_items = responses.shape
        self.n_fixed_effects = x_covs.shape[1]
        self.n_fixed_effects1 = x_covs1.shape[1]  # added 0
        self.continuous_cols = params_hat_all["continuous_cols"]
        self.binary_cols = params_hat_all["binary_cols"]
        self.ordinal_cols = params_hat_all["ordinal_cols"]
        ## split according to data types and initialize missing values
        for col in range(self.n_items):
            unique_values = torch.unique(
                responses[~torch.isnan(responses[:, col]), col]
            )
            if col in self.binary_cols:
                prob = unique_values.mean()
                responses[torch.isnan(responses[:, col]), col] = torch.bernoulli(
                    prob * torch.ones((torch.isnan(responses[:, col]).sum(),))
                )
            elif col in self.ordinal_cols:
                responses[torch.isnan(responses[:, col]), col] = torch.randint(
                    high=len(unique_values) - 1,
                    size=(torch.isnan(responses[:, col]).sum(),),
                ).type(torch.get_default_dtype())
            else:
                mean_val = unique_values.mean()
                std_val = unique_values.std()
                responses[torch.isnan(responses[:, col]), col] = torch.normal(
                    mean=mean_val,
                    std=std_val,
                    size=(torch.isnan(responses[:, col]).sum(),),
                )
        print(
            "continuous cols:"
            + str(self.continuous_cols)
            + ", binary cols:"
            + str(self.binary_cols)
            + ", ordinal cols:"
            + str(self.ordinal_cols)
        )
        self.x_covs1 = x_covs1  # added 0
        self.responses = responses
        self.n_cont = len(self.continuous_cols)
        self.n_bin = len(self.binary_cols)
        self.n_ord = len(self.ordinal_cols)
        self.x_covs = x_covs

        self.W_alpha = params_hat_all["W_alpha"]
        self.w_alpha_mask = torch.ones(self.W_alpha.shape).bool()
        self.w_alpha_mask[-self.n_ord :, 0] = False
        self.B = params_hat_all["B"]
        self.W_beta = params_hat_all["W_beta"]
        self.w_beta_mask = torch.ones(self.W_beta.shape).tril().bool()
        self.log_sigma = params_hat_all["log_sigma"]
        self.Mj = self.B.shape[1]
        self.W_gamma = params_hat_all["W_gamma"]
        self.W_zeta = params_hat_all["W_zeta"]
        self.w_zeta_mask = torch.ones(self.W_zeta.shape).tril().bool()
        self.W_kappa = params_hat_all["W_kappa"]
        self.n_xi, self.n_eta = self.W_kappa.shape

        self.W_coeff_eta = params_hat_all["W_coeff_eta"]  # added 1
        self.W_coeff_xi = params_hat_all["W_coeff_xi"]  # added 2

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.z_ord = torch.randn(self.n_responses, self.n_ord)
        self.pg = PyPolyaGamma(seed=1)

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta[self.ordinal_cols, :].T
        )
        Omega_ord = torch.tensor(
            # random_polyagamma(h=2, z=temp_ord - self.z_ord),
            polya_gamma_sample(h=2, z=(temp_ord - self.z_ord).numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        ## sample z_ord latent variables from truncated normal
        D_temp = trans_B_to_D(self.B)
        D_upper = torch.zeros(self.n_responses, self.n_ord)
        D_lower = torch.zeros(self.n_responses, self.n_ord)
        for j in range(self.n_ord):
            D_upper[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int() + 1]
            D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]
        GD_upper, GD_lower = my_g_func(D_upper, D_lower, temp_ord, Omega_ord)
        self.z_ord = (
            Normal(0, 1).icdf(
                (GD_upper - GD_lower) * torch.rand(self.n_responses, self.n_ord)
                + GD_lower
            )
            * (1.0 / Omega_ord.sqrt())
            + temp_ord
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_nonig_poly_reg_jit(  # modified 1
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.z_ord,
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.W_beta[self.ordinal_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T,
            1.0 / self.log_sigma.exp().pow(2),
            self.W_kappa,
            self.xi - self.x_covs1 @ self.W_coeff_xi.T,  # modified 2
            self.x_covs1 @ self.W_coeff_eta.T,  # added 3
            Omega,
            Omega_ord,
            StandNormals,
        )

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            # random_polyagamma(h=1, z=temp),
            polya_gamma_sample(h=1, z=temp.numpy(), pg=self.pg),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_reg_jit(  # modified
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.W_kappa.T,
            self.x_covs1 @ self.W_coeff_xi.T,  # added 4
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        ## generate ordinal data from graded response model
        D_temp = trans_B_to_D(self.B)
        temp_ord = (
            self.x_covs[:, 1:] @ self.W_alpha[self.ordinal_cols, 1:].T
            + self.eta @ self.W_beta.tril()[self.ordinal_cols, :].T
        )
        logits_temp = torch.zeros(self.n_responses, self.n_ord, self.Mj + 2)
        for k in range(self.Mj + 2):
            logits_temp[:, :, k] = temp_ord - D_temp[:, k].squeeze()

        prob_temp = torch.sigmoid(logits_temp)
        response_temp[:, self.ordinal_cols] = (
            torch.rand(self.n_responses, self.n_ord).unsqueeze(-1) < prob_temp
        ).sum(dim=2).type(torch.get_default_dtype()) - 1
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(
        self,
        log_sigma,
        W_alpha_reduced,
        W_beta_reduced,
        B,
        W_gamma,
        W_zeta_reduced,
        W_kappa,
        W_coeff_eta,
        W_coeff_xi,
    ):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        loglik = 0.0
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
            # loglik += Normal(torch.zeros(self.n_cont), log_sigma.exp()).log_prob(
            #     self.responses[:, self.continuous_cols] - temp_cont
            # ).sum()
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = self.x_covs @ W_gamma.T + self.xi @ W_zeta.tril().T
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1)
            .log_prob(self.xi - self.eta @ W_kappa.T - self.x_covs1 @ W_coeff_xi.T)
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta - self.x_covs1 @ W_coeff_eta.T).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.W_kappa,
            self.W_coeff_eta,
            self.W_coeff_xi,
        )
        with torch.no_grad():
            jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(
        self,
        log_sigma,
        W_alpha_reduced,
        W_beta_reduced,
        B,
        W_gamma,
        W_zeta_reduced,
        W_kappa,
        W_coeff_eta,
        W_coeff_xi,
    ):
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        loglik_vec = torch.zeros(self.n_responses)
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik_vec += MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            ).log_prob(self.responses[:, self.continuous_cols] - temp_cont)
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik_vec += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum(1)
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B)
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik_vec += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum(1)
        # log p(z | xi)
        temp_z = self.x_covs @ W_gamma.T + self.xi @ W_zeta.tril().T
        loglik_vec += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum(
            1
        )
        # log p(xi | eta)
        loglik_vec += (
            Normal(0, 1)
            .log_prob(self.xi - self.eta @ W_kappa.T - self.x_covs1 @ W_coeff_xi.T)
            .sum(1)
        )
        # log p(eta)
        loglik_vec += (
            Normal(0, 1).log_prob(self.eta - self.x_covs1 @ W_coeff_eta.T).sum(1)
        )
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        with torch.no_grad():
            jaco_efficient = jacfwd(
                self.calcu_loglik_vec, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8)
            )(
                self.log_sigma,
                self.W_alpha[self.w_alpha_mask],
                self.W_beta[self.w_beta_mask],
                self.B,
                self.W_gamma,
                self.W_zeta[self.w_zeta_mask],
                self.W_kappa,
                self.W_coeff_eta,
                self.W_coeff_xi,
            )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        (
            log_sigma,
            W_alpha_reduced,
            W_beta_reduced,
            B,
            W_gamma,
            W_zeta_reduced,
            W_kappa,
            W_coeff_eta,
            W_coeff_xi,
        ) = inputs_vec.split(
            [
                self.n_cont,
                self.w_alpha_mask.sum(),
                self.w_beta_mask.sum(),
                self.n_ord * self.Mj,
                self.n_items * self.n_fixed_effects,
                self.w_zeta_mask.sum(),
                self.n_xi * self.n_eta,
                self.n_eta * self.n_fixed_effects1,
                self.n_xi * self.n_fixed_effects1,
            ]
        )
        W_alpha = torch.zeros(self.n_items, self.n_fixed_effects)
        W_alpha[self.w_alpha_mask] = W_alpha_reduced
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y_cont | eta)
        loglik = 0.0
        # temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        if self.n_cont > 0:
            temp_cont = (
                self.x_covs @ W_alpha[self.continuous_cols, :].T
                + self.eta @ W_beta[self.continuous_cols, :].T
            )
            loglik += (
                MultivariateNormal(
                    torch.zeros(self.n_cont),
                    covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
                )
                .log_prob(self.responses[:, self.continuous_cols] - temp_cont)
                .sum()
            )
        # log p(y_bin | eta)
        if self.n_bin > 0:
            temp_bin = (
                self.x_covs @ W_alpha[self.binary_cols, :].T
                + self.eta @ W_beta[self.binary_cols, :].T
            )
            loglik += (
                self.responses[:, self.binary_cols] * temp_bin
                - torch.log1p(torch.exp(temp_bin))
            ).sum()
        # log p(y_ord | eta)
        if self.n_ord > 0:
            temp_ord = (
                self.x_covs[:, 1:] @ W_alpha[self.ordinal_cols, 1:].T
                + self.eta @ W_beta[self.ordinal_cols, :].T
            )
            D_temp = trans_B_to_D(B.view(self.n_ord, self.Mj))
            D_upper = torch.zeros(self.n_responses, self.n_ord)
            D_lower = torch.zeros(self.n_responses, self.n_ord)
            for j in range(self.n_ord):
                D_upper[:, j] = D_temp[
                    j, self.responses[:, self.ordinal_cols[j]].int() + 1
                ]
                D_lower[:, j] = D_temp[j, self.responses[:, self.ordinal_cols[j]].int()]

            loglik += torch.log(
                torch.sigmoid(temp_ord - D_lower) - torch.sigmoid(temp_ord - D_upper)
            ).sum()
        # log p(z | xi)
        temp_z = (
            self.x_covs @ W_gamma.view(self.n_items, self.n_fixed_effects).T
            + self.xi @ W_zeta.tril().T
        )
        loglik += (self.z_variables * temp_z - torch.log1p(torch.exp(temp_z))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1)
            .log_prob(
                self.xi
                - self.eta @ W_kappa.view(self.n_xi, self.n_eta).T
                - self.x_covs1 @ W_coeff_xi.view(self.n_xi, self.n_fixed_effects1).T
            )
            .sum()
        )
        # log p(eta)
        loglik += (
            Normal(0, 1)
            .log_prob(
                self.eta
                - self.x_covs1 @ W_coeff_eta.view(self.n_eta, self.n_fixed_effects1).T
            )
            .sum()
        )
        return loglik

    def calcu_hessian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha[self.w_alpha_mask],
            self.W_beta[self.w_beta_mask],
            self.B,
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.W_kappa,
            self.W_coeff_eta,
            self.W_coeff_xi,
        )
        with torch.no_grad():
            hessian_tuples = torch.autograd.functional.hessian(
                self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
            )
        return hessian_tuples

    def infer(self, mis_copies=20, M=2000, thinning=1, burn_in=1000):
        params_num = (
            self.log_sigma.numel()
            + self.w_alpha_mask.sum()
            + self.w_beta_mask.sum()
            + self.n_ord * self.Mj
            + self.W_gamma.numel()
            + self.w_zeta_mask.sum()
            + self.W_kappa.numel()
            + self.W_coeff_eta.numel()
            + self.W_coeff_xi.numel()
        )
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_i_obs = torch.zeros(self.n_responses, params_num)

        ## burn-in
        for i in range(burn_in):
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{M}")

        responses_all = []
        S_ij_all1 = []

        k_count = 1.0
        for i in range(M):
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{M}")

            for j in range(thinning):
                with torch.no_grad():
                    self.sample_eta()
                    self.sample_xi()
                    self.impute()

            factor = 1.0
            jacobian_temp = self.calcu_jacobian()
            hessian_temp = self.calcu_hessian()
            res1 += factor * (
                -hessian_temp - torch.outer(jacobian_temp, jacobian_temp) - res1
            )
            res2 += factor * (jacobian_temp - res2)
            mis_info1_ave = mis_info1_ave + (res1 - mis_info1_ave) / k_count
            mis_info2_ave = mis_info2_ave + (res2 - mis_info2_ave) / k_count

            S_i_temp = self.calcu_jacobian_per_sample()
            S_per_sample += factor * (S_i_temp - S_per_sample)
            S_i_obs = S_i_obs + (S_per_sample - S_i_obs) / k_count

            ## every M/mis_copies times, save the responses and S_ij_all1
            if (i + 1) % (M / mis_copies) == 0:
                responses_all.append(self.responses.clone())
                S_ij_latent = self.calcu_jacobian_per_sample()
                S_ij_all1.append(S_ij_latent.clone())

            k_count += 1.0

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        I_obs = (
            I_obs + I_obs.T
        ) / 2  ## make sure it is symmetric to avoid numerical issues
        I_obs_inv = I_obs.inverse()
        D_hat_i_all = S_i_obs @ I_obs_inv * self.n_responses
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        return responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, I_obs_inv


def mean_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, weights_reps=None
):
    if weights_reps is None:
        weights_reps = torch.ones(responses_all[0].shape[0])

    weights_reps = weights_reps / weights_reps.sum()

    n_responses = responses_all[0].shape[0]
    responses_ave = sum(responses_all) / len(responses_all)
    # beta_hat = responses_ave.mean(0)
    ## weighted mean
    beta_hat = responses_ave.T @ weights_reps
    # U_i_bar_all = ((responses_ave - beta_hat).T * weights_reps).T
    U_i_bar_all = responses_ave - beta_hat
    # Omega_c_hat = (responses_ave.T @ responses_ave) / n_responses - torch.outer(
    #     beta_hat, beta_hat
    # )
    Omega_c_hat = U_i_bar_all.T @ (U_i_bar_all.T * weights_reps).T
    kappa_hat1 = [
        (r_ij - beta_hat).T * weights_reps @ (s_ij - S_i_obs)
        for s_ij, r_ij in zip(S_ij_all1, responses_all)
    ]
    kappa_hat1 = sum(kappa_hat1) / len(kappa_hat1)
    temp21 = kappa_hat1 @ D_hat_i_all.T @ (U_i_bar_all.T * weights_reps).T
    term31 = temp21 + temp21.T
    Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
    tau_hat_inv = torch.eye(Omega_hat1.shape[0], dtype=Omega_hat1.dtype)
    Sigma_hat1 = tau_hat_inv @ Omega_hat1 @ tau_hat_inv
    beta_vars1 = Sigma_hat1.diag() * weights_reps.square().sum()

    return beta_hat, beta_vars1


def cond_mean_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, ind=0, weights_reps=None
):
    n_responses, n_items = responses_all[0].shape
    if weights_reps is None:
        weights_reps = torch.ones(n_responses)
    # n_copies = len(responses_all)

    cols = torch.arange(n_items) != ind
    temp = [(y_m[:, cols].T * y_m[:, ind]).T for y_m in responses_all]
    temp1 = [y_m[:, ind].sum() for y_m in responses_all]
    temp2 = [y_m[:, ind].square().sum() for y_m in responses_all]
    temp3 = [y_m[:, cols] for y_m in responses_all]
    quant_val = (sum(temp) / len(temp)).mean(0)
    quant_val1 = sum(temp1) / len(temp1) / n_responses
    quant_val2 = sum(temp2) / len(temp2) / n_responses
    quant_val3 = (sum(temp3) / len(temp3)).mean(0)
    beta_hat = (quant_val - quant_val3 * quant_val1) / (
        quant_val2 - quant_val1.square()
    )
    alpha_hat = quant_val3 - quant_val1 * beta_hat
    U_i_m = [
        torch.cat(
            (
                y_m[:, cols] - alpha_hat - torch.outer(y_m[:, ind], beta_hat),
                (
                    (y_m[:, cols] - alpha_hat - torch.outer(y_m[:, ind], beta_hat)).T
                    * y_m[:, ind]
                ).T,
            ),
            dim=1,
        )
        for y_m in responses_all
    ]
    U_i_bar_all = sum(U_i_m) / len(U_i_m)
    tau_inv = torch.inverse(
        torch.kron(
            torch.tensor(
                [
                    [1, sum(temp1) / len(temp1) / n_responses],
                    [
                        sum(temp1) / len(temp1) / n_responses,
                        sum(temp2) / len(temp2) / n_responses,
                    ],
                ]
            ),
            torch.eye(n_items - 1, n_items - 1),
        )
    )
    Omega_c_hat = (U_i_bar_all.T @ U_i_bar_all) / n_responses
    kappa_hat1 = [
        torch.cat(
            (
                y_m[:, cols] - alpha_hat - torch.outer(y_m[:, ind], beta_hat),
                (
                    (y_m[:, cols] - alpha_hat - torch.outer(y_m[:, ind], beta_hat)).T
                    * y_m[:, ind]
                ).T,
            ),
            dim=1,
        ).T
        @ (s_ij - S_i_obs)
        for s_ij, y_m in zip(S_ij_all1, responses_all)
    ]
    kappa_hat1 = sum(kappa_hat1) / len(kappa_hat1) / n_responses
    temp21 = kappa_hat1 @ D_hat_i_all.T @ U_i_bar_all
    term31 = (temp21 + temp21.T) / n_responses
    Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
    Sigma_hat1 = tau_inv @ Omega_hat1 @ tau_inv
    beta_vars1 = Sigma_hat1.diag() / n_responses

    return torch.concat([alpha_hat, beta_hat]), beta_vars1


def esti_func_all(sigma_x, sigma_y, rho, data):
    x_all = data["x_all"]
    y_all = data["y_all"]
    mu_x = data["mu_x"]
    mu_y = data["mu_y"]

    dx = x_all - mu_x
    dy = y_all - mu_y
    one_minus_rho_sq = 1.0 - rho**2

    # Precompute common terms
    dx_sq = dx**2
    dy_sq = dy**2
    dx_dy = dx * dy
    sigma_x_sq = sigma_x**2
    sigma_y_sq = sigma_y**2

    # Calculate scores more efficiently
    scores_all0 = -0.5 * (
        1.0 / sigma_x_sq
        + 1.0
        / one_minus_rho_sq
        * (-dx_sq / sigma_x_sq**2 + rho * dx_dy / (sigma_x_sq * sigma_x * sigma_y))
    )
    scores_all1 = -0.5 * (
        1.0 / sigma_y_sq
        + 1.0
        / one_minus_rho_sq
        * (-dy_sq / sigma_y_sq**2 + rho * dx_dy / (sigma_x * sigma_y_sq * sigma_y))
    )
    scores_all2 = (
        rho / one_minus_rho_sq
        - rho / one_minus_rho_sq**2 * (dx_sq / sigma_x_sq + dy_sq / sigma_y_sq)
        + (1.0 + rho**2) / one_minus_rho_sq**2 * dx_dy / (sigma_x * sigma_y)
    )

    # Combine scores into a single tensor
    scores = torch.stack([scores_all0, scores_all1, scores_all2], dim=1)

    return scores


def esti_func(sigma_x, sigma_y, rho, data):
    x_all = data["x_all"]
    y_all = data["y_all"]
    mu_x = data["mu_x"]
    mu_y = data["mu_y"]
    weights = data["weights"]

    dx = x_all - mu_x
    dy = y_all - mu_y
    one_minus_rho_sq = 1.0 - rho**2

    # Precompute common terms
    dx_sq = dx**2
    dy_sq = dy**2
    dx_dy = dx * dy
    sigma_x_sq = sigma_x**2
    sigma_y_sq = sigma_y**2

    # Calculate scores more efficiently
    scores_all0 = -0.5 * (
        1.0 / sigma_x_sq
        + 1.0
        / one_minus_rho_sq
        * (-dx_sq / sigma_x_sq**2 + rho * dx_dy / (sigma_x_sq * sigma_x * sigma_y))
    )
    scores_all1 = -0.5 * (
        1.0 / sigma_y_sq
        + 1.0
        / one_minus_rho_sq
        * (-dy_sq / sigma_y_sq**2 + rho * dx_dy / (sigma_x * sigma_y_sq * sigma_y))
    )
    scores_all2 = (
        rho / one_minus_rho_sq
        - rho / one_minus_rho_sq**2 * (dx_sq / sigma_x_sq + dy_sq / sigma_y_sq)
        + (1.0 + rho**2) / one_minus_rho_sq**2 * dx_dy / (sigma_x * sigma_y)
    )

    # Combine scores into a single tensor and apply weights
    scores = torch.stack(
        [scores_all0.mean(1), scores_all1.mean(1), scores_all2.mean(1)], dim=1
    )

    return (scores * weights.unsqueeze(1)).sum(dim=0)


def calculate_jacobian(sigma_x, sigma_y, rho, data):
    def esti_func_wrapper(params):
        return esti_func(params[0], params[1], params[2], data)

    params = torch.tensor([sigma_x, sigma_y, rho], requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(esti_func_wrapper, params)

    return jacobian


def corr_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat, weights_reps=None
):
    if weights_reps is None:
        weights_reps = torch.ones(
            responses_all[0].shape[0], dtype=responses_all[0].dtype
        )
    imputation_param_num = Lambda_hat.shape[0]
    copy_num = len(responses_all)
    weights_reps = weights_reps / weights_reps.sum()
    x_all = torch.stack([response[:, 0] for response in responses_all]).T
    y_all = torch.stack([response[:, 1] for response in responses_all]).T

    mu_x = x_all.mean(1) @ weights_reps
    mu_y = y_all.mean(1) @ weights_reps
    sigma2_x_hat = (x_all - mu_x).square().mean(1) @ weights_reps
    sigma2_y_hat = (y_all - mu_y).square().mean(1) @ weights_reps
    rho_hat = (
        ((x_all - mu_x) * (y_all - mu_y)).mean(1)
        @ weights_reps
        / sigma2_x_hat.sqrt()
        / sigma2_y_hat.sqrt()
    )

    data = {
        "x_all": x_all,
        "y_all": y_all,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "weights": weights_reps,
    }
    U_i_m_all = esti_func_all(sigma2_x_hat.sqrt(), sigma2_y_hat.sqrt(), rho_hat, data)
    U_i_bar_all = U_i_m_all.mean(2)
    Omega_c_hat = U_i_bar_all.T @ (U_i_bar_all.T * weights_reps).T
    S_ij_all1_stacked = torch.stack(S_ij_all1, dim=2)
    kappa_hat1 = torch.zeros(3, imputation_param_num, dtype=responses_all[0].dtype)
    for i in range(copy_num):
        kappa_hat1 += (U_i_m_all[:, :, i].T * weights_reps) @ (
            S_ij_all1_stacked[:, :, i] - S_i_obs
        )

    kappa_hat1 = kappa_hat1 / copy_num
    temp21 = kappa_hat1 @ D_hat_i_all.T @ (U_i_bar_all.T * weights_reps).T
    term31 = temp21 + temp21.T
    Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
    tau_hat_inv = -calculate_jacobian(
        sigma2_x_hat.sqrt(), sigma2_y_hat.sqrt(), rho_hat, data
    ).inverse()
    Sigma_hat1 = tau_hat_inv @ Omega_hat1 @ tau_hat_inv
    beta_vars1 = Sigma_hat1.diag() * weights_reps.square().sum()

    return rho_hat, beta_vars1[-1]


def calculate_corr_var_complete_data(ori_variable, weights_reps_ori):
    # Keep only rows with complete data for both variables
    complete_data_mask = ~ori_variable.isnan().any(dim=1)
    data_y12 = ori_variable[complete_data_mask, :]
    weights_reps = weights_reps_ori[complete_data_mask]
    weights_reps = weights_reps / weights_reps.sum()
    # x_all = torch.stack([response[:, 1] for response in responses_all]).T
    # y_all = torch.stack([response[:, 2] for response in responses_all]).T

    # mu_x = x_all.mean(1) @ weights_reps
    # mu_y = y_all.mean(1) @ weights_reps
    # sigma2_x_hat = (x_all - mu_x).square().mean(1) @ weights_reps
    # sigma2_y_hat = (y_all - mu_y).square().mean(1) @ weights_reps
    # rho_hat = (
    #     ((x_all - mu_x) * (y_all - mu_y)).mean(1)
    #     @ weights_reps
    #     / sigma2_x_hat.sqrt()
    #     / sigma2_y_hat.sqrt()
    # )

    # calculate the mean of y1 and y2
    mu_x = data_y12[:, 0] @ weights_reps
    mu_y = data_y12[:, 1] @ weights_reps

    sigma2_x_hat = (data_y12[:, 0] - mu_x).square() @ weights_reps
    sigma2_y_hat = (data_y12[:, 1] - mu_y).square() @ weights_reps

    rho_hat = (
        ((data_y12[:, 0] - mu_x) * (data_y12[:, 1] - mu_y))
        @ weights_reps
        / sigma2_x_hat.sqrt()
        / sigma2_y_hat.sqrt()
    )

    data = {
        "x_all": data_y12[:, 0].unsqueeze(1),
        "y_all": data_y12[:, 1].unsqueeze(1),
        "mu_x": mu_x,
        "mu_y": mu_y,
        "weights": weights_reps,
    }
    tau_hat_inv = -calculate_jacobian(
        sigma2_x_hat.sqrt(), sigma2_y_hat.sqrt(), rho_hat, data
    ).inverse()
    beta_vars1 = tau_hat_inv.diag() * weights_reps.square().sum()

    return rho_hat, beta_vars1[-1]
