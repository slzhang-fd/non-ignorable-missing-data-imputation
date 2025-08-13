## depend functions
import numpy as np
import torch
from polyagamma import random_polyagamma
from torch.distributions import MultivariateNormal, Normal
from torch.func import jacfwd
from torch.nn import Parameter
import matplotlib.pyplot as plt
import seaborn as sns


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def logistic_loglik(y, temp):
    p = torch.sigmoid(temp)
    return (y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum()


def generate_linear_model_data(x_covs, measurement_sigma, W_alpha, W_beta):
    n_responses, _ = x_covs.shape
    n_items, n_factors = W_beta.shape
    # Generate random effects from multivariate normal with the specified correlation matrix
    x_factors = MultivariateNormal(torch.zeros(n_factors), torch.eye(n_factors)).sample(
        (n_responses,)
    )

    # Generate responses
    return (
        x_covs @ W_alpha.T
        + x_factors @ W_beta.T
        + torch.randn(n_responses, n_items) * measurement_sigma
    )


def generate_logistic_model_data(x_covs, W_gamma, W_zeta):
    n_responses, _ = x_covs.shape
    n_items, n_factors = W_zeta.shape
    # Generate random effects from multivariate normal with the specified correlation matrix
    x_factors = MultivariateNormal(torch.zeros(n_factors), torch.eye(n_factors)).sample(
        (n_responses,)
    )

    # Generate responses
    temp = x_covs @ W_gamma.T + x_factors @ W_zeta.T
    z_variables = torch.rand(n_responses, n_items) < torch.sigmoid(temp)
    return z_variables.float()


def generate_non_ignorable_data(
    x_covs, measurement_sigma, W_alpha, W_beta, W_gamma, W_zeta, W_kappa
):
    n_responses, n_covs = x_covs.shape
    n_items, n_eta = W_beta.shape
    _, n_xi = W_zeta.shape
    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    xi = eta @ W_kappa.T + MultivariateNormal(
        torch.zeros(n_xi), torch.eye(n_xi)
    ).sample((n_responses,))
    responses = (
        x_covs @ W_alpha.T
        + eta @ W_beta.T
        + torch.randn(n_responses, n_items) * measurement_sigma
    )
    temp = x_covs @ W_gamma.T + xi @ W_zeta.T
    mask = torch.rand(n_responses, n_items) < torch.sigmoid(temp)
    responses_masked = torch.where(mask, torch.tensor(float("nan")), responses)
    return responses, responses_masked


def MAR_mask(Y, p, q):
    Y_masked = Y.clone().float()
    for j in range(Y.shape[1] - 1):
        mask = torch.rand(Y.shape[0]) < (p[j] * Y[:, -1] + q[j] * (1 - Y[:, -1]))
        Y_masked[mask, j] = float("nan")

    return Y_masked


def randomly_mask(Y, p):
    mask = torch.rand(*Y.shape) < p
    Y_masked = torch.where(mask, torch.tensor(float("nan")), Y)
    return Y_masked


class MySGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad.zero_()


class MySA_Ruppert:
    def __init__(self, params, lr, alpha=0.51, init_steps=100):
        self.params = params
        self.avg_param = [torch.zeros_like(param) for param in self.params]
        # self.avg_param_all = []
        self.lr = lr
        self.alpha = alpha
        self.t = 0.0
        self.ruppert_t = 1.0
        self.init_steps = init_steps

    def step(self):
        with torch.no_grad():
            for idx, param in enumerate(self.params):
                if self.t < self.init_steps:
                    param -= self.lr * param.grad
                else:
                    # Update parameter with decreasing step size and average gradient
                    param -= (self.lr / (self.ruppert_t**self.alpha)) * param.grad

                    # Update the parameter for Ruppert averaging
                    self.avg_param[idx] = (1 - 1 / self.ruppert_t) * self.avg_param[
                        idx
                    ] + (1 / self.ruppert_t) * param.clone()
                    self.ruppert_t += 1

            # self.avg_param_all.append(self.avg_param.copy())
            # Update the time step
            self.t += 1

    def copy_avg_to_params(self):
        with torch.no_grad():
            for idx, param in enumerate(self.params):
                param.copy_(self.avg_param[idx])

    def zero_grad(self):
        for param in self.params:
            param.grad.zero_()


class MySA_Ruppert1:
    def __init__(self, params, lr, alpha=0.51, burn_in=1000):
        self.params = params
        self.avg_param = [torch.zeros_like(param) for param in self.params]
        self.lr = lr
        self.alpha = alpha
        self.t = 1.0  # Start from 1 to avoid division by zero
        self.ruppert_t = 1.0
        self.burn_in = burn_in

    def step(self):
        with torch.no_grad():
            for idx, param in enumerate(self.params):
                # Update parameter with decreasing step size
                step_size = self.lr / (self.t**self.alpha)
                param -= step_size * param.grad

                # Start Ruppert averaging after burn-in period
                if self.t >= self.burn_in:
                    # Update the parameter for Ruppert averaging
                    # Using numerically stable formula: avg = avg + (param - avg) / t
                    self.avg_param[idx] = (
                        self.avg_param[idx]
                        + (param.clone() - self.avg_param[idx]) / self.ruppert_t
                    )
                    self.ruppert_t += 1.0

            # Update the time step
            self.t += 1.0

    def copy_avg_to_params(self):
        with torch.no_grad():
            for idx, param in enumerate(self.params):
                param.copy_(self.avg_param[idx])

    def zero_grad(self):
        for param in self.params:
            param.grad.zero_()


@torch.jit.script
def sample_xi_jit(z_variables, W_gamma, W_zeta, x_covs, Omega, StandardNormals):
    n_responses, _ = z_variables.shape
    _, n_xi = W_zeta.shape
    xi = torch.zeros(n_responses, n_xi)
    for i in range(n_responses):
        Sigma_xi_pos = torch.inverse(
            W_zeta.tril().T @ torch.diag(Omega[i, :]) @ W_zeta.tril() + torch.eye(n_xi)
        )
        L = torch.linalg.cholesky(Sigma_xi_pos)
        mu_xi_pos = Sigma_xi_pos @ (
            -W_zeta.tril().T
            @ torch.diag(Omega[i, :])
            @ W_gamma
            @ x_covs[i, :].unsqueeze(-1)
            + W_zeta.tril().T @ (z_variables[i, :].unsqueeze(-1) - 0.5)
        )
        xi[i, :] = mu_xi_pos.squeeze() + StandardNormals[i, :] @ L.T

    return xi


class BinaryResponseMeasureModel:
    def __init__(self, z_variables, n_xi, x_covs, initial_values):
        self.z_variables = z_variables
        self.n_responses = z_variables.shape[0]
        self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]
        self.x_covs = x_covs

        self.xi = torch.randn(self.n_responses, self.n_xi)

        self.W_gamma = Parameter(initial_values["W_gamma"])
        self.W_zeta = Parameter(initial_values["W_zeta"])

        self.losses = []

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_jit(
            self.z_variables,
            self.W_gamma,
            self.W_zeta,
            self.x_covs,
            Omega,
            StandNormals,
        )
        # for i in range(self.n_responses):
        #     Sigma_xi_pos = torch.inverse(
        #         self.W_zeta.tril().T @ torch.diag(Omega[i, :]) @ self.W_zeta.tril() + torch.eye(self.n_xi)
        #     )
        #     mu_xi_pos = Sigma_xi_pos @ (
        #         -self.W_zeta.tril().T @ torch.diag(Omega[i, :]) @ self.W_gamma @ self.x_covs[i, :].unsqueeze(-1)
        #         + self.W_zeta.tril().T @ (self.z_variables[i, :].unsqueeze(-1) - 0.5)
        #     )
        #     self.xi[i, :] = mu_xi_pos.squeeze() + MultivariateNormal(torch.zeros(self.n_xi), Sigma_xi_pos).sample()

    def calcu_loglik(self):
        # log p(z | xi, omega)
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        # loglik = (temp * (self.z_variables - 0.5)).sum() - (0.5 * self.omega * temp**2).sum()
        loglik = (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        # loglik = logistic_loglik(self.z_variables, temp)
        # log p(xi)
        loglik += Normal(0, 1).log_prob(self.xi).sum()
        return loglik / self.n_responses

    def fit(self, max_iter=200, lr=0.1):
        optimizer = torch.optim.SGD([self.W_gamma, self.W_zeta], lr=lr)
        # optimizer = MySGD([self.log_sigma, self.W_alpha, self.W_beta, self.W_gamma, self.W_zeta, self.W_kappa], lr=lr)

        for i in range(max_iter):
            # Sample x_factors from the posterior distribution
            with torch.no_grad():
                self.sample_xi()

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
            if (i + 1) % 20 == 0:
                print(f"Epoch {i+1}/{max_iter}, Loss: {loss.item()}")


class BinaryResponseMeasureModel_stand_alone:
    def __init__(self, z_variables, n_xi, x_covs, initial_values):
        self.z_variables = z_variables
        self.n_responses, self.n_items = z_variables.shape
        self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]
        self.x_covs = x_covs

        self.xi = torch.randn(self.n_responses, self.n_xi)

        self.W_gamma = initial_values["W_gamma"]
        self.W_zeta = initial_values["W_zeta"]

        self.losses = []

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_jit(
            self.z_variables,
            self.W_gamma,
            self.W_zeta,
            self.x_covs,
            Omega,
            StandNormals,
        )

    def calcu_loglik(self, W_gamma, W_zeta):
        temp = self.x_covs @ W_gamma.T + self.xi @ W_zeta.T
        loglik = (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        loglik += Normal(0, 1).log_prob(self.xi).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (self.W_gamma, self.W_zeta)
        jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(self, W_gamma, W_zeta):
        temp = self.x_covs @ W_gamma.T + self.xi @ W_zeta.T
        loglik_vec = (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum(1)
        loglik_vec += Normal(0, 1).log_prob(self.xi).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        jaco_efficient = jacfwd(self.calcu_loglik_vec, argnums=(0, 1))(
            self.W_gamma, self.W_zeta
        )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        W_gamma, W_zeta = inputs_vec.split(
            [self.n_items * self.n_fixed_effects, self.n_items * self.n_xi]
        )
        temp = (
            self.x_covs @ W_gamma.view(self.n_items, self.n_fixed_effects).T
            + self.xi @ W_zeta.view(self.n_items, self.n_xi).T
        )
        loglik = (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        loglik += Normal(0, 1).log_prob(self.xi).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (self.W_gamma, self.W_zeta)
        hessian_tuples = torch.autograd.functional.hessian(
            self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
        )
        return hessian_tuples

    def infer(self, M=200, thinning=10, alpha=0.51):
        k_count = 1.0
        params_num = self.W_gamma.numel() + self.W_zeta.numel()
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_per_sample_ave = torch.zeros(self.n_responses, params_num)

        for i in range(M * thinning):
            self.sample_xi()
            if (i + 1) % thinning == 0:
                print(f"Sample {i+1}/{M*thinning}")
                factor = k_count ** (-alpha)

                jacobian_temp = self.calcu_jacobian()
                res1 += factor * (
                    -self.calcu_hessian()
                    - torch.outer(jacobian_temp, jacobian_temp)
                    - res1
                )
                res2 += factor * (jacobian_temp - res2)
                S_per_sample += factor * (
                    self.calcu_jacobian_per_sample() - S_per_sample
                )

                mis_info1_ave = mis_info1_ave + (res1 - mis_info1_ave) / k_count
                mis_info2_ave = mis_info2_ave + (res2 - mis_info2_ave) / k_count
                S_per_sample_ave = (
                    S_per_sample_ave + (S_per_sample - S_per_sample_ave) / k_count
                )
                k_count += 1.0

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        D_hat_i_all = S_per_sample_ave @ torch.inverse(I_obs / self.n_responses)
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        return Lambda_hat, I_obs.inverse()


class NonIgnorableImputer:
    def __init__(self, responses, n_eta, n_xi, x_covs, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.W_gamma = Parameter(initial_values["W_gamma"])
        self.W_zeta = Parameter(initial_values["W_zeta"])
        self.log_W_kappa = Parameter(initial_values["log_W_kappa"])
        self.log_sigma = Parameter(initial_values["log_sigma"])

        self.x_covs = x_covs
        self.z_variables = self.missing_indices.int()
        self.responses = responses
        ## initialize missing values
        self.responses[self.missing_indices] = torch.randn(self.missing_indices.sum())
        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.losses = []

    def sample_eta(self):
        Sigma_eta_pos = torch.inverse(
            self.W_beta.T @ torch.diag(1.0 / self.log_sigma.exp().pow(2)) @ self.W_beta
            + self.log_W_kappa.exp().T @ self.log_W_kappa.exp()
            + torch.eye(self.n_eta)
        )
        mu_eta_pos = Sigma_eta_pos @ (
            self.W_beta.T
            @ torch.diag(1.0 / self.log_sigma.exp().pow(2))
            @ (self.responses - self.x_covs @ self.W_alpha.T).T
            + self.log_W_kappa.exp().T @ self.xi.T
        )
        self.eta = mu_eta_pos.T + MultivariateNormal(
            torch.zeros(self.n_eta), Sigma_eta_pos
        ).sample((self.n_responses,))

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        ## second, sample xi from the posterior distribution
        for i in range(self.n_responses):
            Sigma_xi_pos = torch.inverse(
                self.W_zeta.tril().T @ torch.diag(Omega[i, :]) @ self.W_zeta.tril()
                + torch.eye(self.n_xi)
            )
            mu_xi_pos = Sigma_xi_pos @ (
                self.log_W_kappa.exp() @ self.eta[i, :].unsqueeze(-1)
                - self.W_zeta.tril().T
                @ torch.diag(Omega[i, :])
                @ self.W_gamma
                @ self.x_covs[i, :].unsqueeze(-1)
                + self.W_zeta.tril().T @ (self.z_variables[i, :].unsqueeze(-1) - 0.5)
            )
            self.xi[i, :] = (
                mu_xi_pos.squeeze()
                + MultivariateNormal(torch.zeros(self.n_xi), Sigma_xi_pos).sample()
            )

    def impute(self):
        response_temp = (
            self.x_covs @ self.W_alpha.T
            + self.eta @ self.W_beta.T
            + torch.randn(self.n_responses, self.n_items) * self.log_sigma.exp()
        )
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y | eta)
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_items),
                covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
            )
            .log_prob(
                self.responses - self.x_covs @ self.W_alpha.T - self.eta @ self.W_beta.T
            )
            .sum()
        )
        # log p(z | xi)
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        loglik += (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1).log_prob(self.xi - self.eta @ self.log_W_kappa.exp().T).sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik / self.n_responses

    def fit(self, max_iter=200, lr=0.1):
        optimizer = torch.optim.SGD(
            [
                self.log_sigma,
                self.W_alpha,
                self.W_beta,
                self.W_gamma,
                self.W_zeta,
                self.log_W_kappa,
            ],
            lr=lr,
        )
        # optimizer = MySGD([self.log_sigma, self.W_alpha, self.W_beta, self.W_gamma, self.W_zeta, self.W_kappa], lr=lr)

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


@torch.jit.script
def sample_xi_combined_jit(
    z_variables, W_zeta, x_covs_gamma, eta_kappa, Omega, StandardNormals
):
    n_responses, _ = z_variables.shape
    _, n_xi = W_zeta.shape
    xi = torch.zeros(n_responses, n_xi)
    for i in range(n_responses):
        Sigma_xi_pos = torch.inverse(
            W_zeta.tril().T @ torch.diag(Omega[i, :]) @ W_zeta.tril() + torch.eye(n_xi)
        )
        L = torch.linalg.cholesky(Sigma_xi_pos)
        mu_xi_pos = Sigma_xi_pos @ (
            eta_kappa[i, :].unsqueeze(-1)
            - W_zeta.tril().T
            @ torch.diag(Omega[i, :])
            @ x_covs_gamma[i, :].unsqueeze(-1)
            + W_zeta.tril().T @ (z_variables[i, :].unsqueeze(-1) - 0.5)
        )
        xi[i, :] = mu_xi_pos.squeeze() + StandardNormals[i, :] @ L.T

    return xi


class NonIgnorableImputerInfer:
    def __init__(self, responses, n_eta, n_xi, x_covs, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.W_gamma = Parameter(initial_values["W_gamma"])
        self.W_zeta = Parameter(initial_values["W_zeta"])
        self.log_W_kappa = Parameter(initial_values["log_W_kappa"])
        self.log_sigma = Parameter(initial_values["log_sigma"])

        self.x_covs = x_covs
        self.z_variables = self.missing_indices.int()
        self.responses = responses
        ## initialize missing values
        self.responses[self.missing_indices] = torch.randn(self.missing_indices.sum())
        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)
        self.losses = []

    def sample_eta(self):
        Sigma_eta_pos = torch.inverse(
            self.W_beta.T @ torch.diag(self.log_sigma.exp().pow(-2)) @ self.W_beta
            + self.log_W_kappa.exp().T @ self.log_W_kappa.exp()
            + torch.eye(self.n_eta)
        )
        mu_eta_pos = Sigma_eta_pos @ (
            self.W_beta.T
            @ torch.diag(self.log_sigma.exp().pow(-2))
            @ (self.responses - self.x_covs @ self.W_alpha.T).T
            + self.log_W_kappa.exp().T @ self.xi.T
        )
        self.eta = mu_eta_pos.T + MultivariateNormal(
            torch.zeros(self.n_eta), Sigma_eta_pos
        ).sample((self.n_responses,))

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_jit(
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.log_W_kappa.exp().T,
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = (
            self.x_covs @ self.W_alpha.T
            + self.eta @ self.W_beta.T
            + torch.randn(self.n_responses, self.n_items) * self.log_sigma.exp()
        )
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y | eta)
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_items),
                covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
            )
            .log_prob(
                self.responses
                - self.x_covs @ self.W_alpha.T
                - self.eta @ self.W_beta.tril().T
            )
            .sum()
        )
        # log p(z | xi)
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.tril().T
        loglik += (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1).log_prob(self.xi - self.eta @ self.log_W_kappa.exp().T).sum()
        )
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
                    self.W_gamma,
                    self.W_zeta,
                    self.log_W_kappa,
                ],
                lr=lr,
            )
        elif optimizer_choice == "Adam":
            optimizer = torch.optim.Adam(
                [
                    self.log_sigma,
                    self.W_alpha,
                    self.W_beta,
                    self.W_gamma,
                    self.W_zeta,
                    self.log_W_kappa,
                ],
                lr=lr,
            )
        elif optimizer_choice == "MySA_Ruppert":
            optimizer = MySA_Ruppert(
                [
                    self.log_sigma,
                    self.W_alpha,
                    self.W_beta,
                    self.W_gamma,
                    self.W_zeta,
                    self.log_W_kappa,
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


class NonIgnorableImputerInfer_stand_alone:
    def __init__(self, responses, n_eta, n_xi, x_covs, params_hat_all):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_xi = n_xi
        self.n_fixed_effects = x_covs.shape[1]

        self.W_alpha = params_hat_all["W_alpha"]
        self.W_beta = params_hat_all["W_beta"]
        self.w_beta_mask = torch.ones(self.W_beta.shape).tril().bool()
        self.W_gamma = params_hat_all["W_gamma"]
        self.W_zeta = params_hat_all["W_zeta"]
        self.w_zeta_mask = torch.ones(self.W_zeta.shape).tril().bool()
        self.log_W_kappa = params_hat_all["log_W_kappa"]
        self.log_sigma = params_hat_all["log_sigma"]

        self.x_covs = x_covs
        self.z_variables = self.missing_indices.int()
        self.responses = responses
        ## initialize missing values
        self.responses[self.missing_indices] = torch.randn(self.missing_indices.sum())
        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.xi = torch.randn(self.n_responses, self.n_xi)

    def sample_eta(self):
        Sigma_eta_pos = torch.inverse(
            self.W_beta.T @ torch.diag(self.log_sigma.exp().pow(-2)) @ self.W_beta
            + self.log_W_kappa.exp().T @ self.log_W_kappa.exp()
            + torch.eye(self.n_eta)
        )
        mu_eta_pos = Sigma_eta_pos @ (
            self.W_beta.T
            @ torch.diag(self.log_sigma.exp().pow(-2))
            @ (self.responses - self.x_covs @ self.W_alpha.T).T
            + self.log_W_kappa.exp().T @ self.xi.T
        )
        self.eta = mu_eta_pos.T + MultivariateNormal(
            torch.zeros(self.n_eta), Sigma_eta_pos
        ).sample((self.n_responses,))

    def sample_xi(self):
        temp = self.x_covs @ self.W_gamma.T + self.xi @ self.W_zeta.T
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_xi)
        self.xi = sample_xi_combined_jit(
            self.z_variables,
            self.W_zeta,
            self.x_covs @ self.W_gamma.T,
            self.eta @ self.log_W_kappa.exp().T,
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = (
            self.x_covs @ self.W_alpha.T
            + self.eta @ self.W_beta.T
            + torch.randn(self.n_responses, self.n_items) * self.log_sigma.exp()
        )
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(
        self, log_sigma, W_alpha, W_beta_reduced, W_gamma, W_zeta_reduced, log_W_kappa
    ):
        # log p(y | eta)
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_items),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            )
            .log_prob(self.responses - self.x_covs @ W_alpha.T - self.eta @ W_beta.T)
            .sum()
        )
        # log p(z | xi)
        temp = self.x_covs @ W_gamma.T + self.xi @ W_zeta.T
        loglik += (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        # log p(xi | eta)
        loglik += Normal(0, 1).log_prob(self.xi - self.eta @ log_W_kappa.exp().T).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha,
            self.W_beta[self.w_beta_mask],
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.log_W_kappa,
        )
        with torch.no_grad():
            jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(
        self, log_sigma, W_alpha, W_beta_reduced, W_gamma, W_zeta_reduced, log_W_kappa
    ):
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        # log p(y | eta)
        loglik_vec = MultivariateNormal(
            torch.zeros(self.n_items),
            covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
        ).log_prob(self.responses - self.x_covs @ W_alpha.T - self.eta @ W_beta.T)
        # log p(z | xi)
        temp = self.x_covs @ W_gamma.T + self.xi @ W_zeta.T
        loglik_vec += (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum(1)
        # log p(xi | eta)
        loglik_vec += (
            Normal(0, 1).log_prob(self.xi - self.eta @ log_W_kappa.exp().T).sum(1)
        )
        # log p(eta)
        loglik_vec += Normal(0, 1).log_prob(self.eta).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        with torch.no_grad():
            jaco_efficient = jacfwd(self.calcu_loglik_vec, argnums=(0, 1, 2, 3, 4, 5))(
                self.log_sigma,
                self.W_alpha,
                self.W_beta[self.w_beta_mask],
                self.W_gamma,
                self.W_zeta[self.w_zeta_mask],
                self.log_W_kappa,
            )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        log_sigma, W_alpha, W_beta_reduced, W_gamma, W_zeta_reduced, log_W_kappa = (
            inputs_vec.split(
                [
                    self.n_items,
                    self.n_items * self.n_fixed_effects,
                    self.w_beta_mask.sum(),
                    self.n_items * self.n_fixed_effects,
                    self.w_zeta_mask.sum(),
                    self.n_xi * self.n_eta,
                ]
            )
        )
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        W_zeta = torch.zeros(self.n_items, self.n_xi)
        W_zeta[self.w_zeta_mask] = W_zeta_reduced
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_items),
                covariance_matrix=torch.diag(log_sigma.view(-1).exp().pow(2)),
            )
            .log_prob(
                self.responses
                - self.x_covs @ W_alpha.view(self.n_items, self.n_fixed_effects).T
                - self.eta @ W_beta.T
            )
            .sum()
        )
        # log p(z | xi)
        temp = (
            self.x_covs @ W_gamma.view(self.n_items, self.n_fixed_effects).T
            + self.xi @ W_zeta.T
        )
        loglik += (self.z_variables * temp - torch.log(1 + torch.exp(temp))).sum()
        # log p(xi | eta)
        loglik += (
            Normal(0, 1)
            .log_prob(
                self.xi - self.eta @ log_W_kappa.view(self.n_xi, self.n_eta).exp().T
            )
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha,
            self.W_beta[self.w_beta_mask],
            self.W_gamma,
            self.W_zeta[self.w_zeta_mask],
            self.log_W_kappa,
        )
        with torch.no_grad():
            hessian_tuples = torch.autograd.functional.hessian(
                self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
            )
        return hessian_tuples

    def infer(self, mis_copies=3, M=200, thinning=1, thinning2=5, alpha=0.51):
        k_count = 1.0
        params_num = (
            self.log_sigma.numel()
            + self.W_alpha.numel()
            + self.w_beta_mask.sum()
            + self.W_gamma.numel()
            + self.w_zeta_mask.sum()
            + self.log_W_kappa.numel()
        )
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_i_obs = torch.zeros(self.n_responses, params_num)
        responses_ave = torch.zeros(self.n_responses, self.n_items)

        for i in range(M * thinning):
            with torch.no_grad():
                self.sample_eta()
                self.sample_xi()
                self.impute()
            if (i + 1) % 10 == 0:
                print(f"Sample {i+1}/{M*thinning}")
                factor = k_count ** (-alpha)

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
            k_count = 1.0
            with torch.no_grad():
                self.impute()
            responses_all.append(self.responses.clone())
            # default thinning for each data copy
            for j in range(thinning2):
                with torch.no_grad():
                    self.sample_eta()
                    self.sample_xi()
                S_ij_latent = self.calcu_jacobian_per_sample()
                k_count += 1.0
            S_ij_all1.append(S_ij_latent.clone())

        responses_ave = sum(responses_all) / len(responses_all)
        beta_hat = responses_ave.mean(0)
        U_i_bar_all = responses_ave - beta_hat
        Omega_c_hat = (
            responses_ave.T @ responses_ave
        ) / self.n_responses - torch.outer(beta_hat, beta_hat)
        kappa_hat1 = [
            (r_ij - beta_hat).T @ (s_ij - S_i_obs)
            for s_ij, r_ij in zip(S_ij_all1, responses_all)
        ]
        kappa_hat1 = sum(kappa_hat1) / len(kappa_hat1) / self.n_responses

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        I_obs_inv = I_obs.inverse()
        D_hat_i_all = S_i_obs @ I_obs_inv * self.n_responses
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        temp21 = kappa_hat1 @ D_hat_i_all.T @ U_i_bar_all
        term31 = (temp21 + temp21.T) / self.n_responses
        Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
        Sigma_hat1 = Omega_hat1
        return D_hat_i_all, Lambda_hat, I_obs_inv, beta_hat, Sigma_hat1


class LinearFactorModelInfer:
    def __init__(self, responses, n_eta, x_covs, initial_values):
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_fixed_effects = x_covs.shape[1]

        self.log_sigma = Parameter(initial_values["log_sigma"])
        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])

        self.x_covs = x_covs
        self.responses = responses
        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.losses = []

    def sample_eta(self):
        Sigma_eta_pos = torch.inverse(
            self.W_beta.T @ torch.diag(1.0 / self.log_sigma.exp().pow(2)) @ self.W_beta
            + torch.eye(self.n_eta)
        )
        mu_eta_pos = (
            Sigma_eta_pos
            @ self.W_beta.T
            @ torch.diag(1.0 / self.log_sigma.exp().pow(2))
            @ (self.responses - self.x_covs @ self.W_alpha.T).T
        )

        self.eta = mu_eta_pos.T + MultivariateNormal(
            torch.zeros(self.n_eta), Sigma_eta_pos
        ).sample((self.n_responses,))

    def calcu_loglik(self):
        # log p(y | eta)
        loglik = (
            Normal(0, self.log_sigma.exp())
            .log_prob(
                self.responses - self.x_covs @ self.W_alpha.T - self.eta @ self.W_beta.T
            )
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik / self.n_responses

    def fit(self, optimizer_choice="SGD", max_iter=200, lr=0.1):
        if optimizer_choice == "SGD":
            optimizer = torch.optim.SGD(
                [self.log_sigma, self.W_alpha, self.W_beta], lr=lr
            )
        elif optimizer_choice == "MySA_Ruppert":
            optimizer = MySA_Ruppert(
                [self.log_sigma, self.W_alpha, self.W_beta],
                lr=lr,
                init_steps=(int)(max_iter / 2),
            )
        else:
            raise ValueError("optimizer_choice must be SGD or MySA_Ruppert")
        # optimizer = MySGD([self.log_sigma, self.W_alpha, self.W_beta], lr=lr)

        for i in range(max_iter):
            # Sample x_factors from the posterior distribution
            with torch.no_grad():
                self.sample_eta()

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

        # if optimizer_choice == "MySA_Ruppert":
        #     optimizer.copy_avg_to_params()


class LinearFactorModelInfer_stand_alone:
    def __init__(self, responses, n_eta, x_covs, params_hat):
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_fixed_effects = x_covs.shape[1]

        self.log_sigma = params_hat["log_sigma"]
        self.W_alpha = params_hat["W_alpha"]
        self.W_beta = params_hat["W_beta"]

        self.x_covs = x_covs
        self.responses = responses
        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.Sigma_eta_pos = torch.inverse(
            self.W_beta.T @ torch.diag(1.0 / self.log_sigma.exp().pow(2)) @ self.W_beta
            + torch.eye(self.n_eta)
        )
        self.mu_eta_pos = (
            self.Sigma_eta_pos
            @ self.W_beta.T
            @ torch.diag(1.0 / self.log_sigma.exp().pow(2))
            @ (self.responses - self.x_covs @ self.W_alpha.T).T
        )
        self.L = torch.linalg.cholesky(self.Sigma_eta_pos)

    def sample_eta(self):
        self.eta = (
            self.mu_eta_pos.T + torch.randn(self.n_responses, self.n_eta) @ self.L
        )

    def calcu_loglik(self, inputs_vec):
        log_sigma, W_alpha, W_beta = inputs_vec.split(
            [
                self.n_items,
                self.n_items * self.n_fixed_effects,
                self.n_items * self.n_eta,
            ]
        )
        # log p(y | eta)
        loglik = (
            Normal(0, log_sigma.view(-1).exp())
            .log_prob(
                self.responses
                - self.x_covs @ W_alpha.view(self.n_items, self.n_fixed_effects).T
                - self.eta @ W_beta.view(self.n_items, self.n_eta).T
            )
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (self.log_sigma, self.W_alpha, self.W_beta)
        jac_tuples = torch.autograd.functional.jacobian(
            self.calcu_loglik, torch.cat([t.view(-1) for t in inputs])
        )

        return jac_tuples

    def calcu_loglik_vec(self, log_sigma, W_alpha, W_beta):
        # log p(y | eta)
        loglik_vec = (
            Normal(0, log_sigma.exp())
            .log_prob(self.responses - self.x_covs @ W_alpha.T - self.eta @ W_beta.T)
            .sum(1)
        )
        # log p(eta)
        loglik_vec += Normal(0, 1).log_prob(self.eta).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        jaco_efficient = jacfwd(self.calcu_loglik_vec, argnums=(0, 1, 2))(
            self.log_sigma, self.W_alpha, self.W_beta
        )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        log_sigma, W_alpha, W_beta = inputs_vec.split(
            [
                self.n_items,
                self.n_items * self.n_fixed_effects,
                self.n_items * self.n_eta,
            ]
        )
        loglik = (
            Normal(0, log_sigma.view(-1).exp())
            .log_prob(
                self.responses
                - self.x_covs @ W_alpha.view(self.n_items, self.n_fixed_effects).T
                - self.eta @ W_beta.view(self.n_items, self.n_eta).T
            )
            .sum()
        )
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (self.log_sigma, self.W_alpha, self.W_beta)
        hessian_tuples = torch.autograd.functional.hessian(
            self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
        )
        return hessian_tuples

    def infer(self, M=200):
        k_count = 1.0
        params_num = self.log_sigma.numel() + self.W_alpha.numel() + self.W_beta.numel()
        res1 = torch.zeros(params_num, params_num)
        res2 = torch.zeros(params_num)
        S_per_sample = torch.zeros(self.n_responses, params_num)
        mis_info1_ave = torch.zeros(params_num, params_num)
        mis_info2_ave = torch.zeros(params_num)
        S_per_sample_ave = torch.zeros(self.n_responses, params_num)
        res1_svds = []
        res2_svds = []
        # S_per_sample_all = []

        for i in range(M):
            self.sample_eta()
            if (i + 1) % 100 == 0:
                print(f"Sample {i+1}/{M}")

            jacobian_temp = self.calcu_jacobian()
            hessian_temp = self.calcu_hessian()
            res1 = -hessian_temp - torch.outer(jacobian_temp, jacobian_temp)
            res2 = jacobian_temp
            res1_svds.append(hessian_temp.clone())
            res2_svds.append(jacobian_temp.clone())
            S_per_sample = self.calcu_jacobian_per_sample()
            # S_per_sample_all.append(S_per_sample)

            mis_info1_ave = mis_info1_ave + (res1 - mis_info1_ave) / k_count
            mis_info2_ave = mis_info2_ave + (res2 - mis_info2_ave) / k_count
            S_per_sample_ave = (
                S_per_sample_ave + (S_per_sample - S_per_sample_ave) / k_count
            )
            k_count += 1.0

        I_obs = mis_info1_ave + torch.outer(mis_info2_ave, mis_info2_ave)
        I_obs_inv = I_obs.inverse()
        D_hat_i_all = S_per_sample_ave @ I_obs_inv * self.n_responses
        Lambda_hat = (D_hat_i_all.T @ D_hat_i_all) / self.n_responses

        return Lambda_hat, I_obs_inv, res1_svds, res2_svds


def plot_analysis_point_estimation(analysis_params_true, estimated_mean, filename):
    # Create scatter plot with custom styling
    sns.set_style("whitegrid")
    plt.scatter(analysis_params_true, estimated_mean, color="#1f77b4")

    # Add reference line through origin with slope 1
    min_val = min(analysis_params_true.min(), estimated_mean.min())
    max_val = max(analysis_params_true.max(), estimated_mean.max())
    plt.axline(
        (min_val, min_val), (max_val, max_val), color="black", linestyle="--", alpha=0.5
    )

    # Customize plot
    plt.xlabel("True Value", fontsize=12)
    plt.ylabel("Estimated Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Make plot square and set same limits on both axes
    # plt.gca().set_aspect("equal")
    plt.xlim(min_val * 0.9, max_val * 1.1)
    plt.ylim(min_val * 0.9, max_val * 1.1)

    plt.tight_layout()
    plt.savefig(
        filename,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_analysis_boxplot(difference, filename, xaxis_ticks=True):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 4))

    # Create boxplot with customized appearance
    sns.boxplot(
        data=difference,
        color="skyblue",
        width=0.5,
        fliersize=2,
        linewidth=1,
        medianprops={"color": "red", "linewidth": 1.5},
        boxprops={"alpha": 0.8},
    )

    # Customize axes and labels
    if xaxis_ticks:
        plt.xticks(range(difference.size()[1]), range(1, difference.size()[1] + 1))
    else:
        plt.xticks([])  # Remove x-axis ticks
    plt.xlim(-1, difference.size()[1])
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Parameter Index", fontsize=12)
    plt.ylabel("Estimation Error", fontsize=12)
    # plt.title("Estimation Bias Across Parameters", fontsize=14)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        filename,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_analysis_sd_estimation(empirical_sd, estimated_sd, filename):
    plt.scatter(empirical_sd, estimated_sd, color="#1f77b4")

    # Add reference line
    max_val = max(empirical_sd.max(), estimated_sd.max())
    min_val = min(empirical_sd.min(), estimated_sd.min())
    plt.axline(
        (min_val, min_val), (max_val, max_val), color="black", linestyle="--", alpha=0.5
    )

    # Customize plot
    plt.xlabel("Empirical Standard Error", fontsize=12)
    plt.ylabel("Estimated Standard Error", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Make plot square
    # plt.gca().set_aspect("equal")

    # Set same limits on both axes
    plt.xlim(min_val * 0.9, max_val * 1.1)
    plt.ylim(min_val * 0.9, max_val * 1.1)

    plt.tight_layout()
    plt.savefig(
        filename,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_analysis_coverage(coverage_ratios, filename, xaxis_tick_num=1):
    plt.figure(figsize=(8, 4))
    plt.scatter(range(coverage_ratios.shape[0]), coverage_ratios, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlim(-1, coverage_ratios.shape[0])  # Set x-axis limits from -1 to 20

    # Create ticks every other value from 1
    tick_positions = range(0, coverage_ratios.shape[0], xaxis_tick_num)
    tick_labels = range(1, coverage_ratios.shape[0] + 1, xaxis_tick_num)
    plt.xticks(tick_positions, tick_labels)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(
        y=0.95, color="red", linestyle="--", label="95% Coverage"
    )  # 95% line for reference
    plt.xlabel("Parameter Index")
    plt.ylabel("Coverage Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        filename,
        format="png",
        dpi=300,
    )
    plt.show()
