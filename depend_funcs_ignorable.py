## depend functions
import torch
from polyagamma import random_polyagamma
from torch.distributions import MultivariateNormal, Normal
from torch.func import jacfwd
from torch.nn import Parameter

from depend_funcs import MySA_Ruppert1


def generate_ignorable_mixed_data(
    x_covs, measurement_sigma, W_alpha, W_beta, n_cont, n_bin
):
    n_responses, _ = x_covs.shape
    _, n_eta = W_beta.shape
    continuous_cols = torch.arange(n_cont)
    binary_cols = torch.arange(n_cont, n_cont + n_bin)
    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    responses = x_covs @ W_alpha.T + eta @ W_beta.T
    responses[:, continuous_cols] += (
        torch.randn(n_responses, n_cont) * measurement_sigma
    )
    responses[:, binary_cols] = torch.bernoulli(
        torch.sigmoid(responses[:, binary_cols])
    )
    return responses


def MAR_mask(Y, p, q):
    Y_masked = Y.clone()
    for j in range(Y.shape[1] - 1):
        # mask = np.random.rand(Y.shape[0]) < (p[j] * Y[:,-1] + q[j] * (1-Y[:,-1]))
        mask = torch.rand(Y.shape[0]) < (p[j] * Y[:, -1] + q[j] * (1 - Y[:, -1]))
        Y_masked[mask, j] = float("nan")

    return Y_masked


def generate_non_ignorable_mixed_data(
    x_covs, measurement_sigma, W_alpha, W_beta, W_gamma, W_zeta, W_kappa, n_cont, n_bin
):
    n_responses, _ = x_covs.shape
    n_items, n_eta = W_beta.shape
    _, n_xi = W_zeta.shape
    continuous_cols = torch.arange(n_cont)
    binary_cols = torch.arange(n_cont, n_cont + n_bin)
    eta = MultivariateNormal(torch.zeros(n_eta), torch.eye(n_eta)).sample(
        (n_responses,)
    )
    xi = eta @ W_kappa.T + MultivariateNormal(
        torch.zeros(n_xi), torch.eye(n_xi)
    ).sample((n_responses,))
    responses = x_covs @ W_alpha.T + eta @ W_beta.T
    responses[:, continuous_cols] += (
        torch.randn(n_responses, n_cont) * measurement_sigma
    )
    responses[:, binary_cols] = torch.bernoulli(
        torch.sigmoid(responses[:, binary_cols])
    )

    temp = x_covs @ W_gamma.T + xi @ W_zeta.T
    mask = torch.rand(n_responses, n_items) < torch.sigmoid(temp)
    responses_masked = torch.where(mask, torch.tensor(float("nan")), responses)
    return responses, responses_masked


@torch.jit.script
def sample_eta_mixed_jit(
    response_cont,
    response_bin,
    W_beta_cont,
    W_beta_bin,
    x_cov_alpha_c,
    x_cov_alpha_b,
    sigma_sq_inv,
    Omega,
    StandardNormals,
):
    n_responses, _ = response_cont.shape
    _, n_eta = W_beta_cont.shape
    eta = torch.zeros(n_responses, n_eta)
    Sigma_temp = W_beta_cont.T @ torch.diag(sigma_sq_inv) @ W_beta_cont + torch.eye(
        n_eta
    )
    for i in range(n_responses):
        Sigma_eta_pos = torch.inverse(
            Sigma_temp + W_beta_bin.T @ torch.diag(Omega[i, :]) @ W_beta_bin
        )
        L = torch.linalg.cholesky(Sigma_eta_pos)
        mu_eta_pos = Sigma_eta_pos @ (
            W_beta_cont.T
            @ torch.diag(sigma_sq_inv)
            @ (response_cont[i, :].unsqueeze(-1) - x_cov_alpha_c[i, :].unsqueeze(-1))
            - W_beta_bin.T @ torch.diag(Omega[i, :]) @ x_cov_alpha_b[i, :].unsqueeze(-1)
            + W_beta_bin.T @ (response_bin[i, :].unsqueeze(-1) - 0.5)
        )
        eta[i, :] = mu_eta_pos.squeeze() + StandardNormals[i, :] @ L.T

    return eta


class IgnorableImputerInfer:
    def __init__(self, responses, n_eta, x_covs, initial_values):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = []
        self.binary_cols = []
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
        self.x_covs = x_covs

        self.W_alpha = Parameter(initial_values["W_alpha"])
        self.W_beta = Parameter(initial_values["W_beta"])
        self.log_sigma = Parameter(initial_values["log_sigma"])

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)
        self.losses = []

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_mixed_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            1.0 / self.log_sigma.exp().pow(2),
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
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self):
        # log p(y_cont | eta)
        temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.tril().T
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(self.log_sigma.exp().pow(2)),
            )
            .log_prob(
                self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
            )
            .sum()
        )
        # log p(y_bin | eta)
        loglik += (
            self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
            - torch.log1p(torch.exp(temp[:, self.binary_cols]))
        ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik / self.n_responses

    def fit(self, optimizer_choice="SGD", max_iter=200, lr=0.1, alpha=0.51):
        if optimizer_choice == "SGD":
            optimizer = torch.optim.SGD(
                [self.log_sigma, self.W_alpha, self.W_beta], lr=lr
            )
        elif optimizer_choice == "Adam":
            optimizer = torch.optim.Adam(
                [self.log_sigma, self.W_alpha, self.W_beta], lr=lr
            )
        elif optimizer_choice == "MySA_Ruppert":
            optimizer = MySA_Ruppert1(
                [self.log_sigma, self.W_alpha, self.W_beta],
                # lr=lr,
                # alpha=alpha,
                # init_steps=(int)(max_iter / 2),
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


class IgnorableImputerInfer_stand_alone:
    def __init__(self, responses, n_eta, x_covs, params_hat_all):
        self.missing_indices = torch.isnan(responses)
        self.n_responses, self.n_items = responses.shape
        self.n_eta = n_eta
        self.n_fixed_effects = x_covs.shape[1]
        self.continuous_cols = []
        self.binary_cols = []
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
        self.x_covs = x_covs

        self.W_alpha = params_hat_all["W_alpha"]
        self.W_beta = params_hat_all["W_beta"]
        self.w_beta_mask = torch.ones(self.W_beta.shape).tril().bool()
        self.log_sigma = params_hat_all["log_sigma"]

        ## initialize latent variables
        self.eta = torch.randn(self.n_responses, self.n_eta)

    def sample_eta(self):
        temp = (
            self.x_covs @ self.W_alpha[self.binary_cols, :].T
            + self.eta @ self.W_beta[self.binary_cols, :].T
        )
        Omega = torch.tensor(
            random_polyagamma(torch.ones(temp.shape), temp),
            dtype=torch.get_default_dtype(),
        )
        StandNormals = torch.randn(self.n_responses, self.n_eta)
        self.eta = sample_eta_mixed_jit(
            self.responses[:, self.continuous_cols],
            self.responses[:, self.binary_cols],
            self.W_beta[self.continuous_cols, :],
            self.W_beta[self.binary_cols, :],
            self.x_covs @ self.W_alpha[self.continuous_cols, :].T,
            self.x_covs @ self.W_alpha[self.binary_cols, :].T,
            1.0 / self.log_sigma.exp().pow(2),
            Omega,
            StandNormals,
        )

    def impute(self):
        response_temp = self.x_covs @ self.W_alpha.T + self.eta @ self.W_beta.T
        response_temp[:, self.continuous_cols] += (
            torch.randn(self.n_responses, self.n_cont) * self.log_sigma.exp()
        )
        response_temp[:, self.binary_cols] = torch.bernoulli(
            torch.sigmoid(response_temp[:, self.binary_cols])
        )
        self.responses[self.missing_indices] = response_temp[self.missing_indices]

    def calcu_loglik(self, log_sigma, W_alpha, W_beta_reduced):
        # log p(y_cont | eta)
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            )
            .log_prob(
                self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
            )
            .sum()
        )
        # log p(y_bin | eta)
        loglik += (
            self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
            - torch.log1p(torch.exp(temp[:, self.binary_cols]))
        ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_jacobian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha,
            self.W_beta[self.w_beta_mask],
        )
        with torch.no_grad():
            jac_tuples = torch.autograd.functional.jacobian(self.calcu_loglik, inputs)

        return torch.cat([t.view(-1) for t in jac_tuples], dim=0)

    def calcu_loglik_vec(self, log_sigma, W_alpha, W_beta_reduced):
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        temp = self.x_covs @ W_alpha.T + self.eta @ W_beta.T
        # log p(y | eta)
        loglik_vec = MultivariateNormal(
            torch.zeros(self.n_cont),
            covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
        ).log_prob(
            self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
        )
        # log p(z | xi)
        loglik_vec += (
            self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
            - torch.log1p(torch.exp(temp[:, self.binary_cols]))
        ).sum(1)
        # log p(eta)
        loglik_vec += Normal(0, 1).log_prob(self.eta).sum(1)
        return loglik_vec

    def calcu_jacobian_per_sample(self):
        with torch.no_grad():
            jaco_efficient = jacfwd(self.calcu_loglik_vec, argnums=(0, 1, 2))(
                self.log_sigma,
                self.W_alpha,
                self.W_beta[self.w_beta_mask],
            )
        return torch.cat([x.view(self.n_responses, -1) for x in jaco_efficient], dim=1)

    def calcu_loglik_closure(self, inputs_vec):
        # log p(y | eta)
        log_sigma, W_alpha, W_beta_reduced = inputs_vec.split(
            [
                self.n_cont,
                self.n_items * self.n_fixed_effects,
                self.w_beta_mask.sum(),
            ]
        )
        W_beta = torch.zeros(self.n_items, self.n_eta)
        W_beta[self.w_beta_mask] = W_beta_reduced
        temp = (
            self.x_covs @ W_alpha.view(self.n_items, self.n_fixed_effects).T
            + self.eta @ W_beta.T
        )
        loglik = (
            MultivariateNormal(
                torch.zeros(self.n_cont),
                covariance_matrix=torch.diag(log_sigma.exp().pow(2)),
            )
            .log_prob(
                self.responses[:, self.continuous_cols] - temp[:, self.continuous_cols]
            )
            .sum()
        )
        # log p(y_bin | eta)
        loglik += (
            self.responses[:, self.binary_cols] * temp[:, self.binary_cols]
            - torch.log1p(torch.exp(temp[:, self.binary_cols]))
        ).sum()
        # log p(eta)
        loglik += Normal(0, 1).log_prob(self.eta).sum()
        return loglik

    def calcu_hessian(self):
        inputs = (
            self.log_sigma,
            self.W_alpha,
            self.W_beta[self.w_beta_mask],
        )
        with torch.no_grad():
            hessian_tuples = torch.autograd.functional.hessian(
                self.calcu_loglik_closure, torch.cat([t.view(-1) for t in inputs])
            )
        return hessian_tuples

    def infer(self, mis_copies=20, M=2000, thinning=1, burn_in=1000):
        params_num = (
            self.log_sigma.numel() + self.W_alpha.numel() + self.w_beta_mask.sum()
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

            for j in range(thinning):
                with torch.no_grad():
                    self.sample_eta()
                    self.impute()

        k_count = 1.0
        responses_all = []
        S_ij_all1 = []

        ## burn-in done, start the main loop
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


def mean_analysis_model(responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat):
    n_responses = responses_all[0].shape[0]
    responses_ave = sum(responses_all) / len(responses_all)
    beta_hat = responses_ave.mean(0)
    U_i_bar_all = responses_ave - beta_hat
    Omega_c_hat = (responses_ave.T @ responses_ave) / n_responses - torch.outer(
        beta_hat, beta_hat
    )
    kappa_hat1 = [
        (r_ij - beta_hat).T @ (s_ij - S_i_obs)
        for s_ij, r_ij in zip(S_ij_all1, responses_all)
    ]
    kappa_hat1 = sum(kappa_hat1) / len(kappa_hat1) / n_responses
    temp21 = kappa_hat1 @ D_hat_i_all.T @ U_i_bar_all
    term31 = (temp21 + temp21.T) / n_responses
    Omega_hat1 = Omega_c_hat + kappa_hat1 @ Lambda_hat @ kappa_hat1.T + term31
    Sigma_hat1 = Omega_hat1
    beta_vars1 = Sigma_hat1.diag() / n_responses

    return beta_hat, beta_vars1


def cond_mean_analysis_model(
    responses_all, S_ij_all1, S_i_obs, D_hat_i_all, Lambda_hat
):
    n_responses, n_items = responses_all[0].shape
    # n_copies = len(responses_all)

    temp = [(y_m[:, :-1].T * y_m[:, -1]).T for y_m in responses_all]
    temp1 = [y_m[:, -1].sum() for y_m in responses_all]
    temp2 = [y_m[:, -1].square().sum() for y_m in responses_all]
    temp3 = [y_m[:, :-1] for y_m in responses_all]
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
                y_m[:, :-1] - alpha_hat - torch.outer(y_m[:, -1], beta_hat),
                (
                    (y_m[:, :-1] - alpha_hat - torch.outer(y_m[:, -1], beta_hat)).T
                    * y_m[:, -1]
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
                y_m[:, :-1] - alpha_hat - torch.outer(y_m[:, -1], beta_hat),
                (
                    (y_m[:, :-1] - alpha_hat - torch.outer(y_m[:, -1], beta_hat)).T
                    * y_m[:, -1]
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
