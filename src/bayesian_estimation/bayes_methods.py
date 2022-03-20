import numpy as np
from scipy.stats import f

import pymc as pm
import aesara.tensor as at
from src.bayesian_estimation.pystan_models import GPD, Fisher
import scipy.special as sc

# beta_frechet = 1/2
# quant_th = np.zeros(len(q))
# for i in range(len(q)):
#     quant_th[i] = pow(-log(q[i]), -beta_frechet)


def jprior_alpha(alpha: float):
    """
    Jeffrey priors on alpha in the GPD
    :param alpha:
    :return:
    """
    assert alpha >= 0, 'alpha is negative'
    lpdf = lambda param_alpha: -np.log(param_alpha + 1) - 1 / 2 * (np.log(param_alpha) + np.log(param_alpha + 2))
    return pm.DensityDist('jeffrey_alpha', logp=lpdf(alpha))


def jprior_beta(beta: float):
    """
    Jeffrey priors on beta in the GPD
    :param beta:
    :return:
    """
    assert beta >= 0, 'alpha is negative'
    lpdf = lambda param_beta: np.log(1 / param_beta)
    return pm.DensityDist('jeffrey_beta', logp=lpdf(beta))


def gpd_model(excesses: np.ndarray):
    # log pdf of the gpd used
    myGPD_lpdf = lambda x, alpha, beta: -(alpha + 1) * (np.log(1 + x / beta)) + (np.log(alpha) - np.log(beta))
    # log pdf of the Fisher used
    myFisher_lpdf = lambda x, alpha1, alpha2, beta: -sc.beta(alpha1, alpha2) - np.log(beta) + (alpha1 - 1) * np.log(x / beta) - (alpha1 + alpha2) * np.log(1 + x / beta);
    # defining Jeffrey prior

    # theta1 = pm.DensityDist('theta1', jprior_alpha, value, y=theta2)

    with pm.Model() as gpd_model:
        alpha = pm.Gamma('alpha', 1, 1)

        # Create custom densities
        lGPD = pm.DensityDist('lGPD', logp=myGPD_lpdf(jprior_alpha, ), observed=excesses)

        # Generate a MCMC chain
        start = pm.find_MAP()  # Find starting value by optimization
        trace = pm.sample(10000, pm.NUTS(), progressbar=False)  # Use NUTS sampling

        lFisher = pm.DensityDist('lFisher', logp=myFisher_lpdf)
        # Create likelihood
    with gpd_model:
        # obtain starting values via MAP
        startvals = pm.find_MAP(model=gpd_model)

        # instantiate sampler
        # step = pm.Metropolis()
        step = pm.HamiltonianMC()
        # step = pm.NUTS()

        # draw 5000 posterior samples
        trace = pm.sample(start=startvals, draws=1000, step=step, tune=500, chains=4, cores=1,
                          discard_tuned_samples=True)

        # Obtaining Posterior Predictive Sampling:
        post_pred = pm.sample_posterior_predictive(trace, samples=500)
        print(post_pred['observed_data'].shape)


def gpd_model_(excesses: np.ndarray):
    with pm.Model() as model:
        # Parameters:
        # Prior Distributions:
        # BoundedNormal = pm.Bound(pm.Exponential, lower=0.0, upper=np.inf)
        # c = BoundedNormal('c', lam=10)
        # c = pm.Uniform('c', lower=0, upper=300)
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Normal('mu', mu=0, sigma=1)
        sd = pm.HalfNormal('sd', sigma=1)

        # Observed data is from a Multinomial distribution:
        # Likelihood distributions:
        # bradford = pm.DensityDist('observed_data', logp=bradford_logp, observed=dict(value=S1, loc=mu, scale=sd, c=c))
        observed_data = pm.Beta('observed_data', alpha=alpha, beta=beta, mu=mu, sd=sd, observed=excesses)

    with model:
        # obtain starting values via MAP
        startvals = pm.find_MAP(model=model)

        # instantiate sampler
        # step = pm.Metropolis()
        step = pm.HamiltonianMC()
        # step = pm.NUTS()

        # draw 5000 posterior samples
        trace = pm.sample(start=startvals, draws=1000, step=step, tune=500, chains=4, cores=1,
                          discard_tuned_samples=True)

        # Obtaining Posterior Predictive Sampling:
        post_pred = pm.sample_posterior_predictive(trace, samples=500)
        print(post_pred['observed_data'].shape)


def fisher_model():
    pass


class bayes_methods:
    def __init__(self, data: np.ndarray, quantile_levels: list, excesses: np.ndarray, thresholds: np.ndarray):
        self.data = data  # original Burr data
        self.quantile_levels = quantile_levels
        self.excesses = excesses
        self.thresholds = thresholds
        self.n_excesses = excesses.shape[0]
        self.chain_length = 1000
        self.burn_up = 200

    def bayes_GPD(self):
        """
        we forward to function array of the quantiles as excesses, the number of excesses, k, and the border value u
        """
        k = self.excesses.shape[0]
        quant_GPD = np.zeros(len(self.quantile_levels))
        bayesian_quant_GPD = np.zeros(len(self.quantile_levels))
        median_quant_GPD = np.zeros(self.chain_length - self.burn_up)
        all_median_quant_GPD = np.zeros(self.chain_length - self.burn_up)

        # here we fit GPD to excesses via PyStan
        m = gpd_model(self.excesses)
        data = dict(N=k, y=self.excesses)
        fit = StanModel(model_code=GPD).sampling(data=data, iter=self.chain_length, warmup=self.burn_up, chains=1)

        # we save the params from the fit to calculate GPD quantiles and their traceplots to calculate Bayesian GPD quantiles
        traceplot_beta_GPD = list(fit.extract().values())[1].tolist()
        traceplot_alpha = list(fit.extract().values())[0].tolist()
        traceplot_gamma = np.divide(np.ones(len(traceplot_alpha)), traceplot_alpha)
        beta_GPD = np.mean(list(fit.extract().values())[1].tolist())
        alpha = np.mean(list(fit.extract().values())[0].tolist())
        gamma = 1 / alpha

        plug_in_median_GPD = []
        quant_GPD = []
        for quantile_level in self.quantile_levels:
            plug_in_median_GPD.append(self.thresholds + np.median(traceplot_beta_GPD) * (
                    pow(N * (1 - quantile_level) / k, - 1 / np.median(traceplot_alpha)) - 1))

            quant_GPD.append(self.thresholds + beta_GPD * (pow(N * (1 - quantile_level) / k, -gamma) - 1))

            for j in range(len(traceplot_gamma)):
                bayesian_quant_GPD[i] += self.thresholds + traceplot_beta_GPD[j] * \
                                         (pow(N * (1 - quantile_level) / k, - traceplot_gamma[j]) - 1)
                median_quant_GPD[j] = self.thresholds + traceplot_beta_GPD[j] *\
                                      (pow(N * (1 - quantile_level) / k, - traceplot_gamma[j]) - 1)

            all_median_quant_GPD = np.column_stack((all_median_quant_GPD, median_quant_GPD))

        bayesian_quant_GPD = bayesian_quant_GPD / len(traceplot_gamma)
        all_median_quant_GPD = np.delete(all_median_quant_GPD, 0, 1)
        #     for j in ran?ge(len(traceplot_gamma)):
        store_medians = np.zeros(len(self.quantile_levels))
        # taking a median of quantiles computed in Bayesian method
        store_medians = []
        for ind, _ in enumerate(self.quantile_levels):
            store_medians.append(np.median(all_median_quant_GPD[:, ind]))

        list_of_params = [alpha, beta_GPD]
        return quant_GPD, bayesian_quant_GPD, list_of_params, store_medians, plug_in_median_GPD
        # it return arrays: quant_GPD, bayesian_quant_GPD and values alpha, beta_GPD

    def quantiles_Fisher(self, excesses, k, u):
        plug_in_median_Fisher = np.zeros(len(self.quantile_levels))
        quant_Fisher = np.zeros(len(self.quantile_levels))
        bayesian_quant_Fisher = np.zeros(len(self.quantile_levels))
        median_quant_Fisher = np.zeros(self.chain_length - self.burn_up)
        all_median_quant_Fisher = np.zeros(self.chain_length - self.burn_up)

        # here we fit Fisher to excesses via PyStan
        data = dict(N=k, y=excesses)
        fit = StanModel(model_code=Fisher).sampling(data=data, iter=self.chain_length, warmup=self.burn_up, chains=1)

        # we save the params from the fit to calculate Fisher quantiles and their traceplots to calculate Bayesian Fisher quantiles
        traceplot_alpha1 = list(fit.extract().values())[0].tolist()
        traceplot_alpha2 = list(fit.extract().values())[1].tolist()
        traceplot_beta = list(fit.extract().values())[2].tolist()

        alpha1 = np.mean(list(fit.extract().values())[0].tolist())
        alpha2 = np.mean(list(fit.extract().values())[1].tolist())
        beta = np.mean(list(fit.extract().values())[2].tolist())

        beta0 = alpha2 / alpha1
        beta0_plugin = np.median(traceplot_alpha2) / np.median(traceplot_alpha1)

        def get_quantiles(N, u, k, quant_level, a1, a2, b):
            return u + f.isf(N/k * (1-quant_level), a1, a2, b)

        for i in range(len(self.quantile_levels)):
            if np.median(traceplot_alpha1) > 0 and np.median(traceplot_alpha2) > 0:
                plug_in_median_Fisher[i] = get_quantiles(N, u, k, self.quantile_levels[i],
                                                         a1, a2, np.median(traceplot_beta) / beta0_plugin)
            #     plug_in_median_Fisher[i] = u + f.isf(N / k * (1 - self.quantile_levels[i]),
            #                                          2 * np.median(traceplot_alpha1),
            #                                          2 * np.median(traceplot_alpha2),
            #                                          loc=0,
            #                                          scale=np.median(traceplot_beta) / beta0_plugin)
            # if alpha1 > 0 and alpha2 > 0:
                quant_Fisher[i] = u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                            2 * alpha1,
                                            2 * alpha2,
                                            loc=0,
                                            scale=beta / beta0)
            for j in range(len(traceplot_alpha1)):
                if not np.isnan(u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                          2 * traceplot_alpha1[j],
                                          2 * traceplot_alpha2[j],
                                          loc=0,
                                          scale=(traceplot_alpha1[j] * traceplot_beta[j]) / traceplot_alpha2[j])):
                    bayesian_quant_Fisher[i] += u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                                          2 * traceplot_alpha1[j],
                                                          2 * traceplot_alpha2[j],
                                                          loc=0,
                                                          scale=(traceplot_alpha1[j] * traceplot_beta[j]) /
                                                                traceplot_alpha2[j])
                if not np.isnan(u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                          2 * traceplot_alpha1[j],
                                          2 * traceplot_alpha2[j],
                                          loc=0,
                                          scale=(traceplot_alpha1[j] * traceplot_beta[j]) / traceplot_alpha2[j])):
                    median_quant_Fisher[j] = u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                                       2 * traceplot_alpha1[j],
                                                       2 * traceplot_alpha2[j],
                                                       loc=0,
                                                       scale=(traceplot_alpha1[j] * traceplot_beta[j]) / traceplot_alpha2[
                                                           j])
            all_median_quant_Fisher = np.column_stack((all_median_quant_Fisher, median_quant_Fisher))

        bayesian_quant_Fisher = bayesian_quant_Fisher / len(traceplot_alpha1)
        all_median_quant_Fisher = np.delete(all_median_quant_Fisher, 0, 1)

        store_medians = np.zeros(len(self.quantile_levels))
        # taking a median of quantiles compyted in Bayesian method
        for i in range(len(self.quantile_levels)):
            store_medians[i] = np.median(all_median_quant_Fisher[:, i])

        list_of_params = [alpha1, alpha2, beta]
        return quant_Fisher, bayesian_quant_Fisher, list_of_params, traceplot_alpha1, store_medians, plug_in_median_Fisher


