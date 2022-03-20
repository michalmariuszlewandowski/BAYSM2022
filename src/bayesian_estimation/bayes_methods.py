import numpy as np
import pymc3 as pm
import arviz as az

from scipy.stats import f
import scipy.special as sc


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


def myGPD_lpdf(alpha, beta, value):
    """
    log of pdf of gpd used. value stands for the observations (x)
    :param alpha:
    :param beta:
    :param value:
    :return:
    """
    assert alpha >= 0 and beta >= 0, 'Invalid values of params'
    return -(alpha + 1) * (np.log(1 + value / beta)) + (np.log(alpha) - np.log(beta))


def myFisher_lpdf(alpha1, alpha2, beta, value):
    """
    log of pdf of fisher used. value stands for the observations
    :param alpha1:
    :param alpha2:
    :param beta:
    :param value:
    :return:
    """
    assert alpha1 >= 0 and alpha2 >= 0 and beta >= 0, 'Invalid values of params'
    return -sc.beta(alpha1, alpha2) - np.log(beta) + (alpha1 - 1) * np.log(value / beta) - (alpha1 + alpha2) * np.log(
        1 + value / beta)


def gpd_model(excesses: np.ndarray):
    with pm.Model() as model:
        # trying with simple priors to begin with
        alpha = pm.Gamma('alpha', 1, 1)
        beta = pm.Gamma('beta', 1, 1)
        # Create custom densities
        lGPD = pm.DensityDist('lGPD', logp=myGPD_lpdf, observed={'alpha': alpha, 'beta': beta, 'value': excesses})

    with model:
        trace = pm.sample()  # pm.Metropolis(), pm.HamiltonianMC(), pm.NUTS()  # here I have errors with Theano
    # if this suceeds, trace is the only variable of three parameters (alpha1, alpha2, beta) I need to do quantile estimation
    az.plot_trace(trace)


class bayes_methods:
    def __init__(self, data: np.ndarray, quantile_levels: list, excesses: np.ndarray, thresholds: np.ndarray):
        self.data = data  # original Burr data
        self.quantile_levels = quantile_levels
        self.excesses = excesses
        self.thresholds = thresholds
        self.n_excesses = excesses.shape[0]
        self.chain_length = 1000
        self.burn_up = 200

    def get_quantiles(self, quant_level, a1, a2, b):
        N = self.data.shape[0]
        k = self.n_excesses
        u = self.excesses
        return u + f.isf(N / k * (1 - quant_level), a1, a2, b)

    def bayes_GPD(self):
        gpd_model(self.excesses)

    def fisher_quantiles(self):
        for quantile_level in self.quantile_levels:
            pass
            # median_Fisher[i] = self.get_quantiles(self.quantile_levels[i], a1, a2, np.median(traceplot_beta) / beta0_plugin)
            # median_Fisher[i]=u+f.isf(N/k*(1-quantile_level),2*np.median(traceplot_alpha1),2*np.median(traceplot_alpha2),loc=0,scale=np.median(traceplot_beta)/beta0_plugin)

    # return params (all values after burn_up) of both GPD and Fisher
