import numpy as np
from scipy.stats import f

from main import quantile_levels, n_burr_samples


def PWM_GPD(excesses: np.ndarray, threshold_u: float) -> list:
    """
    estimate parameters of the GPD using probability weighted moments
    :param excesses:
    :param threshold_u:
    :return: estimate of quantiles for the given excesses and threshold
    """
    quant_PWM_GPD = []
    n_excesses = excesses.shape[0]
    sum1, sum2 = 0, 0
    for quantile_level in range(n_excesses):
        sum1 += excesses[quantile_level]
        sum2 += quantile_level * excesses[quantile_level]
    # params
    mu0 = sum1 / n_excesses
    mu1 = sum1 / n_excesses - sum2 / n_excesses / n_excesses
    sigma = 2 * mu0 * mu1 / (mu0 - 2 * mu1)
    gamma = (mu0 - 4 * mu1) / (mu0 - 2 * mu1)

    for quantile_level in quantile_levels:  # run the calculations for each different level of quantile
        quant_PWM_GPD.append(threshold_u + sigma / gamma * (
                    pow(n_burr_samples * (1 - quantile_level) / n_excesses, -gamma) - 1))
    # for each quantile level we need to use the same n_excesses and the same threshold

    return quant_PWM_GPD


def MOM_Fisher(excesses: np.ndarray, thresholds: np.ndarray) -> tuple:
    """
    estimate parameters of the Fisher distribution using method of moments
    :param excesses:
    :param thresholds:
    :return:
    """
    quant_MOM_Fisher = np.zeros(len(quantile_levels))
    n_excesses = excesses.shape[0]
    c0 = np.mean(excesses)
    c1 = np.sum([pow(x, 1 / 2) for x in excesses]) / np.sum([pow(x, -1 / 2) for x in excesses])
    c2 = np.sum([pow(x, 3 / 4) for x in excesses]) / np.sum([pow(x, -1 / 4) for x in excesses])

    alpha2_prim = (c2 - c0) / 2 / (-c0 - c1 + 2 * c2)
    alpha1_prim = (c1 * alpha2_prim) / 2 / (c0 * (alpha2_prim - 1 / 2) - alpha2_prim * c1)

    alpha1 = alpha1_prim + 1 / 2
    alpha2 = alpha2_prim + 1 / 2
    assert alpha1 >= 0 and alpha2 >= 0, 'alpha1 or alpha2 is negative'
    beta = c1 * (alpha2 - 1 / 2) / (alpha1 - 1 / 2)
    #     print("k = ", k, "alpha1 = ", alpha1, "alpha2 = ", alpha2, "beta = ", beta, "expectation = ", c0)
    beta0 = alpha2 / alpha1
    for i in range(len(quantile_levels)):
        quant_MOM_Fisher[i] = thresholds + f.isf(n_burr_samples / n_excesses * (1 - quantile_levels[i]), 2 * alpha1,
                                                 2 * alpha2, loc=0,
                                                 scale=beta / beta0)
         # quant_MOM_Fisher[i] = u +  f.isf(N / k * (1 - q[i]), 2 * alpha1, 2 * alpha2, loc = 0, scale = beta / beta0)
    return quant_MOM_Fisher


def MOM_GPD(excesses, k, u):
    quant_MOM_GPD = np.zeros(len(quantile_levels))
    c0 = np.mean(excesses)
    c1 = np.sum([pow(x, 1 / 2) for x in excesses]) / np.sum([pow(x, -1 / 2) for x in excesses])
    # c2 = np.sum([pow(x,3/4) for x in excesses])/np.sum([pow(x,-1/4) for x in excesses])
    alpha2 = (c1 - c0) / (2 * c1 - c0)
    beta = c0 * (alpha2 - 1)
    #     alpha2 = alpha2_prim + 1/2
    for i in range(len(quantile_levels)):
        quant_MOM_GPD[i] = u + beta * (pow(n_burr_samples * (1 - quantile_levels[i]) / k, - 1 / alpha2) - 1)
    return quant_MOM_GPD
