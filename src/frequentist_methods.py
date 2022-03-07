import numpy as np
from scipy.stats import f

from main import quantile_levels, n_burr_samples


def PWM_GPD(excesses, k, u):
    quant_PWM_GPD = np.zeros(len(quantile_levels))
    sum1, sum2 = 0, 0
    for i in range(k):
        sum1 += excesses[i]
        sum2 += i * excesses[i]
    mu0 = sum1 / k
    mu1 = sum1 / k - sum2 / k / k
    sigma = 2 * mu0 * mu1 / (mu0 - 2 * mu1)
    gamma = (mu0 - 4 * mu1) / (mu0 - 2 * mu1)
    for i in range(len(quantile_levels)):
        quant_PWM_GPD[i] = u + sigma / gamma * (pow(n_burr_samples * (1 - quantile_levels[i]) / k, -gamma) - 1)
    return (quant_PWM_GPD)


def MOM_Fisher(excesses, k, u):
    quant_MOM_Fisher = np.zeros(len(quantile_levels))
    c0 = np.mean(excesses)
    c1 = np.sum([pow(x, 1 / 2) for x in excesses]) / np.sum([pow(x, -1 / 2) for x in excesses])
    c2 = np.sum([pow(x, 3 / 4) for x in excesses]) / np.sum([pow(x, -1 / 4) for x in excesses])

    alpha2_prim = (c2 - c0) / 2 / (-c0 - c1 + 2 * c2)
    alpha1_prim = (c1 * alpha2_prim) / 2 / (c0 * (alpha2_prim - 1 / 2) - alpha2_prim * c1)

    alpha1 = alpha1_prim + 1 / 2
    alpha2 = alpha2_prim + 1 / 2
    beta = c1 * (alpha2 - 1 / 2) / (alpha1 - 1 / 2)
    #     print("k = ", k, "alpha1 = ", alpha1, "alpha2 = ", alpha2, "beta = ", beta, "expectation = ", c0)
    beta0 = alpha2 / alpha1
    counter = 0
    for i in range(len(quantile_levels)):
        if alpha1 > 0 and alpha2 > 0:
            quant_MOM_Fisher[i] = u + f.isf(n_burr_samples / k * (1 - quantile_levels[i]), 2 * alpha1, 2 * alpha2, loc=0, scale=beta / beta0)
            counter += 1
    #         quant_MOM_Fisher[i] = u +  f.isf(N / k * (1 - q[i]), 2 * alpha1, 2 * alpha2, loc = 0, scale = beta / beta0)
    return ([quant_MOM_Fisher, counter / 5])


def MOM_GPD(excesses, k, u):
    quant_MOM_GPD = np.zeros(len(quantile_levels))
    c0 = np.mean(excesses)
    c1 = np.sum([pow(x, 1 / 2) for x in excesses]) / np.sum([pow(x, -1 / 2) for x in excesses])
    #     c2 = np.sum([pow(x,3/4) for x in excesses])/np.sum([pow(x,-1/4) for x in excesses])
    alpha2 = (c1 - c0) / (2 * c1 - c0)
    beta = c0 * (alpha2 - 1)
    #     alpha2 = alpha2_prim + 1/2
    for i in range(len(quantile_levels)):
        quant_MOM_GPD[i] = u + beta * (pow(n_burr_samples * (1 - quantile_levels[i]) / k, - 1 / alpha2) - 1)
    return (quant_MOM_GPD)