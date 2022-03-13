import numpy as np
from scipy.optimize import minimize
from scipy.stats import f


class freq_methods:
    def __init__(self, data: np.ndarray, quantile_levels: list, excesses: np.ndarray, thresholds: np.ndarray):
        self.data = data  # original full data, not excesses
        self.quantile_levels = quantile_levels
        self.excesses = excesses
        self.thresholds = thresholds
        self.n_excesses = excesses.shape[0]
        # I can add here a function which returns excesses and thresholds

    def PWM_GPD(self) -> list:
        """
        estimate parameters of the GPD using probability weighted moments
        :param excesses:
        :param thresholds:
        :return: estimate of quantiles for the given excesses and threshold
        """
        quant_PWM_GPD = []
        # n_excesses = self.excesses.shape[0]
        sum1, sum2 = 0, 0
        for quantile_level in range(self.n_excesses):
            sum1 += self.excesses[quantile_level]
            sum2 += quantile_level * self.excesses[quantile_level]
        # params
        mu0 = sum1 / self.n_excesses
        mu1 = sum1 / self.n_excesses - sum2 / self.n_excesses / self.n_excesses
        sigma = 2 * mu0 * mu1 / (mu0 - 2 * mu1)
        gamma = (mu0 - 4 * mu1) / (mu0 - 2 * mu1)

        for quantile_level in self.quantile_levels:  # run the calculations for each different level of quantile
            quant_PWM_GPD.append(self.thresholds + sigma / gamma * (
                        pow(self.data.shape[1] * (1 - quantile_level) / self.n_excesses, -gamma) - 1))
        # for each quantile level we need to use the same n_excesses and the same threshold

        return quant_PWM_GPD

    def MOM_Fisher(self) -> list:
        """
        estimate parameters of the Fisher distribution using method of moments
        :param excesses:
        :param thresholds:
        :return:
        """
        quant_MOM_Fisher = []
        # n_excesses = excesses.shape[0]
        c0 = np.mean(self.excesses)
        c1 = np.sum([pow(x, 1 / 2) for x in self.excesses]) / np.sum([pow(x, -1 / 2) for x in self.excesses])
        c2 = np.sum([pow(x, 3 / 4) for x in self.excesses]) / np.sum([pow(x, -1 / 4) for x in self.excesses])

        alpha2_prim = (c2 - c0) / 2 / (-c0 - c1 + 2 * c2)
        alpha1_prim = (c1 * alpha2_prim) / 2 / (c0 * (alpha2_prim - 1 / 2) - alpha2_prim * c1)

        alpha1 = alpha1_prim + 1 / 2
        alpha2 = alpha2_prim + 1 / 2
        assert alpha1 >= 0 and alpha2 >= 0, 'alpha1 or alpha2 is negative'
        beta = c1 * (alpha2 - 1 / 2) / (alpha1 - 1 / 2)
        #     print("k = ", k, "alpha1 = ", alpha1, "alpha2 = ", alpha2, "beta = ", beta, "expectation = ", c0)
        beta0 = alpha2 / alpha1
        for i in range(len(self.quantile_levels)):
            quant_MOM_Fisher.append(self.thresholds + f.isf(self.data.shape[1] / self.n_excesses * (1 - self.quantile_levels[i]), 2 * alpha1,
                                                     2 * alpha2, loc=0,
                                                     scale=beta / beta0))
             # quant_MOM_Fisher[i] = u +  f.isf(N / k * (1 - q[i]), 2 * alpha1, 2 * alpha2, loc = 0, scale = beta / beta0)
        return quant_MOM_Fisher

    def MOM_GPD(self)->list:
        # n_excesses = excesses.shape[0]
        quant_MOM_GPD = []
        c0 = np.mean(self.excesses)
        c1 = np.sum([pow(x, 1 / 2) for x in self.excesses]) / np.sum([pow(x, -1 / 2) for x in self.excesses])
        # c2 = np.sum([pow(x,3/4) for x in excesses])/np.sum([pow(x,-1/4) for x in excesses])
        alpha2 = (c1 - c0) / (2 * c1 - c0)
        beta = c0 * (alpha2 - 1)
        #     alpha2 = alpha2_prim + 1/2
        for i in range(len(self.quantile_levels)):
            quant_MOM_GPD.append(self.thresholds + beta * (pow(self.data.shape[1] * (1 - self.quantile_levels[i]) / self.n_excesses, - 1 / alpha2) - 1))
        return quant_MOM_GPD

    def _mle_equations_for_gpd(self, x, df):
        """
        equations for gpd which we solve using a numerical solver. These equations arise from MLE
        :param x:
        :param df:
        :return:
        """
        kk = len(df)  # todo what was kk?
        log_arg = np.sum([np.log(1 + x * y) for y in df])
        return kk * np.log(1 / kk / x * log_arg) + log_arg + kk

    def MLE_GPD(self) -> list:
        """
        estimate extreme quantiles of GPD by MLE
        :param excesses:
        :param n_excesses:
        :param thresholds:
        :return:
        """
        quantiles_gpd_by_mle = []
        # n_excesses = excesses.shape[0]
        # constraints: x > 0
        contraints = ({'type': 'ineq', 'fun': lambda x: x - 1e-6})

        tau = minimize(self._mle_equations_for_gpd, 2, args=self.excesses, constraints=contraints, method='SLSQP')

        gamma = 1 / self.excesses.shape[0] * np.sum([np.log(1 + tau.x * y) for y in self.excesses])
        sigma = gamma / tau.x
        print('alpha = ', 1 / gamma, '\n beta = ', sigma / gamma)

        for quantile_level in self.quantile_levels:
            quantiles_gpd_by_mle.append(
                self.thresholds + sigma / gamma * (
                        pow(self.data.shape[1] * (1 - quantile_level) / self.n_excesses, - gamma) - 1)
            )

        return quantiles_gpd_by_mle