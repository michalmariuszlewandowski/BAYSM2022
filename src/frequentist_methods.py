import numpy as np
from scipy.optimize import minimize
from scipy.stats import f, rv_continuous
import scipy.special as sc


class fisher_gen(rv_continuous):
    """
    for our parametrization of the Fisher distribution
    """

    def _cdf(self, x, beta, alpha1, alpha2):
        return sc.betainc(alpha1, alpha2, x / (beta + x)) / sc.beta(alpha1, alpha2)


class freq_methods:
    def __init__(self, data: np.ndarray, quantile_levels: list, excesses: np.ndarray, thresholds: np.ndarray):
        self.data = data  # original Burr data
        self.quantile_levels = quantile_levels
        self.excesses = excesses
        self.thresholds = thresholds
        self.n_excesses = excesses.shape[0]

    def PWM_GPD(self) -> list:
        """
        estimate parameters of the GPD using probability weighted moments
        :param excesses:
        :param thresholds:
        :return: estimate of quantiles for the given excesses and threshold
        """
        quant_PWM_GPD = []
        # n_excesses = self.excesses.shape[0]
        sum1 = np.sum(self.excesses)
        sum2 = 0
        for ind in range(self.n_excesses):
            sum2 += ind * self.excesses[ind]
        # params
        mu0 = sum1 / self.n_excesses
        mu1 = sum1 / self.n_excesses - sum2 / self.n_excesses / self.n_excesses
        sigma = 2 * mu0 * mu1 / (mu0 - 2 * mu1)
        gamma = (mu0 - 4 * mu1) / (mu0 - 2 * mu1)

        for quantile_level in self.quantile_levels:  # run the calculations for each different level of quantile
            quant_PWM_GPD.append(self.thresholds + sigma / gamma * (
                    pow(self.data.shape[1] * (1 - quantile_level) / self.n_excesses, -gamma) - 1))

        return np.array(quant_PWM_GPD).mean(axis=1)

    def empirical_moment(self, degree: float) -> np.ndarray:
        """
        estimate empirical moments of excesses
        :param degree:
        :return:
        """
        return np.sum([pow(x, degree) for x in self.excesses if x > 0])

    def estim_mom_fisher_params(self, c1: float, c2: float, c3: float) -> tuple:
        """
        from a system of equations using method of moments to get Fisher params
        """
        alpha2 = (2 * c3 + 5 * c2 - 7 * c1) / (5 * c3 + 30 * c2 - 35 * c1)
        beta = 5 * c2 - 6 * c1 - 30 * alpha2 * (c2 - c1)
        alpha1 = ((5 * alpha2 - 1) * c1 + 4 * beta) / 5 / beta
        return alpha1, alpha2, beta

    def MOM_Fisher(self) -> np.ndarray:
        """
        estimate parameters of the Fisher distribution using method of moments
        :param excesses:
        :param thresholds:
        :return:
        """
        quant_MOM_Fisher = []
        # n_excesses = excesses.shape[0]
        c1 = self.empirical_moment(1 / 5) / self.empirical_moment(-4 / 5)
        c2 = self.empirical_moment(1 / 6) / self.empirical_moment(-5 / 6)
        c3 = self.empirical_moment(2 / 5) / self.empirical_moment(-3 / 5)

        alpha1, alpha2, beta = self.estim_mom_fisher_params(c1, c2, c3)
        assert alpha1 >= 0 and alpha2 >= 0, 'alpha1 or alpha2 is negative'

        beta0 = alpha2 / alpha1  # if using fisher distribution in scipy (different parametrization than ours)
        fisher = fisher_gen(a=0.0, name='fisher')

        for quantile_level in self.quantile_levels:
            quant_MOM_Fisher.append(
                # self.thresholds + fisher.isf(self.data.shape[1] / self.n_excesses * (1 - quantile_level), beta, alpha1, alpha2)
                self.thresholds + f.isf(self.data.shape[1] / self.n_excesses * (1 - quantile_level),
                                        2 * alpha1, 2 * alpha2, loc=0, scale=beta / beta0)
            )
        return np.array(quant_MOM_Fisher).mean(axis=1)

    def MOM_GPD(self) -> np.ndarray:
        # n_excesses = excesses.shape[0]
        quant_MOM_GPD = []

        c0 = np.mean(self.excesses)
        c1 = self.empirical_moment(1 / 2) / self.empirical_moment(-1 / 2)

        alpha2 = (c1 - c0) / (2 * c1 - c0)
        beta = c0 * (alpha2 - 1)

        for quantile_level in self.quantile_levels:
            quant_MOM_GPD.append(
                self.thresholds + beta *
                (pow(self.data.shape[1] * (1 - quantile_level) / self.n_excesses, - 1 / alpha2) - 1)
            )
        return np.array(quant_MOM_GPD).mean(axis=1)

    def _mle_equations_for_gpd(self, x, df):
        """
        equations for gpd which we solve using a numerical solver. These equations arise from MLE
        :param x:
        :param df:
        :return:
        """
        kk = len(df)  # what was kk?
        log_arg = np.sum([np.log(1 + x * y) for y in df])
        return kk * np.log(1 / kk / x * log_arg) + log_arg + kk

    def MLE_GPD(self) -> np.ndarray:
        """
        estimate extreme quantiles of GPD by MLE
        :param excesses:
        :param n_excesses:
        :param thresholds:
        :return:
        """
        gpd_by_mle = []
        # n_excesses = excesses.shape[0]
        # constraints: x > 0
        contraints = ({'type': 'ineq', 'fun': lambda x: x - 1e-6})

        tau = minimize(self._mle_equations_for_gpd, 2, args=self.excesses, constraints=contraints, method='SLSQP')

        gamma = 1 / self.excesses.shape[0] * np.sum([np.log(1 + tau.x * y) for y in self.excesses])
        sigma = gamma / tau.x

        for quantile_level in self.quantile_levels:
            gpd_by_mle.append(
                self.thresholds + sigma / gamma * (
                        pow(self.data.shape[1] * (1 - quantile_level) / self.n_excesses, - gamma) - 1)
            )

        return np.array(gpd_by_mle).mean(axis=1)
