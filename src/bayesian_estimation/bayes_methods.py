import numpy as np
from pystan import StanModel
from scipy.stats import f

# beta_frechet = 1/2
# quant_th = np.zeros(len(q))
# for i in range(len(q)):
#     quant_th[i] = pow(-log(q[i]), -beta_frechet)
from src.bayesian_estimation.pystan_models import GPD, Fisher


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

        for i in range(len(self.quantile_levels)):
            if np.median(traceplot_alpha1) > 0 and np.median(traceplot_alpha2) > 0:
                plug_in_median_Fisher[i] = u + f.isf(N / k * (1 - self.quantile_levels[i]),
                                                     2 * np.median(traceplot_alpha1),
                                                     2 * np.median(traceplot_alpha2),
                                                     loc=0,
                                                     scale=np.median(traceplot_beta) / beta0_plugin)
            if alpha1 > 0 and alpha2 > 0:
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


