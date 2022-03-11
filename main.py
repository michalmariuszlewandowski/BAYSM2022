import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import rv_continuous
import datetime
from scipy.optimize import minimize
import argparse


def main_args():
    parser = argparse.ArgumentParser(description='Parsing Arguments.')

    # general params
    parser.add_argument('--n_points_each_sample', default=1000, type=int, help='')
    parser.add_argument('--n_excesses', default=100, type=int)
    parser.add_argument('--n_different_thresholds', default=15, type=int, help='for how many thresholds do we test')
    parser.add_argument('--n_different_samples', default=5,
                        type=int, help='how many time we average over the result')
    # burr params
    parser.add_argument('--burr_beta', default=.25, type=float)
    parser.add_argument('--burr_tau', default=1., type=float)
    parser.add_argument('--burr_lambda', default=4., type=float)

    parser.add_argument('--quantiles',
                        default=[0.98, 0.99, 0.995, 0.999, 0.9995],
                        nargs='+', type=float,
                        help='Extreme quantiles we check')

    return parser.parse_args()


class burr_gen(rv_continuous):
    """
    simulate iid burr observations
    """

    def _cdf(self, x, beta, tau, Lambda):
        return 1 - pow(beta / (beta + pow(x, tau)), Lambda)


def compute_true_quantiles(quantile_levels: list, burr_beta: float, burr_lambda: float, burr_tau: float) -> list:
    """
    Compute true quantiles which we will compare with the true ones
    :param burr_tau:
    :param burr_lambda:
    :param burr_beta:
    :param quantile_levels:
    :return:
    """
    true_quantiles = []

    for quantile in quantile_levels:
        true_quantiles.append(pow(burr_beta / pow(1 - quantile, 1 / burr_lambda) - burr_beta, 1 / burr_tau))
    return true_quantiles


class estimate_gpd_with_mle:
    def __init__(self):
        pass


def mle_equations_for_gpd(x, df):
    """
    equations for gpd which we solve using a numerical solver. These equations arise from MLE
    :param x:
    :param df:
    :return:
    """
    kk = len(df)  # todo what was kk?
    log_arg = np.sum([np.log(1 + x * y) for y in df])
    return kk * np.log(1 / kk / x * log_arg) + log_arg + kk


def qnts_gpd_mle(excesses: np.ndarray, threshold: float) -> list:
    """
    estimate extreme quantiles of GPD by MLE
    :param excesses:
    :param n_excesses:
    :param threshold:
    :return:
    """
    quantiles_gpd_by_mle = []
    n_excesses = excesses.shape[0]
    # constraints: x > 0
    contraints = ({'type': 'ineq', 'fun': lambda x: x - 1e-6})

    tau = minimize(mle_equations_for_gpd, 2, args=excesses, constraints=contraints, method='SLSQP')

    gamma = 1 / excesses.shape[0] * np.sum([np.log(1 + tau.x * y) for y in excesses])
    sigma = gamma / tau.x
    print('alpha = ', 1 / gamma, '\n beta = ', sigma / gamma)

    for quantile_level in quantile_levels:
        quantiles_gpd_by_mle.append(
            threshold + sigma / gamma * (
                    pow(n_burr_samples * (1 - quantile_level) / n_excesses, - gamma) - 1)
        )

    return quantiles_gpd_by_mle


def k_greatest_values_matrices(input_matrix: numpy.ndarray, n_excesses: int) -> tuple:
    """
    :param input_matrix: sorted matrix, observations stacked row-wise
    :return: k greatest elements from the input and (k - 1)th value (1st extreme)
    """
    excesses = input_matrix[:, -n_excesses:]
    thresholds = input_matrix[:, -n_excesses]
    return (excesses.T - thresholds).T, thresholds


quantile_levels = [0.98, 0.99, 0.995, 0.999, 0.9995]  #
n_burr_samples = 1000  # todo rename to n_points_each_sample


def create_burr_data(n_different_samples: int, n_points_each_sample: int, burr_params: dict) -> np.ndarray:
    burr = burr_gen(a=0.0, name='burr')  # specify support [a,b], no b means b = infinity

    burr_observations_holder = np.zeros((n_different_samples, n_points_each_sample))

    # todo it takes a lot of time, better create once and store somewhere
    for i in range(n_different_samples):
        rv_burr = burr.rvs(
            burr_params.get('burr_beta'), burr_params.get('burr_tau'), burr_params.get('burr_lambda'),
            size=n_points_each_sample
        )
        burr_observations_holder[i] = np.sort(rv_burr)

    return burr_observations_holder


def estimate_quantiles_frequentist_methods(args):
    from src.frequentist_methods import PWM_GPD, MOM_Fisher, MOM_GPD

    # todo put to a separate function as a data maker, and here just load the data
    burr_observations_holder = create_burr_data(
        args.n_different_samples, args.n_points_each_sample,
        {
            'burr_beta': args.burr_beta, 'burr_tau': args.burr_tau, 'burr_lambda': args.burr_lambda
        }
    )

    # How many values do we want to consider as excesses. The more, the better approximation should we obtain
    n_excesses_ = np.linspace(100, 500, args.n_different_thresholds).astype(int)

    # place holder
    keep_quantiles = {'pwm_gpd': np.zeros((len(quantile_levels), n_excesses_)).T,
                      'mom_gpd': np.zeros((len(quantile_levels), n_excesses_)).T,
                      'mom_fisher': np.zeros((len(quantile_levels), n_excesses_)).T,
                      'mle_gpd': np.zeros((len(quantile_levels), n_excesses_)).T}

    for ind, n_excesses in enumerate(n_excesses_):  # for different number of excesses
        # n_excesses = n_excesses_[j]
        shifted_excesses, thresholds = k_greatest_values_matrices(burr_observations_holder, n_excesses)

        qnts_pwm_gpd = np.zeros(len(quantile_levels))
        qnts_mom_fisher = np.zeros(len(quantile_levels))
        qnts_mom_gpd = np.zeros(len(quantile_levels))
        qnts_mle_gpd = np.zeros(len(quantile_levels))

        for excesses, threshold in zip(shifted_excesses, thresholds):
            # fit GPD and Fisher distributions to excesses from each dataset
            # todo put these methods as a class
            qnts_pwm_gpd += PWM_GPD(excesses, threshold)
            qnts_mom_fisher += MOM_Fisher(excesses, threshold)[0]
            qnts_mom_gpd += MOM_GPD(excesses, threshold)
            qnts_mle_gpd += qnts_gpd_mle(excesses, threshold)

        keep_quantiles.get('pwm_gpd')[ind] = qnts_pwm_gpd / args.n_different_samples
        keep_quantiles.get('mom_gpd')[ind] = qnts_mom_gpd / args.n_different_samples
        keep_quantiles.get('mom_fisher')[ind] = qnts_mom_fisher / args.n_different_samples
        keep_quantiles.get('mle_gpd')[ind] = qnts_mle_gpd / args.n_different_samples

    return keep_quantiles


def plot_quantiles(qnts_pwm_gpd, qnts_mom_gpd, qnts_mom_fisher, qnts_mle_gpd):
    """
    for plotting the results against true quantiles
    :param qnts_pwm_gpd:
    :param qnts_mom_gpd:
    :param qnts_mom_fisher:
    :param qnts_mle_gpd:
    :return:
    """
    true_quantiles = compute_true_quantiles(quantile_levels, args.burr_beta, args.burr_lambda, args.burr_tau)
    n_excesses_ = qnts_pwm_gpd.shape[1]  # nb of columns in any above
    for ind in range(len(quantile_levels)):
        fig, ax = plt.subplots()

        ax.hlines(y=true_quantiles[ind],
                  xmin=min(n_excesses_),
                  xmax=max(n_excesses_),
                  color='gray',
                  zorder=1,
                  label='theoretical value')

        ax.plot(n_excesses_, qnts_mom_fisher[ind, :], 'red', label='MOM Fisher')
        ax.plot(n_excesses_, qnts_mom_gpd[ind, :], 'deepskyblue', label='MOM GPD')
        ax.plot(n_excesses_, qnts_pwm_gpd[ind, :], 'aqua', label='PWM GPD')
        ax.plot(n_excesses_, qnts_mle_gpd[ind, :], 'mediumblue', label='MLE GPD')

        ax.xlabel('number of excesses')
        ax.ylabel('value of quantile at level ' + str(quantile_levels[ind]))
        ax.title('Variability of quantile at level ' + str(quantile_levels[ind]))
        ax.legend()

        fig.show()

    print('Plotted results')


def helper_norm_ss(arr: np.ndarray, ind: int, true_quantile: float):
    assert 0 < true_quantile < 1, 'Quantile level out of range'
    squared_norm_ss = np.sum([pow(x - true_quantile, 2) for x in arr[ind, :]]) / pow(true_quantile, 2)

    return np.sqrt(squared_norm_ss / arr.shape[1])  # arr.shape should be the nb of different thresholds tried?


def main(args):

    keep_quantiles = estimate_quantiles_frequentist_methods(args)
    # qnts_pwm_gpd, qnts_mom_gpd, qnts_mom_fisher, qnts_mle_gpd
    # columns: different nb of excesses used to estimate quantiles
    # rows: quantiles of different levels

    # now for testing how good the fit is
    MOM_Fisher = np.zeros(len(quantile_levels))
    MOM_GPD = np.zeros(len(quantile_levels))
    MLE_GPD = np.zeros(len(quantile_levels))
    PWM_GPD = np.zeros(len(quantile_levels))

    for ind in range(len(quantile_levels)):
        # q[i] - current level - get all distance of all methods for this level
        # this is already for testing how good the estimate is
        MOM_Fisher[ind] = helper_norm_ss(keep_quantiles.get('mom_fisher'), ind, quantile_levels[ind])
        MOM_GPD[ind] = helper_norm_ss(keep_quantiles.get('mom_gpd'), ind, quantile_levels[ind])
        MLE_GPD[ind] = helper_norm_ss(keep_quantiles.get('mle_gpd'), ind, quantile_levels[ind])
        PWM_GPD[ind] = helper_norm_ss(keep_quantiles.get('pwm_gpd'), ind, quantile_levels[ind])

    M = [[round(x, 6) for x in MOM_Fisher],
         [round(x, 6) for x in MOM_GPD],
         [round(x, 6) for x in MLE_GPD],
         [round(x, 6) for x in PWM_GPD]]

    d = {'MOM Fisher': M[0][1:],
         'MOM GPD': M[1][1:],
         'MLE GPD': M[2][1:],
         'PWM GPD': M[3][1:]}
    #      'col5': [round(x, rounding) for x in MLE_GPD]}
    df = pd.DataFrame(data=d, index=['0.99', '0.995', '0.999', '0.9995'])
    print(df.T.to_latex())

    # for ind in range(len(quantile_levels)):
    #     print(qnts_mom_fisher[ind, :])


if __name__ == '__main__':
    beginning = datetime.datetime.now()
    print('Starting')
    args = main_args()
    main(args)
    ending = datetime.datetime.now()

    print('Time: ', ending - beginning)
