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
    parser.add_argument('--burr_beta', default=1/4, type=float)
    parser.add_argument('--burr_tau', default=1, type=float)
    parser.add_argument('--burr_lambda', default=4., type=float)

    parser.add_argument('--quantiles', default=[0.98, 0.99, 0.995, 0.999, 0.9995],
                        nargs='+', type=float,
                        help='Extreme quantiles we check')

    return parser.parse_args()


class burr_gen(rv_continuous):
    """
    simulate iid burr observations
    """
    def _cdf(self, x, beta, tau, Lambda):
        return 1 - pow(beta / (beta + pow(x, tau)), Lambda)


def compute_true_quantiles(levels: list, burr_beta: float, burr_lambda: float, burr_tau: float) -> list:
    """
    Compute true quantiles which we will compare with the true ones
    :param levels:
    :return:
    """
    true_quantiles = []

    for quantile in levels:
        true_quantiles.append(pow(burr_beta / pow(1 - quantile, 1 / burr_lambda) - burr_beta, 1 / burr_tau))
    return true_quantiles


def mle_equations_for_gpd(x, df):
    """
    equations for gpd which we solve using a numerical solver. These equations arise from MLE
    :param x:
    :param df:
    :return:
    """
    kk = len(df)
    log_arg = np.sum([np.log(1 + x * y) for y in df])
    return kk * np.log(1 / kk / x * log_arg) + log_arg + kk


def estimate_gpd_params_with_mle(excesses, k, u):
    """
    numerically solve MLE for GPD
    :param excesses:
    :param k:
    :param u:
    :return:
    """
    quant_MLE_GPD = np.zeros(len(quantile_levels))
    # constraints: x > 0
    abstol = 1e-6
    cons = ({'type': 'ineq', 'fun': lambda x: x - abstol})

    tau = minimize(mle_equations_for_gpd, 2, args=excesses, constraints=cons, method='SLSQP')
    gamma = 1 / len(excesses) * np.sum([np.log(1 + tau.x * y) for y in excesses])
    sigma = gamma / tau.x
    print('alpha = ', 1 / gamma, '\n beta = ', sigma / gamma)

    for i in range(len(quantile_levels)):
        quant_MLE_GPD[i] = u + sigma / gamma * (pow(n_burr_samples * (1 - quantile_levels[i]) / k, - gamma) - 1)

    return quant_MLE_GPD


def k_greatest_values_matrices(input_matrix: numpy.ndarray, n_excesses: int) -> tuple:
    """
    :input sorted matrix
    returns k greatest elements from the input and (k - 1)th value (1st extreme)
    """
    return input_matrix.T[-n_excesses:].T, input_matrix.T[-n_excesses:].T[0]

quantile_levels = [0.98, 0.99, 0.995, 0.999, 0.9995]
n_burr_samples = 1000


def main(args):
    from src.frequentist_methods import PWM_GPD, MOM_Fisher, MOM_GPD

    burr = burr_gen(a=0.0, name='burr')  # specify support [a,b], no b means b = infinity

    true_quantiles = compute_true_quantiles(quantile_levels, args.burr_beta, args.burr_lambda, args.burr_tau)
    burr_observations_holder = np.zeros((args.n_different_samples, args.n_points_each_sample))

    # todo it takes a lot of time, better create once and store somewhere
    for i in range(args.n_different_samples):
        rv_burr = burr.rvs(args.burr_beta, args.burr_tau, args.burr_lambda, size=args.n_points_each_sample)
        burr_observations_holder[i] = np.sort(rv_burr)

    # How many values do we want to consider as excesses. The more, the better approximation should we obtain
    n_excesses = np.linspace(100, 500, args.n_different_thresholds).astype(int)

    # probability_weighted_moments_GPD = np.zeros(len(q))
    # todo for sure there is smarted way of storing these data
    concatenated_PWM_GPD = np.zeros(len(quantile_levels))
    concatenated_MOM_Fisher = np.zeros(len(quantile_levels))
    concatenated_MOM_GPD = np.zeros(len(quantile_levels))
    concatenated_MLE_GPD = np.zeros(len(quantile_levels))

    for j in range(len(n_excesses)):  # for different threshold of excesses
        k = n_excesses[j]
        data_frechet, u = k_greatest_values_matrices(burr_observations_holder, k)
        # delete first column (indexed by 0) of a matrix A, to match the sizes
        A = np.delete(data_frechet, 0, 1)
        # from the array of u values we create matrix, in columns we have repeated u values
        B = [[x] * k for x in u]
        # here we subtract u_i from excesses in each dataset
        C = np.array(A) - np.array(B).transpose()

        averaged_PWM_GPD = np.zeros(len(quantile_levels))
        averaged_MOM_Fisher = np.zeros(len(quantile_levels))
        averaged_MOM_GPD = np.zeros(len(quantile_levels))
        averaged_MLE_GPD = np.zeros(len(quantile_levels))
        #     counter = np.zeros(len(q))
        counter = 0

        for ind in range(args.n_different_samples):
            # we fit GPD and Fisher distributions to excesses from each dataset
            excesses_array = C[:, ind]
            averaged_PWM_GPD += 1 / args.n_different_samples * PWM_GPD(excesses_array, k, u[ind])
            averaged_MOM_Fisher += 1 / args.n_different_samples * MOM_Fisher(excesses_array, k, u[ind])[0]
            averaged_MOM_GPD += 1 / args.n_different_samples * MOM_GPD(excesses_array, k, u[ind])
            averaged_MLE_GPD += 1 / args.n_different_samples * estimate_gpd_params_with_mle(excesses_array, k, u[ind])
            counter += MOM_Fisher(excesses_array, k, u[ind])[1]

        concatenated_MOM_GPD = np.column_stack((concatenated_MOM_GPD, averaged_MOM_GPD))
        concatenated_MOM_Fisher = np.column_stack((concatenated_MOM_Fisher, averaged_MOM_Fisher * args.n_different_samples / counter)) #*n/counter
        concatenated_PWM_GPD = np.column_stack((concatenated_PWM_GPD, averaged_PWM_GPD))
        concatenated_MLE_GPD = np.column_stack((concatenated_MLE_GPD, averaged_MLE_GPD))

    # we delete first column which was created as zeros
    # we need those to plot quantiles on one plot and to compare them
    concatenated_MOM_GPD = np.delete(concatenated_MOM_GPD, 0, 1)
    concatenated_PWM_GPD = np.delete(concatenated_PWM_GPD, 0, 1)
    concatenated_MOM_Fisher = np.delete(concatenated_MOM_Fisher, 0, 1)
    concatenated_MLE_GPD = np.delete(concatenated_MLE_GPD, 0, 1)

    for ind in range(len(quantile_levels)):

        fig = plt.figure()

        plt.hlines(y=true_quantiles[ind],
                   xmin=min(n_excesses),
                   xmax=max(n_excesses),
                   color='gray',
                   zorder=1,
                   label='theoretical value')

        plt.plot(n_excesses, concatenated_MOM_Fisher[ind, :], 'red', label='MOM Fisher')
        plt.plot(n_excesses, concatenated_MOM_GPD[ind, :], 'deepskyblue', label='MOM GPD')
        plt.plot(n_excesses, concatenated_PWM_GPD[ind, :], 'aqua', label='PWM GPD')
        plt.plot(n_excesses, concatenated_MLE_GPD[ind, :], 'mediumblue', label='MLE GPD')

        plt.xlabel('number of excesses')
        plt.ylabel('value of quantile at level ' + str(quantile_levels[ind]))
        plt.title('Variability of quantile at level ' + str(quantile_levels[ind]))
        plt.legend()

    plt.show()

    MOM_Fisher = np.zeros(len(quantile_levels))
    MOM_GPD = np.zeros(len(quantile_levels))
    MLE_GPD = np.zeros(len(quantile_levels))
    PWM_GPD = np.zeros(len(quantile_levels))

    for ind in range(len(quantile_levels)):
        # todo this can be put in function
        # q[i] - current level - get all distance of all methods for this level
        MOM_Fisher[ind] = np.sqrt(1 / args.n_different_thresholds * np.nansum(
            [pow(x - true_quantiles[ind], 2) for x in concatenated_MOM_Fisher[ind, :]]) / pow(true_quantiles[ind],
                                                                                                        2))  # nansum - treating nans as zero
        MOM_GPD[ind] = np.sqrt(1 / args.n_different_thresholds * np.nansum(
            [pow(x - true_quantiles[ind], 2) for x in concatenated_MOM_GPD[ind, :]]) / pow(true_quantiles[ind], 2))
        MLE_GPD[ind] = np.sqrt(1 / args.n_different_thresholds * np.nansum(
            [pow(x - true_quantiles[ind], 2) for x in concatenated_MLE_GPD[ind, :]]) / pow(true_quantiles[ind], 2))
        PWM_GPD[ind] = np.sqrt(1 / args.n_different_thresholds * np.nansum(
            [pow(x - true_quantiles[ind], 2) for x in concatenated_PWM_GPD[ind, :]]) / pow(true_quantiles[ind], 2))
        # plot_table(q[i], MLE_GPD, PWM_GPD, MOM_GPD, MOM_Fisher)

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
    # print(df.to_latex())
    print(df.T.to_latex())

    for ind in range(len(quantile_levels)):
        print(concatenated_MOM_Fisher[ind, :])


if __name__ == '__main__':
    beginning = datetime.datetime.now()
    print('Starting')
    args = main_args()
    main(args)
    ending = datetime.datetime.now()

    print('Time', (ending-beginning).seconds)
