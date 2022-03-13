import numpy
import numpy as np
import pandas as pd

import datetime
from pathlib import Path
from arg_parser import main_args
from src.data.make_data import create_burr_data


def compute_true_quantiles(quantile_levels: list, burr_beta: float, burr_lambda: float, burr_tau: float) -> list:
    """
    Compute true quantiles which we will compare with the true ones
    :param burr_tau:
    :param burr_lambda:
    :param burr_beta:
    :return:
    """
    true_quantiles = []

    for quantile_level in quantile_levels:
        true_quantiles.append(pow(burr_beta / pow(1 - quantile_level, 1 / burr_lambda) - burr_beta, 1 / burr_tau))
    return true_quantiles


def k_greatest_values_matrices(input_matrix: numpy.ndarray, n_excesses: int) -> tuple:
    """
    :param input_matrix: sorted matrix, observations stacked row-wise
    :return: k greatest elements from the input and (k - 1)th value (1st extreme)
    """
    excesses = input_matrix[:, -n_excesses:]
    thresholds = input_matrix[:, -n_excesses]
    return (excesses.T - thresholds).T, thresholds


def estimate_quantiles_frequentist_methods(args, burr_data):
    from src.frequentist_methods import freq_methods
    # How many values do we want to consider as excesses. The more, the better approximation should we obtain
    n_excesses_ = np.linspace(100, 500, args.n_different_thresholds).astype(int)

    keep_quantiles = {'pwm_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mom_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mom_fisher': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mle_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T}

    for ind, n_excesses in enumerate(n_excesses_):  # for different number of excesses
        shifted_excesses, thresholds = k_greatest_values_matrices(burr_data, n_excesses)

        qnts = np.zeros((len(keep_quantiles), len(args.quantile_levels)))

        for excesses, threshold in zip(shifted_excesses, thresholds):
            frequentist = freq_methods(data=burr_data, quantile_levels=args.quantile_levels,
                                       excesses=excesses, thresholds=thresholds)
            # fit GPD and Fisher distributions to excesses from each dataset
            qnts[0] += frequentist.PWM_GPD()
            qnts[1] += frequentist.MOM_GPD()
            qnts[2] += frequentist.MLE_GPD()
            qnts[3] += frequentist.MOM_Fisher()

        keep_quantiles.get('pwm_gpd')[ind] = qnts[0] / args.n_different_samples
        keep_quantiles.get('mom_gpd')[ind] = qnts[1] / args.n_different_samples
        keep_quantiles.get('mle_gpd')[ind] = qnts[2] / args.n_different_samples
        keep_quantiles.get('mom_fisher')[ind] = qnts[3] / args.n_different_samples

    return keep_quantiles


def l2_norm(arr: np.ndarray, ind: int, true_quantile: float):
    """
    l2 norm of (scaled) differences between the true quantile and its estimate
    :param arr:
    :param ind:
    :param true_quantile:
    :return:
    """
    assert 0 < true_quantile < 1, 'Quantile level out of range'
    squared_norm_ss = np.sum([pow(x - true_quantile, 2) for x in arr[ind, :]]) / pow(true_quantile, 2)

    return np.sqrt(squared_norm_ss / arr.shape[1])  # arr.shape should be the nb of different thresholds tried?


def main(args):
    # load data
    save_dir = Path.cwd() / 'src/data/data_burr/'
    file_name = save_dir.__str__() + f'\\beta_{args.burr_beta}__tau_{args.burr_tau}__lambda_{args.burr_lambda}__n_samples_{args.n_different_samples}__n_obs_per_sample_{args.n_points_each_sample}.npy'
    with open(file_name, 'rb') as f:
        burr_data = np.load(f)
    import matplotlib.pyplot as plt
    # take random sample
    ind = int(np.random.uniform(burr_data.shape[0], size=1))
    plt.hist(burr_data[ind], bins='auto')
    plt.title(f'Empirical mean {round(np.mean(burr_data[ind]),3)}, '
              f'var {round(np.var(burr_data[ind]), 3)}')
    plt.show()

    keep_quantiles = estimate_quantiles_frequentist_methods(args, burr_data)
    # qnts_pwm_gpd, qnts_mom_gpd, qnts_mom_fisher, qnts_mle_gpd
    # columns: different nb of excesses used to estimate quantiles
    # rows: quantiles of different levels

    # now for testing how good the fit is
    MOM_Fisher = np.zeros(len(args.quantile_levels))
    MOM_GPD = np.zeros(len(args.quantile_levels))
    MLE_GPD = np.zeros(len(args.quantile_levels))
    PWM_GPD = np.zeros(len(args.quantile_levels))

    for ind in range(len(args.quantile_levels)):
        # q[i] - current level - get all distance of all methods for this level
        # this is already for testing how good the estimate is
        MOM_Fisher[ind] = l2_norm(keep_quantiles.get('mom_fisher'), ind, args.quantile_levels[ind])
        MOM_GPD[ind] = l2_norm(keep_quantiles.get('mom_gpd'), ind, args.quantile_levels[ind])
        MLE_GPD[ind] = l2_norm(keep_quantiles.get('mle_gpd'), ind, args.quantile_levels[ind])
        PWM_GPD[ind] = l2_norm(keep_quantiles.get('pwm_gpd'), ind, args.quantile_levels[ind])


if __name__ == '__main__':
    beginning = datetime.datetime.now()
    print('Starting')
    args = main_args()
    main(args)
    ending = datetime.datetime.now()

    print('Time: ', ending - beginning)
