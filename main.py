import numpy
import numpy as np
import pandas as pd

import datetime
from pathlib import Path
from arg_parser import main_args
from src.data.make_data import create_burr_data
from src.frequentist_methods import qnts_gpd_mle


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


class estimate_gpd_with_mle:
    def __init__(self):
        pass


def k_greatest_values_matrices(input_matrix: numpy.ndarray, n_excesses: int) -> tuple:
    """
    :param input_matrix: sorted matrix, observations stacked row-wise
    :return: k greatest elements from the input and (k - 1)th value (1st extreme)
    """
    excesses = input_matrix[:, -n_excesses:]
    thresholds = input_matrix[:, -n_excesses]
    return (excesses.T - thresholds).T, thresholds


def estimate_quantiles_frequentist_methods(args, burr_data):
    from src.frequentist_methods import PWM_GPD, MOM_Fisher, MOM_GPD

    # How many values do we want to consider as excesses. The more, the better approximation should we obtain
    n_excesses_ = np.linspace(100, 500, args.n_different_thresholds).astype(int)

    # place holder
    keep_quantiles = {'pwm_gpd': np.zeros((len(args.quantile_levels), n_excesses_)).T,
                      'mom_gpd': np.zeros((len(args.quantile_levels), n_excesses_)).T,
                      'mom_fisher': np.zeros((len(args.quantile_levels), n_excesses_)).T,
                      'mle_gpd': np.zeros((len(args.quantile_levels), n_excesses_)).T}

    for ind, n_excesses in enumerate(n_excesses_):  # for different number of excesses
        # n_excesses = n_excesses_[j]
        shifted_excesses, thresholds = k_greatest_values_matrices(burr_data, n_excesses)

        qnts_pwm_gpd = np.zeros(len(args.quantile_levels))
        qnts_mom_fisher = np.zeros(len(args.quantile_levels))
        qnts_mom_gpd = np.zeros(len(args.quantile_levels))
        qnts_mle_gpd = np.zeros(len(args.quantile_levels))

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


def helper_norm_ss(arr: np.ndarray, ind: int, true_quantile: float):
    assert 0 < true_quantile < 1, 'Quantile level out of range'
    squared_norm_ss = np.sum([pow(x - true_quantile, 2) for x in arr[ind, :]]) / pow(true_quantile, 2)

    return np.sqrt(squared_norm_ss / arr.shape[1])  # arr.shape should be the nb of different thresholds tried?


def main(args):
    # load data
    save_dir = Path.cwd() / 'data/data_burr/'
    file_name = save_dir.__str__() + f'\\beta_{args.burr_beta}__tau_{args.burr_tau}__lambda_{args.burr_lambda}__n_samples_{args.n_different_samples}__n_obs_per_sample_{args.n_points_each_sample}.npy'
    with open(file_name, 'rb') as f:
        burr_data = np.load(f)

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
        MOM_Fisher[ind] = helper_norm_ss(keep_quantiles.get('mom_fisher'), ind, args.quantile_levels[ind])
        MOM_GPD[ind] = helper_norm_ss(keep_quantiles.get('mom_gpd'), ind, args.quantile_levels[ind])
        MLE_GPD[ind] = helper_norm_ss(keep_quantiles.get('mle_gpd'), ind, args.quantile_levels[ind])
        PWM_GPD[ind] = helper_norm_ss(keep_quantiles.get('pwm_gpd'), ind, args.quantile_levels[ind])

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


if __name__ == '__main__':
    beginning = datetime.datetime.now()
    print('Starting')
    args = main_args()
    main(args)
    ending = datetime.datetime.now()

    print('Time: ', ending - beginning)
