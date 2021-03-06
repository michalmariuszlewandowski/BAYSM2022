import numpy
import numpy as np

import datetime
from pathlib import Path
from arg_parser import main_args


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


def estimate_quantiles_frequentist_methods(args, burr_data, n_excesses_):
    from src.frequentist_methods import freq_methods
    # How many values do we want to consider as excesses. The more, the better approximation should we obtain

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
            qnts[1] += frequentist.MOM_GPD()  #
            qnts[2] += frequentist.MLE_GPD()
            qnts[3] += frequentist.MOM_Fisher()

        keep_quantiles.get('pwm_gpd')[ind] = qnts[0] / args.n_different_samples
        keep_quantiles.get('mom_gpd')[ind] = qnts[1] / args.n_different_samples
        keep_quantiles.get('mle_gpd')[ind] = qnts[2] / args.n_different_samples
        keep_quantiles.get('mom_fisher')[ind] = qnts[3] / args.n_different_samples

    return keep_quantiles


def estimate_quantiles_bayesian_methods(args, burr_data, n_excesses_):
    from src.bayesian_estimation.bayes_methods import bayes_methods
    # How many values do we want to consider as excesses

    keep_quantiles = {'pwm_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mom_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mom_fisher': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T,
                      'mle_gpd': np.zeros((len(args.quantile_levels), n_excesses_.shape[0])).T}

    for ind, n_excesses in enumerate(n_excesses_):  # for different number of excesses
        shifted_excesses, thresholds = k_greatest_values_matrices(burr_data, n_excesses)

        qnts = np.zeros((len(keep_quantiles), len(args.quantile_levels)))

        for excesses, threshold in zip(shifted_excesses, thresholds):
            bayes = bayes_methods(data=burr_data, quantile_levels=args.quantile_levels,
                                       excesses=excesses, thresholds=thresholds)
            # fit GPD and Fisher distributions to excesses from each dataset
            qnts[0] += bayes.bayes_GPD()

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


def plot(keep_quantiles: dict):
    """
    todo this should be a class
    :param keep_quantiles:
    :return:
    """
    import matplotlib.pyplot as plt

    col = ['b', 'g', 'r', 'c', 'y']
    symbols = ['*', 'x', '>', '<', '.']
    # fig, ax = plt.subplots()

    def def_marker(key: str) -> str:
        if 'mom_fisher' in key:
            return symbols[0]
        elif 'mom_gpd' in key:
            return symbols[1]
        elif 'mle_gpd' in key:
            return symbols[2]
        elif 'pwm_gpd' in key:
            return symbols[3]

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    fig, ax = plt.subplots()
    for keys, values in keep_quantiles.items():
        if keys is not 'mom_fisher':  # for now omit mom_fisher because the values are very different
            for xe, ye in zip(n_excesses_, values):
                # ax.plot([xe] * len(ye), ye[, def_marker(keys), label=keys)
                for i in range(values.shape[0]):
                    ax.plot([xe] * len([ye[i]]), ye[i], def_marker(keys), label=keys, c=col[i])
    # todo: put on the legend which color corresponds to which quantile level
    #  split keys on '_' for naming and use capital letters
    #  denote on x axis how many quantiles were used

    # f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    #
    # handles = [f("s", colors[i]) for i in range(3)]
    # handles += [f(markers[i], "k") for i in range(3)]

    legend_without_duplicate_labels(ax)

    plt.show()


def main(args, n_excesses_):
    # import matplotlib.pyplot as plt
    # load data
    save_dir = Path.cwd() / 'src/data/data_burr/'
    file_name = save_dir.__str__() + f'\\beta_{args.burr_beta}__tau_{args.burr_tau}__lambda_{args.burr_lambda}__n_samples_{args.n_different_samples}__n_obs_per_sample_{args.n_points_each_sample}.npy'

    with open(file_name, 'rb') as f:
        burr_data = np.load(f)

    # take a random sample for plotting and computing first two moments
    ind = int(np.random.uniform(burr_data.shape[0], size=1))
    # plt.hist(burr_data[ind], bins='auto')
    # plt.title(f'Empirical mean {round(np.mean(burr_data[ind]),3)}, '
    #           f'var {round(np.var(burr_data[ind]), 3)}')
    # plt.show()

    # keep_quantiles = estimate_quantiles_frequentist_methods(args, burr_data, n_excesses_)
    keep_quantiles = estimate_quantiles_bayesian_methods(args, burr_data, n_excesses_)
    # todo here get the bayesian quantile estimates

    # qnts_pwm_gpd, qnts_mom_gpd, qnts_mom_fisher, qnts_mle_gpd
    # columns: different nb of excesses used to estimate quantiles
    # rows: quantiles of different levels
    plot(keep_quantiles)
    print('Done')


if __name__ == '__main__':
    beginning = datetime.datetime.now()
    print('Starting')
    args = main_args()
    n_excesses_ = np.linspace(args.min_nb_obs_considered_excesses,
                              args.max_nb_obs_considered_excesses,
                              args.n_different_excesses).astype(int)

    main(args, n_excesses_)
    ending = datetime.datetime.now()

    print('Time: ', ending - beginning)
