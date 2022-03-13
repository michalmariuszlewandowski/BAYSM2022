"""
loads and plots estimated and saved somewhere else quantiles
"""
from matplotlib import pyplot as plt

from main import compute_true_quantiles, args


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