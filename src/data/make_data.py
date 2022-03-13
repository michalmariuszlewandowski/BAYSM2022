"""
Samples data from the Burr distribution, using the following
CDF_{Burr}(x) = 1 - (beta / (beta + x^tau))^lambda
"""

import numpy as np

from pathlib import Path
from scipy.stats import rv_continuous
from arg_parser import main_args


class burr_gen(rv_continuous):
    """
    simulate iid burr observations
    """

    def _cdf(self, x, beta, tau, Lambda):
        return 1 - pow(beta / (beta + pow(x, tau)), Lambda)


def create_burr_data(n_different_samples: int, n_points_each_sample: int, burr_params: dict) -> np.ndarray:
    burr = burr_gen(a=0.0, name='burr')  # specify support [a,b], no b means b = infinity

    burr_observations_holder = np.zeros((n_different_samples, n_points_each_sample))

    for i in range(n_different_samples):
        rv_burr = burr.rvs(
            burr_params.get('burr_beta'), burr_params.get('burr_tau'), burr_params.get('burr_lambda'),
            size=n_points_each_sample
        )
        burr_observations_holder[i] = np.sort(rv_burr)

    return burr_observations_holder


if __name__ == '__main__':
    args = main_args()
    burr_observations_holder = create_burr_data(
        args.n_different_samples, args.n_points_each_sample,
        {
            'burr_beta': args.burr_beta, 'burr_tau': args.burr_tau, 'burr_lambda': args.burr_lambda
        }
    )
    save_dir = Path.cwd() / 'data/data_burr/'
    save_dir.mkdir(exist_ok=True, parents=True)
    file_name = save_dir.__str__() + f'\\beta_{args.burr_beta}__tau_{args.burr_tau}__lambda_{args.burr_lambda}__n_samples_{args.n_different_samples}__n_obs_per_sample_{args.n_points_each_sample}.npy'
    with open(file_name, 'wb') as f:
        np.save(f, burr_observations_holder)

    print(f'Saved as {file_name}')
