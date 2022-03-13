import argparse


def main_args():
    parser = argparse.ArgumentParser(description='Parsing Arguments.')

    # general params
    parser.add_argument('--n_points_each_sample', default=1000, type=int, help='')
    parser.add_argument('--n_excesses', default=100, type=int)
    parser.add_argument('--n_different_thresholds', default=15, type=int, help='for how many thresholds do we test')
    parser.add_argument('--n_different_samples', default=100,
                        type=int, help='how many time we average over the result')
    # burr params
    parser.add_argument('--burr_beta', default=.25, type=float)
    parser.add_argument('--burr_tau', default=1., type=float)
    parser.add_argument('--burr_lambda', default=4., type=float)

    parser.add_argument('--quantile_levels',
                        default=[0.98, 0.99, 0.995, 0.999, 0.9995],
                        nargs='+', type=float,
                        help='Extreme quantiles we check')

    return parser.parse_args()
