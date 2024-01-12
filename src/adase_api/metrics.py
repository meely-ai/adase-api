from functools import reduce
import numpy as np
from scipy.stats import norm


def get_causality(df, target, feature, start=-28, stop=60, step=2, freq='d'):
    """
    Measure lead/lag between feature and target
    :param df: pd.DataFrame, with datetime index
    :param target: str, target
    :param feature: str, feature
    :param start: int, from time offset
    :param stop: int, until time offset
    :param step: int, step
    :param freq: str, frequency 'h', 'd' etc.
    :return: pd.DataFrame correlation with range of time offsets
    """
    return reduce(lambda l, r: l.join(r), [
        df[[target]]] + [df[feature].shift(i, freq=freq).to_frame(i // step) for i in range(start, stop, step)
                         ]).corr()[target].drop(target)


def pearsonr_ci(x, y, ci=80, n_boots=10000):
    x = np.asarray(x)
    y = np.asarray(y)

    # (n_boots, n_observations) paired arrays
    rand_ixs = np.random.randint(0, x.shape[0], size=(n_boots, x.shape[0]))
    x_boots = x[rand_ixs]
    y_boots = y[rand_ixs]

    # differences from mean
    x_mdiffs = x_boots - x_boots.mean(axis=1)[:, None]
    y_mdiffs = y_boots - y_boots.mean(axis=1)[:, None]

    # sums of squares
    x_ss = np.einsum('ij, ij -> i', x_mdiffs, x_mdiffs)
    y_ss = np.einsum('ij, ij -> i', y_mdiffs, y_mdiffs)

    # pearson correlations
    r_boots = np.einsum('ij, ij -> i', x_mdiffs, y_mdiffs) / np.sqrt(x_ss * y_ss)

    # upper and lower bounds for confidence interval
    ci_low = np.percentile(r_boots, (100 - ci) / 2)
    ci_high = np.percentile(r_boots, (ci + 100) / 2)
    return ci_low, ci_high


def get_pearsonr_ci_shift(df, feature, y, max_offset, step):
    _df = df.copy()
    _df['feature_shifted'] = _df[feature].shift(max_offset * step)
    _df = _df.dropna()
    corr_ci = pearsonr_ci(_df.feature_shifted, _df[y], ci=90)
    return min(corr_ci, key=abs)


def z_test(mu1, mu2, sigma, sample_n):
    standard_error = sigma / sample_n ** (1/2)
    z_score = (mu1 - mu2) / standard_error
    return z_score, norm.cdf(z_score)


def kelly_criterion(e_return, e_volatility):
    """
    Optimal position size allocation
    Example `kelly_criterion(0.55, 0.017)`
    https://en.wikipedia.org/wiki/Kelly_criterion
    :param e_return: float, expected return
    :param e_volatility: float, expected volatility
    :return:
    """
    e_loss = 1 - e_return
    return e_return / e_volatility - e_loss / e_volatility

