from typing import Optional
from functools import reduce
import matplotlib.pyplot as plt
from datetime import timedelta
from pydantic import BaseModel
import numpy as np
from adase_api.metrics import pearsonr_ci


class Experiment(BaseModel):
    ticker: Optional[str] = 'EURUSD=X'
    topic: Optional[str] = 'energy crisis in Europe'
    engine: Optional[str] = 'topic'
    feature: Optional[str] = 'coverage'
    y: Optional[str] = 'pct_change'
    fwd_steps: Optional[int] = 7
    roll_days: Optional[int] = 7
    ada_roll_days: Optional[int] = 7
    freq: Optional[str] = '24h'
    anomaly_bbands: Optional[float] = 1.5
    target_data: Optional[list]
    ada_data: Optional[list]


def split_consec(df, col='coverage'):
    df['g'] = np.append([0], ((df.index[:-1] - df.index[1:]).days != -1).cumsum())
    return df.groupby('g').apply(lambda g: [g.index.min(), g.index.max(), g[col][:1].mean(), g[col][1:].mean()])


def plot_regimes(data, roll_corr, anomalies, q: Experiment, figsize=(12, 5)):
    """
    Plot regime changes: good news --> positive (negative) correlations
    :param roll_corr:
    :param anomalies:
    :param data: pd.DataFrame, ADA & price
    :param q: Experiment, query
    :param figsize: tuple, size of figure
    :return:
    """
    def get_max_offset(causality):
        offset = causality.abs().idxmax()
        return offset

    def get_pearsonr_ci(df, feature, y, max_offset, step):
        _df = df.copy()
        _df['feature_shifted'] = _df[feature].shift(max_offset * step)
        _df = _df.dropna()
        corr_ci = pearsonr_ci(_df.feature_shifted, _df[y], ci=90)
        return min(corr_ci, key=abs)

    fig, ax1 = plt.subplots(figsize=figsize)
    plt.grid()
    ax2 = ax1.twinx()
    ax1.plot(data[q.feature], 'y-', label=q.feature, color='#390175')
    ax1.legend(loc=0, fontsize=14)
    ax2.plot(data[q.y], 'b-', label=q.y, color='#545454')
    ax2.legend(loc=4, fontsize=14)
    ax1.set_ylabel(f"`{q.topic}`", color='#390175')
    ax2.set_ylabel(q.ticker, color='#545454')
    ax2.set_title(f"ADA vs.{q.ticker}\n")

    n = roll_corr[roll_corr[q.feature] < 0].copy()
    p = roll_corr[roll_corr[q.feature] > 0].copy()
    n_consec, p_consec = split_consec(n, col=q.feature), split_consec(p, col=q.feature)

    lcorr, corr_sub = [], []
    for colour, direction, anomaly_ranges in zip(['g', 'r'], [1, -1], [p_consec, n_consec]):
        for _, (start, end, corr, _) in anomaly_ranges.items():
            sub = roll_corr[(roll_corr.index >= start) & (roll_corr.index <= end)]
            if len(sub) == 0:
                print("skip", start, end)
                continue
            sub_data = data[(data.index >= start) & (data.index <= end + timedelta(hours=17))]
            corr_lstd = round(sub[q.feature].median(), 2)
            corr_sub_data = sub_data.corr().loc[q.feature, q.y]
            corr_sub += [[corr_sub_data, direction, len(sub), corr_lstd, start, end]]

            for dt in pd.date_range(sub.index.min(), sub.index.max(), freq='1d'):
                sub2 = roll_corr[roll_corr.index == dt]
                avg_corr = sub2[q.feature].mean()
                if avg_corr != avg_corr:
                    continue
                plt.axvspan(dt, dt + timedelta(days=1), facecolor=colour,
                            alpha=abs(avg_corr) / 1.5 + 0.05)
            #         plt.axvspan(sub.index.min(), sub.index.max(), facecolor='g', alpha=abs(corr)/1.2+0.05)

            #         t = plt.text(sub.index.min(), data[y].max(), corr_lstd, fontsize=12)
            #         t.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='orange'))

            lcorr += [[corr_lstd * direction, len(sub)]]

    for en, (feature, res) in enumerate(anomalies.items()):
        t = plt.text(data.index.min() - timedelta(days=28), data[q.y].max() * (1 + en / 100), feature, fontsize=12)
        t.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='orange'))

        for direction, ranges in res.items():
            if direction == 1:
                color, y_pos = 'b', data[q.y].max() * (1 + en / 100)
            elif direction == -1:
                color, y_pos = 'r', data[q.y].min() * (1 + en / 100)
            else:
                continue
            for _, (start, end, corr, _) in ranges.items():
                ax2.plot((start, end), (y_pos, y_pos), color=color, linewidth=5)

    plt.show()

    unique_offsets = {(i // (q.roll_days//3)) for i in range(1, 36, 2)}
    causality = reduce(lambda l, r: l.join(r), [
        data[[q.y]]] + [data[q.feature].shift(i).to_frame(i) for i in unique_offsets]).corr()[q.y].drop(q.y)

    max_offset = get_max_offset(causality)
    corr_ci = round(get_pearsonr_ci(data, q.feature, q.y, max_offset, step=q.roll_days//3), 3)
    r = f"offset={max_offset}h, corr={corr_ci}\n"

    #     ax3 = causality.plot.bar(figsize=figsize, title=r)
    #     ax3.set_ylabel("Correlation")
    #     ax3.set_xlabel("\nlead hrs")
    #     plt.grid()
    #     plt.show()
    return r, causality, lcorr, corr_sub