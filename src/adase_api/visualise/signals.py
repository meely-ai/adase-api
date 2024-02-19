from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from adase_api.metrics import get_causality, get_pearsonr_ci_shift


def fmt_label(v):
    if isinstance(v, (tuple, list)):
        return ' — '.join(v)
    return v


def draw_double_axis_plot(df, target, feature_r, feature_l=None, title='', xlabel='', path_to_save='', figsize=(12, 5)):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlabel(xlabel)
    plt.grid()
    ax2 = ax1.twinx()
    if feature_l:
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.07))

    label_t, label_r, label_l = [fmt_label(v) for v in [target, feature_r, feature_l]]

    ax1.plot(df[feature_r], label=label_r, color='r')
    ax1.set_ylabel(label_r, color='r')
    ax1.tick_params(axis='y', colors='r', labelsize=12)

    plt.title(title, fontsize=14)
    ax2.plot(df[target], label=label_t, color='#390175')
    # ax2.legend(loc=0, fontsize=12)
    ax2.set_ylabel(label_t, color='#390175')
    ax2.tick_params(axis='y', labelsize=12, colors='#390175')
    if feature_l:
        ax3.plot(df[feature_l], label=label_l, color='b')
        ax3.set_ylabel(label_l, color='b')
        ax3.tick_params(axis='y', colors='b', labelsize=12)

    if path_to_save != '':
        plt.savefig(path_to_save, dpi=200, bbox_inches='tight')


def plot_twin_time_offset(target, feature, df, start=-12, stop=18, step=2, freq='d',
                          xlabel='month shift', title='', figsize=(12, 5)):
    figure, axis = plt.subplots(2, 1, figsize=figsize)
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=.5, wspace=0.2)

    axis[0].grid()
    ax2 = axis[0].twinx()

    axis[0].plot(df.iloc[:, 0], label=fmt_label(df.columns[0]), color='r')
    axis[0].set_ylabel(fmt_label(df.columns[0]), color='r')
    axis[0].tick_params(axis='y', colors='r', labelsize=12)

    ax2.plot(df.iloc[:, 1], label=fmt_label(df.columns[1]), color='b')
    ax2.set_ylabel(fmt_label(df.columns[1]), color='b')
    ax2.tick_params(axis='y', labelsize=12, colors='b')

    causality = get_causality(df, target, feature, start=start, stop=stop, step=step, freq=freq)
    causality.plot.bar(ax=axis[1])
    axis[1].set_title(f"{fmt_label(feature)}\nvs.\n{fmt_label(target)}\n")
    axis[1].set_ylabel('correlation')
    axis[1].set_ylim(causality.min(), causality.max() * 1.1)
    axis[1].set_xlabel(xlabel)
    axis[1].grid()
    plt.title(title, fontsize=14)

    for a in figure.axes:
        a.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,
            top=False,
            labelbottom=True)  # labels along the bottom edge are on

    return causality


def compare_pair(df, target=None, feature=None, start=-12, stop=18, step=2, freq='d',
                 xlabel='month shift', title='', figsize=(12, 5)):
    mrg = pd.merge_asof(df[target], df[feature], on=['date_time']).set_index("date_time")
    causality = plot_twin_time_offset(target, feature, mrg, start=start, stop=stop, step=step, freq=freq,
                                      xlabel=xlabel, title=title, figsize=figsize)
    max_offset = causality.abs().idxmax()
    pearsonr_ci = get_pearsonr_ci_shift(mrg, feature, target, max_offset, 1)
    return feature, target, max_offset, pearsonr_ci


def plot_double_time_shifted(df, label_l, label_r, days_offset=742, figsize=(12, 5)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()

    first = df.copy()
    first.index -= timedelta(days=days_offset)
    first[label_l] = first[label_l].shift(days_offset, freq='d')

    second = df.copy()
    second.loc[second.index[-1], label_r] = 0

    ax2.plot(first.index, first[label_l], label=label_l, color='black')
    ax1.plot(second.index, second[label_r], label=label_r, color='#370182')
    ax1.tick_params(axis='y', colors='black', labelsize=12)

    ax1.set_ylabel(label_l, color='black')
    ax3.set_ylabel(label_r, color='#370182')
    ax2.tick_params(axis='x', colors='black', labelsize=12)
    ax1.tick_params(axis='x', colors='#370182', labelsize=12)
    ax3.tick_params(axis='y', colors='#370182', labelsize=12)

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.17))
    ax1.grid()
    ax1.set_xlabel("Date", fontsize=12, color='#370182')
    ax2.set_xlabel("Shifted", fontsize=12, color='black')
    plt.title("\nADA Sentiment tracker\nz-scores\n")

    plt.show()