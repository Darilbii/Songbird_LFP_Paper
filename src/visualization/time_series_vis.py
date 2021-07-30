import numpy as np
# from scipy import signal
# from neurodsp import filt
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# def plot_pretty_ersp(ersp, event_times):
#
#     # TODO: Make it such that you can control the filters if you want to
#     fc_lo = np.arange(3, 249, 2)
#     fc_hi = np.arange(5, 251, 2)
#
#     cmap = 'RdBu_r'
#     ax = sns.heatmap(np.mean(ersp, 1), xticklabels=event_times, yticklabels=(fc_lo + fc_hi) / 2, cmap=cmap)
#     ax.invert_yaxis()
#     for ind, label in enumerate(ax.get_xticklabels()):
#         if ind % 100 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#     for ind, label in enumerate(ax.get_yticklabels()):
#         if ind % 10 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#     plt.show()


def plot_pretty_ersp(ersp, event_times, cmap=None, **kwargs):
    """

    :param cmap:
    :param ersp:
    :param event_times:
    :param kwargs: Check the Seaborn Options (Lots of control here)
    :return:
    """

    # TODO: Make it such that you can control the filters if you want to
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    if cmap is None:
        cmap = 'RdBu_r'

    ax = sns.heatmap(ersp, xticklabels=event_times, yticklabels=(fc_lo + fc_hi) / 2, cmap=cmap, **kwargs)

    ax.invert_yaxis()

    visible_xticks = []
    visible_xtickslabels = []
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 100 == 0:
            visible_xticks.append(ind)
            visible_xtickslabels.append(label)
    ax.set_xticks(visible_xticks)
    ax.set_xticklabels(visible_xtickslabels)

    for ind, label in enumerate(ax.get_yticklabels()):
        if ind % 10 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    if ax is None:
        plt.show()

# TODO: Change plot_behavior function to accept a properly formated cmap


def plot_behavior(fill_events_context, context_event_times, context_events, show_x=False, ax=None):
    # Setup the Colorbar
    cmap2 = matplotlib.colors.ListedColormap(
        ['black', 'red', 'orange', 'yellow', 'saddlebrown', 'blue', 'green', 'white', 'pink', 'purple'])
    cmap2.set_over('cyan')
    cmap2.set_under('cyan')
    bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

    # PlotBehavior Raster
    num_events = context_events.shape[0]
    max_len = fill_events_context.shape[0]
    bin_width = max_len / num_events
    y_labels = np.arange(0, num_events, 5, dtype=int)
    y_steps = np.linspace(0, y_labels[-1] * bin_width, len(y_labels), dtype=int)
    y_steps[1:] = y_steps[1:] - int(bin_width / 2)

    if ax is None:
        plt.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        plt.yticks(ticks=y_steps[1:], labels=y_labels[1:])
        plt.ylim(0, max_len)
        ax = plt.gca()  # Get the Current Axis

    else:
        ax.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        ax.set_yticks(y_steps[1:])
        ax.set_yticklabels(y_labels[1:])
        ax.set_ylim(0, max_len)

    if show_x:
        ax.set_xticks(np.arange(len(context_event_times)))
        ax.set_xticklabels(context_event_times)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        visible_xticks = []
        visible_xtickslabels = []
        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % 100 == 0:
                visible_xticks.append(ind)
                visible_xtickslabels.append(label)
        ax.set_xticks(visible_xticks)
        ax.set_xticklabels(visible_xtickslabels)
    else:
        ax.set_xticks([])
