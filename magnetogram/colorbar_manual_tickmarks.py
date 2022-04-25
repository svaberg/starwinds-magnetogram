import matplotlib.pyplot as plt
import numpy as np


def auto_symlog_tickmarks(cb, major_tick_step=1, axis="x"):
    """
    Set both minor and major tick marks on a symlog scale in a pretty way.
    :param cb: Matplotlib colorbar object
    :param major_tick_step: Step between powers (default=1)
    :return:
    """

    norm = cb.norm

    cb.ax.minorticks_on()

    major_tick_min = int(np.floor(np.log10(norm.linthresh)))
    major_tick_max = int(np.floor(np.log10(norm.vmax)))

    major_values, major_labels = generate_major(
        major_tick_min,
        major_tick_max,
        major_tick_step)
    minor_values = generate_minor(
        major_tick_min,
        major_tick_max,
        major_tick_step)

    target_axis = None
    if axis == "x":
        target_axis = cb.ax.xaxis
    elif axis == "y":
        target_axis = cb.ax.yaxis

    fully_manual_tickmarks(target_axis,
                           norm,
                           major_values, major_labels,
                           minor_values)


def fully_manual_tickmarks(axis, norm, major_values, major_labels, minor_values):
    """
    Set tickmarks from lists.
    :param axis:
    :param norm:
    :param major_values:
    :param major_labels:
    :param minor_values:
    :return:
    """
    ax = axis.axes
    xlim = ax.get_xlim()
    xmin = xlim[0]
    xrange = xlim[1] - xlim[0]

    xticks_major = xrange * norm(major_values) + xmin
    axis.set_ticks(xticks_major)
    axis.set_ticklabels(major_labels)

    xticks_minor = xrange * norm(minor_values) + xmin

    xticks_minor = xticks_minor[xticks_minor >= xlim[0]]
    xticks_minor = xticks_minor[xticks_minor <= xlim[1]]

    axis.set_ticks(xticks_minor, minor=True)


def generate_major(log10_min, log10_max, log10_step):
    """
    Generate list of major tick marks
    :param log10_min: smallest exponent
    :param log10_max: largest exponent
    :param log10_step: exponent step
    :return:
    """
    exponents = np.arange(log10_min, log10_max + 1, log10_step)

    major_values = list(-10.0 ** exponents)[::-1] + [0] + list(10.0 ** exponents)

    major_labels = \
        list("$-10^{%d}$" % e for e in exponents[::-1]) \
        + ["0"] \
        + list("$10^{%d}$" % e for e in exponents)

    return major_values, major_labels


def generate_minor(log10_min, log10_max, log10_step):
    decimals = np.arange(0, 10)

    all_positive = []
    for exp in range(log10_min, log10_max + 2):
        all_positive.extend(decimals * 10 ** (exp - 1))
        all_positive.extend(-decimals * 10 ** (exp - 1))

    all_values = np.unique(np.sort(all_positive))

    return all_values


def tesla_to_gauss(tickmark_string, exponent_shift=4):
    """
    Convert value in Tesla to Gauss in the colorbar tickmarks strings
    :param tickmark_string: Tickmark string
    :param exponent_shift: Exponent shift (4 converts Tesla to Gauss)
    :return:
    """
    position_0 = tickmark_string.find("{") + 1
    position_1 = tickmark_string.find("}")
    if position_1 < position_0:
        return tickmark_string

    string_left = tickmark_string[:position_0]
    string_middle = tickmark_string[position_0:position_1]
    string_right = tickmark_string[position_1:]

    return string_left + str(int(string_middle) + exponent_shift) + string_right

