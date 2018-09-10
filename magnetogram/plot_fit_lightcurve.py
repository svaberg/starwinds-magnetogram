import numpy as np
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def plot_fit_graphics(data_x, data_y, data_y_std, fitted_x, fitted_y, errors, fit_name='Fit',
                      true_x=None, true_y=None, true_name='True'):

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax3 = ax2.twiny()
    fig.subplots_adjust(hspace=0)

    ax1.errorbar(data_x, data_y, yerr=data_y_std, label='Noisy data',
                 fmt='o',
                 markersize=1,
                 # color=line.get_color(),
                 elinewidth=0.5)

    if fitted_x is not None:
        ax1.plot(fitted_x, fitted_y, label='Fitted %s' % fit_name)

    if true_x is not None:
        ax1.plot(true_x, true_y, 'k', label='True %s' % true_name)

    ax1.grid(True)
    ax1.set_ylabel('Signal')
    ax1.set_xlabel('Velocity')
    ax1.xaxis.set_ticks_position("top")
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.set_label_position("right")
    ax1.set_zorder(1000)
    ax1.autoscale(enable=True, axis='x', tight=True)
    # ax1.spines['bottom'].set_visible(False)
    ax1.patch.set_alpha(0.0)
    ax1.legend()

    # range = np.max(np.abs(errors))
    # bins = np.linspace(-range, range, len(errors)//50)
    # bins = np.round(50 * bins) / 50
    # n, bins,_ = ax2.hist(errors, bins=bins, orientation='horizontal')
    # centers = 0.5 * (bins[1:] + bins[:-1])
    # centers = bins
    # ax2.set_yticks(centers)

    if errors is not None:
        ax2.hist(errors, orientation='horizontal')

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.yaxis.set_ticks_position("right")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Residual")
    #ax2.grid(True, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='x', top=False)

    ax3.plot(data_x, np.zeros_like(data_x), 'k-')

    if errors is not None:
        ax3.plot(data_x, errors, linewidth=0.5, color='gray')
        ax3.plot(data_x, errors, 'ko', label='Residuals', markersize=1.0)

    # ax3.xaxis.set_ticks_position("top")
    # ax3.xaxis.set_label_position("top")
    # ax3.axes.get_xaxis().set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.grid(True, axis='x')
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(axis='x', top=False)
