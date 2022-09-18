import logging
log = logging.getLogger(__name__)
import numpy as np
import pytest
import glob
from tests import context  # Test context
import stellarwinds.magnetogram.plot_lsd as plsd

# TODO stop referring to files outside project.

@pytest.mark.skip("Missing data file")
@pytest.mark.parametrize("parameters", ("VNI", "VI"))
def test_plot_lsd_profile(request, parameters, skip_header=2):

    lsd_file_name = "/Users/u1092841/Documents/PHD/sw_tools/zdipy/zdipy/LSDProf/lopeg_16aug14_v_02.profD"

    base_data = np.genfromtxt(lsd_file_name, skip_header=skip_header)
    norm_data = np.genfromtxt(lsd_file_name + '.norm', skip_header=skip_header)
    model_data = np.genfromtxt(lsd_file_name + '.norm.model', skip_header=skip_header)

    # Save and plot result
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(len(parameters), figsize=(12, 9), sharex=True)
        plsd.plot_lsd_profile(base_data, parameters=parameters)
        plsd.plot_lsd_profile(norm_data, parameters=parameters)
        plsd.plot_lsd_profile(model_data, parameters=parameters)
        axs[0].set_title('Data in %s' % lsd_file_name)
        axs[-1].legend(('profD', 'profD.norm', 'profD.norm.model'))
        plt.savefig(pn.get())


@pytest.mark.skip("Missing data file")
@pytest.mark.parametrize("range_nm", ((500, 600), (600, 604), None))
def test_plot_spectrum(request, range_nm):

    spectrum_file_name = '/Users/u1092841/Documents/PHD/toupies-data-colin/LOPeg/lopeg_31aug14_v_09.s'

    data = np.genfromtxt(spectrum_file_name, skip_header=2)

    if range_nm is not None:
        ids = (data[:, 0] > range_nm[0]) & (data[:, 0] < range_nm[1])
        data = data[ids, ...]

    alpha = 10000/len(data[:, 0])

    # Save and plot result
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        plsd.plot_spectrum(data, alpha=alpha)
        plt.savefig(pn.get())


@pytest.mark.skip("Missing data file")
def test_plot_lsd_stack(request):

    lsd_files = glob.glob("/Users/u1092841/Documents/PHD/sw_tools/zdipy/zdipy/LSDProf/*.profD.norm")

    data = []
    for lsd_file in lsd_files:
        data.append(np.genfromtxt(lsd_file, skip_header=2))

    # Save and plot result
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        plsd.plot_lsd_stack(data, lsd_files)
        plt.savefig(pn.get())
