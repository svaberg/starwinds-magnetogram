import logging
log = logging.getLogger(__name__)
import pytest

from tests import context  # Test context


def test_magnetogram_scripts(script_runner):
    ret = script_runner.run('sw-plot-magnetogram')
    assert ret.returncode != 0

    ret = script_runner.run('sw-convert-magnetogram')
    assert ret.returncode != 0


@pytest.mark.parametrize("options",
                         ("-p map",
                          "-p spectrum",
                          "-p matrix",
                          "-p energy",
                          "-y pfss",
                          "-p polar -v",
                          "-p strength -v",
                          "-p polar -v",
                          "-p azimuthal -v",
                          "-p azimuth -v",
                          "-p radial -v",))
def test_sw_plot_magnetogram(script_runner, options, request, zdi_file):
    pn = context.PlotNamer(__file__, request.node.name)
    ret = script_runner.run('sw-plot-magnetogram', zdi_file, pn.get().replace(' ', ''), *options.split(" "))
    assert ret.returncode == 0, "Expected return code 0 (success)."


@pytest.mark.parametrize("inverse", (None, "--inverse"))
@pytest.mark.parametrize("format_only", (None, "--format-only"))
@pytest.mark.parametrize("degree_max", (None, "--degree-max=3", "--degree-max=60"))
def test_sw_convert_magnetogram(script_runner, request, inverse, format_only, degree_max, zdi_file):
    pn = context.PlotNamer(__file__, request.node.name)

    args = [inverse, format_only, degree_max]
    args = [a for a in args if a is not None]

    ret = script_runner.run('sw-convert-magnetogram',
                            zdi_file,
                            pn.get(".dat").replace(' ', ''),
                            "--radial-only",
                            *args)
    assert ret.returncode == 0, "Expected return code 0 (success)."



