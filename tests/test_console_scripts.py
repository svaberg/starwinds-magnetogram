import logging
log = logging.getLogger(__name__)
import pytest

from tests import context  # Test context


def test_magnetogram_scripts(script_runner):
    ret = script_runner.run('sw-plot-magnetogram')
    assert ret.returncode != 0

    ret = script_runner.run('sw-convert-magnetogram')
    assert ret.returncode != 0


@pytest.mark.parametrize("y_opts", ("", "-y pfss"))
@pytest.mark.parametrize("p_opts",
                         ("",
                          "-p map",
                          "-p energy-summary",
                          "-p energy-matrix",
                          "-p energy-by-degree",
                          "-p polar",
                          "-p strength",
                          "-p polar",
                          "-p azimuthal",
                          "-p azimuth",
                          "-p radial",))
def test_sw_plot_magnetogram(script_runner, y_opts, p_opts, request, zdi_file, pfss_file):
    if y_opts:
        file = pfss_file
    else:
        file = zdi_file

    pn = context.PlotNamer(__file__, request.node.name)

    cmd = f"sw-plot-magnetogram {file} {pn.get().replace(' ', '')} {y_opts} {p_opts}"

    ret = script_runner.run(*cmd.split())
    assert ret.returncode == 0, "Expected return code 0 (success)."


@pytest.mark.parametrize("pfss_to_zdi", (None, "--pfss-to-zdi"))
@pytest.mark.parametrize("degree_max", (None, "--degree-max=3", "--degree-max=60"))
def test_sw_convert_magnetogram(script_runner, request, pfss_to_zdi, degree_max, zdi_file, pfss_file):

    if pfss_to_zdi:
        file = pfss_file
    else:
        file = zdi_file

    pn = context.PlotNamer(__file__, request.node.name)

    args = ['sw-convert-magnetogram',
            file,
            pn.get(".dat").replace(' ', ''),
            pfss_to_zdi,
            degree_max]

    ret = script_runner.run(*[a for a in args if a is not None])
    assert ret.returncode == 0, "Expected return code 0 (success)."



