# The starwinds-magnetogram python package
![Python package](https://github.com/svaberg/starwinds-magnetogram/actions/workflows/python-package.yml/badge.svg)

This package contains basic processing tools for manipulating and visualising Solar and stellar magnetograms. These tools were originally developed during my Ph.D. at the University of Southern Queensland. Please cite 
[Evensberget et al. 2020](https://doi.org/10.1093/mnras/stab1696)
or 
[Evensberget et al. 2021](https://doi.org/10.1093/mnras/stab3557) 
and mention the `starwinds-magnetogram` package if you use these tools in your research. 

## Features
* Convert magnetogram coefficients from ZDI format to Stanford PFSS format suitable for ingestion into the SWMF;
* Plotting of ZDI and PFSS magnetograms;
* Magnetogram generation from e.g. noise and manipulation;
* Free rotation of a magnetogram;
* Some routines related to the Parker wind solution and Alfvén surface.

## Installation
Once the repository is cloned, install using
```bash
pip install --user --editable .
```
the `--user` flag is required when the user cannot install in the root such as on an HPC. The flag will will put entry points in `~/.local/bin`, which must then be on the `$PATH`.

The `--editable` flag is only required for development. 

## Getting started
  * [Magnetogram conversion shell commands](docs/Shell-commands.ipynb)
  * [Magnetogram manipulation notebook](docs/Magnetogram-manipulation.ipynb)

## Development info
This section may be ignored unless you want to make changes to `starwinds-magnetogram`.
### Test summary
To run the tests locally and generate a test summary type
```bash
cd tests
pytest -v --tb=no
```
there should be no failures in the code and tests. Test output is placed in the `tests/artifacts` folder.
The ` --disable-warnings` flag can be used to hide any warnings (should be unnecessary).

### Debugging
To debug through the entry points (e.g. the `sw-plot-magnetogram` command), this should work:
```bash
python -m pdb $(\which sw-plot-magnetogram)
```
The backslash character before `which` is present since the `which` command may be aliased to `type -all` on MacOS. 
