# The starwinds-magnetogram python package
![Python package](https://github.com/svaberg/starwinds-magnetogram/actions/workflows/python-package.yml/badge.svg)

This package contains basic processing tools for manipulating and visualising Solar and stellar magnetograms.

## Features
* Convert magnetogram coefficients from ZDI format to Stanford PFSS format suitable for ingestion into the SWMF.
* Plotting of ZDI and PFSS magnetograms
* Magnetogram generation from e.g. noise and manipulation
* Rotation of magnetogram in polar and azimuthal directions

## Installation
Clone the repository with git. If this does not work get it with `curl`.
Once the repository is cloned, install using
```bash
cd starwinds_magnetogram
pip install --user --editable .
```
the `--user` flag is required when the user cannot install in the root. The flag will will put entry points in `~/.local/bin`, which must then be on the `$PATH`.

The `--editable` flag is only required for development. 

### Getting started
  * [Magnetogram conversion shell commands](docs/Shell-commands.ipynb)
  * [Magnetogram manipulation notebook](docs/Magnetogram-manipulation.ipynb)

## Test summary
Generate a test summary 
```bash
pytest -v --tb=no
```
there should be no failures in the code and tests. 
The ` --disable-warnings` flag can be used to hide any warnings (should be unnecessary).

## Debugging
To debug through the entry points (e.g. the `sw-plot-magnetogram` command), this should work:
```bash
python -m pdb $(\which sw-plot-magnetogram)
```
The backslash character before `which` is present since the `which` command may be aliased to `type -all` on MacOS. 
