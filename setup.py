#!/usr/bin/env python

from distutils.core import setup

exec(open('starwinds_magnetogram/version.py').read())

setup(name='starwinds_magnetogram',
      version=__version__,  # noqa: F821
      description='Stellar magnetogram visualisation and manipulation',
      author='Dag Evensberget',
      author_email='Dag.Evensberget@usq.edu.au',
      url='https://github.com/svaberg/starwinds-magnetogram/',
      packages=['starwinds_magnetogram'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib>3.2',
          'quaternion',
          'spherical_functions',
          ],
      extras_require={'test': [
          # Stuff used for testing and development
          'nbstripout',
          'pytest-console-scripts',
          'pytest',
          'nbval',
          'psutil',
          'pytest-cov',
          'codacy-coverage',
      ]},
      entry_points={'console_scripts': [
          'sw-convert-magnetogram=bin.convert_magnetogram:main',
          'sw-plot-lsd=bin.plot_lsd:main',
          'sw-plot-magnetogram=bin.plot_magnetogram:main',
      ], }, )
