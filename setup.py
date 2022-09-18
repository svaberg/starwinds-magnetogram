#!/usr/bin/env python

from distutils.core import setup

exec(open('starwinds_magnetogram/version.py').read())

setup(name='starwinds_magnetogram',
      version=__version__,
      description='Stellar magnetogram visualisation and manipulation',
      author='Dag Evensberget',
      author_email='Dag.Evensberget@usq.edu.au',
      url='https://bitbucket.org/evensber/stellarwinds/',
      packages=['starwinds_magnetogram'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib>3.2',
        #   'pytecplot>=1.1',
        #   'boltons',
        #   'cmocean',
        #   'chiantipy',
        #   'pyastronomy',
        #   'quantities',
        #   'quaternion',
        #   'spherical_functions',
        #   'psutil',
        #   'packaging',
        #   'ipympl',
        #   'uncertainties',
          # Cartopy does not install on HPC
          # 'cartopy',
          ],
      extras_require={'test': [
          # Stuff used for testing _and development_
          'nbstripout',
          'pytest-console-scripts',
          'pytest',
          'nbval',
          'pytest-cov',
          'codacy-coverage',
        #   'pandas',
        #   'xlrd',
      ]},
      entry_points={'console_scripts': [
          'sw-convert-magnetogram=bin.convert_magnetogram:main',
          'sw-plot-lsd=bin.plot_lsd:main',
          'sw-plot-magnetogram=bin.plot_magnetogram:main',
      ], }, )
