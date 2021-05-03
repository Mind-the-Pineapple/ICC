#! /usr/bin/env python

import os
import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()

# get __version__ from _version.py
ver_file = os.path.join('ICC', '_version.py')
with open(ver_file) as f:
    exec(f.read())

# This call to setup() does all the work
setup(
    name='icc',
    version=__version__,
    description='A Python implementation to calculate the ICC',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/Mind-the-Pineapple/ICC',
    author='Walter Hugo Lopez Pinaya, Jessica Dafflon',
    author_email='walter.diaz_sanz@kcl.ac.uk, jessica.dafflon@kcl.ac.uk',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: Unix',
    ],
    packages=['icc'],
    include_package_data=True,
    install_requires=['scipy', 'numpy'],
)
