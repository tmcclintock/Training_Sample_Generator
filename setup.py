import sys, os, glob
from setuptools import setup, Extension
import subprocess

dist = setup(name="sample_generator",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Tool for generating sampling points for emulators.",
             license="GNU General Public License v2.0",
             url="https://github.com/tmcclintock/Training_Sample_Generator",
             packages=['sample_generator'],
             install_requires=['numpy'],
             tests_require=['pytest'])
