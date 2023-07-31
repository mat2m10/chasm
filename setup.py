"""
Import setuptools
"""
from setuptools import setup
from setuptools import find_packages
import os

"""
Reading the requirements
"""
with open('requirements.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

"""
Setup of the project
"""
setup(name = 'chasm',
      description= "a deep learning tool that computes a polygenic risk score and trys to find interaction between snps",
      install_requires = requirements,
      packages=find_packages(),
      author= "Matthieu de Hemptinne",
      author_email= "m.c.de.hemptinne@vu.nl",
      )
