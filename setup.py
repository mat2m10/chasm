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
      description= "Simulation toolkit for genotype-phenotype data under population structure, for testing GWAS stratification correction (PCs vs. allele frequencies)",
      install_requires = requirements,
      packages=find_packages(),
      url="https://github.com/mat2m10/chasm",
      author= "Matthieu de Hemptinne",
      author_email= "m.c.de.hemptinne@vu.nl",
      )