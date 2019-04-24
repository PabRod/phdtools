# -*- coding: utf-8 -*-

# Learn more: https://github.com/PabRod/phdtools

from setuptools import setup, find_packages


with open('readme.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='phdtools',
    version='0.1.0',
    description='Toy package',
    long_description=readme,
    author='Pablo Rodríguez-Sánchez',
    author_email='pablo.rodriguez.sanchez@gmail.com',
    url='https://github.com/PabRod/phdtools',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'vignettes'))
)
