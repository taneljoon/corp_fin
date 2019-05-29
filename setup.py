# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

#with open('LICENSE') as f:
    license = f.read()

setup(
    name='corp_fin',
    version='0.1.0',
    description='Modelling corporate and project finances',
    long_description=readme,
    author='Tanel Joon',
    author_email='tanel.joon@energia.ee',
    url='https://github.com/taneljoon/corp_fin',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
