#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements_list = f.read().split("\n")
    requirements_list = [x for x in requirements_list if x]

setup(name='ml-experiments',
      version='0.0.1',
      description='My ML Experiments',
      author='Andrei Marin',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=requirements_list,
      python_requires='>=3.6',
      include_package_data=True)
