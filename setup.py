import os
import sys
from setuptools import setup, find_packages

APP_NAME = 'ml-eval'
VERSION = '0.0.1'


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


install_requires = [
    'pyyaml',
    'scipy',
    'sklearn',
    'pytablewriter'
]

setup_info = dict(
    name='metric-learning-evaluator',
    author='Viscovery',
    version=VERSION,
    description='evaluation tool',
    long_discription=read('README.md'),
    license='BSD',
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            '{} = metric_learning_evaluator.app:main'.format(
                APP_NAME)
        ],
    },
)

setup(**setup_info)
