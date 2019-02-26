import os
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'
cur_dir = os.path.dirname(__file__)

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

long_discription = '''Metric Learning Evaluator used for both on-line
and off-line application.
'''

application_name = 'ml_evaluator'

install_requires = [
    'pyyaml', 'scipy', 'sklearn'
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
                application_name)
        ],
    },
)

setup(**setup_info)
