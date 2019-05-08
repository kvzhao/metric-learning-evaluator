import os
import sys
from setuptools import setup, find_packages


# ml, short for `metric learning`
EVAL_APP_NAME = 'ml-evaluation'
INFERENCE_APP_NAME = 'ml-inference'
VERSION = '0.1.0'

ROOT_FOLDER = 'metric_learning_evaluator'
APP_FOLDER = 'application'

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

install_requires = [
    'pyyaml',
    'scipy',
    'sklearn',
    'pytablewriter'
]

eval_setup_info = dict(
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
            '{app_name} = {root_folder}.{app_folder}.eval_app:main'.format(
                app_name=EVAL_APP_NAME,
                root_folder=ROOT_FOLDER,
                app_folder=APP_FOLDER)
        ],
    },
)
# Install evaluation
setup(**eval_setup_info)

inference_setup_info = dict(
    name='metric-learning-inference',
    author='Viscovery',
    version=VERSION,
    description='Inference tool',
    long_discription=read('README.md'),
    license='BSD',
    #install_requires=install_requires,
    include_package_data=True,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            '{app_name} = {root_folder}.{app_folder}.inference_app:main'.format(
                app_name=INFERENCE_APP_NAME,
                root_folder=ROOT_FOLDER,
                app_folder=APP_FOLDER)
        ],
    },
)
# Install inference tools
setup(**inference_setup_info)