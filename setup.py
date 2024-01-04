from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='privbayes-implementation',
    version='0.0.1',
    description='Implementing Privbayes for generating synthetic dataset',
    url='https://github.com/0hex7/Privbayes-implementation.git',
    author='Arun Ashok',
    author_email='iamarunbadri@gmail.com',
    package_dir={'': 'privbayes'},
    install_requires=requirements,
)
