from setuptools import find_packages
from distutils.core import setup

setup(
    name='mhc',
    version='1.0.0',
    author='Aayam, Pan Liu',
    packages=find_packages(),
    description='',
    python_requires='>=3.8',
    install_requires=[
            'isaacgym',
            "gym",
            "torch",
            "omegaconf",
            "termcolor",
            "hydra-core>=1.1",
            "rl-games==1.1.4"]
)