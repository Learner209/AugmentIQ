from setuptools import setup, find_packages

setup(
    name='augmentIQ',
    version='0.1.0',
    packages=find_packages(include=['ImageReward', 'ImageReward.*', 'augmentIQ', 'augmentIQ.*'])
)
