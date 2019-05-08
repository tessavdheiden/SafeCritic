from setuptools import setup

setup(
    name='flora',
    version='0.0.1',
    packages=[
	'sgan',
	'scripts',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
	'torchvision',
    ],
)
