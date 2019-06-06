from setuptools import setup

setup(
    name='flora',
    version='0.0.1',
    packages=[
	'sgan',
	'sgan.model',
	'sgan.evaluation',
	'sgan.context',
	'sgan.model',
	'scripts',
	'scripts.helpers',
	'scripts.evaluation'
    ],
    install_requires=[
        'matplotlib',
        'torch',
	'torchvision',
	'numpy==1.15.0'
    ],
)
