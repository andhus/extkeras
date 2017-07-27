from setuptools import setup, find_packages

VERSION = '0.0.2'

setup(
    name='extkeras',
    version=VERSION,
    description='Playground for external Keras extras',
    url='https://github.com/andhus/extkeras',
    license='MIT',
    install_requires=[
        'numpy>=1.13.0',
        'Keras>=2.0.6',
    ],
    extras_require={
        'h5py': ['h5py'],
    },
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
)
