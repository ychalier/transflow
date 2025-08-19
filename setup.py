import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

import transflow

setup(
    name='transflow',
    version=transflow.__version__,
    packages=["transflow"],
    include_package_data=True,
    license=transflow.__license__,
    description='Set of tools for transferring optical flow from one media to another.',
    long_description=README,
    url='https://chalier.fr/',
    author=transflow.__author__,
    author_email=transflow.__email__,
    install_requires=[
        "aiohttp",
        "av",
        "netifaces",
        "numpy",
        "opencv-python",
        "Pillow",
        "pygame",
        "pyside6",
        "scipy",
        "tqdm",
        "websockets",
    ],
)