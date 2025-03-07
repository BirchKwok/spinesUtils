from setuptools import find_packages, setup
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

def read_requirements(filename):
    with open(CURRENT_DIR / "requirements" / filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

min_reqs = read_requirements("min.txt")
full_reqs = read_requirements("full.txt")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spinesUtils',
    version="0.5.0",
    description='spinesUtils is a user-friendly toolkit for python development.',
    keywords='Helpful tools for python development',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
    url='https://github.com/BirchKwok/spinesUtils',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=min_reqs,
    extras_require={"all": full_reqs},
    zip_safe=False,
    include_package_data=True,
    package_data={
        '': ['requirements/*.txt']
    }
)
