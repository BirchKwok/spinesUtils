from setuptools import find_packages, setup

from pathlib import Path


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


min_reqs = read_requirements(Path('.').joinpath("requirements/min.txt"))
full_reqs = read_requirements(Path('.').joinpath("requirements/full.txt"))

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
    license='Apache 2.0',
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
    include_package_data=True
)
