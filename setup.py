from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spinesUtils',
    version="0.4.0",
    description='spinesUtils is a user-friendly toolkit for the machine learning ecosystem.',
    keywords='machine learning',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    url='https://github.com/BirchKwok/spinesUtils',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'scikit-learn>=1.0.2',
        'numpy>=1.17.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'dask>=2023.6.0',
        'pyarrow>=11.0.0',
        'polars>=0.19.3',
        'pytz>=2021.1'
    ],
    zip_safe=False,
    include_package_data=True
)
