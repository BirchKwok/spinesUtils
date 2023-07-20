from setuptools import Extension, dist, find_packages, setup


setup(
    name='spines',
    version="0.0.1",
    description='模型训练工具集 model trainning  toolsets',
    keywords='computer vision',
    packages=find_packages(),
    long_description='./README.md',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    url='https://github.com/BirchKwok/spines',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'scikit-learn>=1.0.2',
        'torch>=1.4',
        'scipy>=1.7.0',
        'numpy>=1.17.0',
        'pandas>=1.0.0',
        'tabulate>=0.8',
        'tqdm>=4.65.0',
        'matplotlib>=3.7.1',
        'dask>=2023.6.0',
        'shap>=0.41.0'
    ],
    zip_safe=False,
    include_package_data=True
)
