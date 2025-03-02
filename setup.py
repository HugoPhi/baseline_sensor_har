from setuptools import setup, find_packages

setup(
    name='ml',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'toml'
    ],
)
