from setuptools import setup, find_packages

setup(
    name='baseline_human_activity_recognition',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'toml',
        'kagglehub',
    ],
)
