from setuptools import setup, find_packages

setup(
    name="evac_swarm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mesa",
        "solara",
        "numpy",
        "matplotlib",
    ],
) 