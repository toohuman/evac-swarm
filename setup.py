from setuptools import setup, find_packages

setup(
    name="evac_swarm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mesa>=2.1.1",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "rtree>=1.0.1",  # For spatial indexing
        "solara>=1.21.0",  # For the web interface
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    description="A Mesa-based simulation of robot swarms exploring buildings"
)
