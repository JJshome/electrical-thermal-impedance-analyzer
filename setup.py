"""Setup script for the electrical-thermal-impedance-analyzer package."""

from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version from _version.py
version = {}
with open(os.path.join("impedance_analyzer", "_version.py")) as f:
    exec(f.read(), version)

setup(
    name="electrical-thermal-impedance-analyzer",
    version=version["__version__"],
    description="Integrated Electrical-Thermal Impedance Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="JJshome",
    author_email="your_email@example.com",
    url="https://github.com/JJshome/electrical-thermal-impedance-analyzer",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "pandas>=1.0.0",
        "matplotlib>=3.2.0",
        "seaborn>=0.10.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
            "jupyter>=1.0.0",
            "sphinx>=3.2.0",
            "sphinx_rtd_theme>=0.5.0",
        ],
        "ai": [
            "tensorflow>=2.5.0",
            "scikit-learn>=0.24.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    keywords="impedance, spectroscopy, electrical, thermal, analysis",
)