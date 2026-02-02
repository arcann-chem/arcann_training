from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="arcann_training",
    version="1.0.0",
    author="Rolf David",
    author_email="",
    description="Automated enhanced sampling for reactive machine-learning interatomic potentials.",
    url="https://github.com/arcann-chem/arcann",
    license="GNU Affero General Public License v3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "arcann_training.assets": ["*.json"],
        "arcann_training.assets.others": ["*.in", "*.tcl"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=["numpy>=1.15", "pyyaml>=3.13"],
    entry_points={
        "console_scripts": ["arcann-training=arcann_training.__main__:main"]
    },
)
