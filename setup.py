from setuptools import setup, find_packages

setup(
    name="arcann_training",
    version="1.0.0",
    author="Rolf David",
    author_email="",
    description="",
    url="https://github.com/arcann-chem/arcann",
    license="GNU Affero General Public License v3",
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
    ],
    python_requires=">=3.7",
    install_requires=["numpy>=1.17.3", "pyyaml>=6"],
)
