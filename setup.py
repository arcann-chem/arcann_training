from setuptools import setup, find_packages

setup(
    name="deepmd_iterative",
    version="1.0.0",
    author="Rolf David",
    author_email="rolf/github_c@slmail.me",
    description="",
    url="",
    license="GNU Affero General Public License v3",
    packages=find_packages(),
    package_data={
        "deepmd_iterative.assets": ["*.json"],
        "deepmd_iterative.assets.jobs": ["training/*.sh"],
        "deepmd_iterative.assets.others": ["*.in", "*.tcl"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=["numpy>=1.17.3"],
)
