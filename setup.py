from setuptools import setup, find_packages, find_namespace_packages

setup(
    name="deepmd_iterative",
    version="1.0.0",
    description="",
    py_modules=["deepmd_iterative"],
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "deepmd_iterative.data": ["*.json"],
        "deepmd_iterative.data.jobs": ["training/*.sh"],
        "deepmd_iterative.data.others": ["*.in", "*.tcl"],
    },
    author="Rolf David",
    author_email="rolf.david23@gmail.com",
)
