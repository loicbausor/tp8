from setuptools import setup, find_packages

setup(
    name="customML",
    version="0.1",
    description="custom ML package",
    author="***",
    author_email="***",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
    ],
    include_package_data=True,
)
