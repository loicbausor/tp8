from setuptools import setup, find_packages

setup(
    name="customML",
    version="0.1",
    description="Custom ML package which implements ",
    author="Sibaucosor",
    author_email="sicot.lea-marie@hotmail.fr/loic.bausor@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "numpy",
        "sklearn",
        "matplotlib>=3",
        "tensorflow>=2"
    ],
    include_package_data=True
)
