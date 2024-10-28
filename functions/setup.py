from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="climate_processing",  # Replace with your desired package name
    version="0.1.0",  # Start with version 0.1.0
    author="Riccardo",  # Replace with your name
    author_email="riccardosoldan@hotmail.it",  # Replace with your email
    description="A package for downloading and processing climate data input for pyAEZ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Risk-Team/pyAEZ-climate-module",  # Replace with your GitHub repository URL if applicable
    packages=find_packages(),  # Automatically find the packages
    install_requires=[
        "geopandas",
        "netCDF4",
        "pandas",
        "xarray",
        "numpy",
        "tqdm",
        "xclim",
        "rich",
        "dask",
        "shapely"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update with your preferred license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
