# Climate Data Processing Tool

This project provides a tool to download, process, and bias-correct climate data for specific regions or countries using CORDEX-CORE and ERA5 datasets. The tool is designed to facilitate climate data analysis for the pyAEZ climate module by automatically handling data acquisition, processing, and conversion into usable formats.

## Features

- Download and process climate data for both historical and future scenarios.
- Support for various climate variables such as temperature, precipitation, wind speed, relative humidity, and solar radiation.
- Option to perform bias correction using Empirical Quantile Mapping (EQM).
- Ability to handle large datasets in parallel using Dask.
- Geo-localization support to fetch climate data for specific countries or custom bounding boxes.
- Option to work with remote datasets or local storage.
  
## Supported Data

- **CORDEX-CORE climate models** at 0.25° resolution for specific Representative Concentration Pathways (RCPs) (e.g., RCP 2.6, RCP 8.5).
- **ERA5 observational data** for the period from 1980 to 2020.
  
## Requirements

- Python 3.7 or above
- Required libraries:
    - `xarray`
    - `geopandas`
    - `pandas`
    - `numpy`
    - `xclim`
    - `dask`
    - `shapely`
    - `tqdm`
    - `rich`
    
Install the required dependencies using:

```bash
pip install -r requirements.txt
