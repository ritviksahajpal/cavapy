# Introduction
This project is about creating a Python function to automatically get the required climate data needed to run pyAEZ. This is an optional feature of the pyAEZ climate module. 

## Data source
The climate data is available through CAVA (Climate and Agriculture Risk Visualization and Assessment) developed by FAO.
CAVA has available CORDEX-CORE climate models, the high resolution (25 Km) dinamically-downscaled climate models used in the IPCC report AR5. Additionally, CAVA  offers access to state-of-the-art reanalyses datasets, such as W5E5 and ERA5.

The current available data is:

CORDEX-CORE simulations (3 GCMs donwscaled with 2 RCMs for two RCPs)
W5E5 and ERA5 reanalyses datasets
Available variables are:

Daily maximum temperature (tasmax)
Daily minimum temperature (tasmin)
Daily precipitation (pr)
Daily relative humidity (hurs)
Daily wind speed (sfcWind)
Daily solar radiation (rsds)
## Python
Accessing this datasets can be done with:
```
import xarray as xr

# URL to ERA5 data
obs_url =  "https://data.meteo.unican.es/thredds/dodsC/copernicus/cds/ERA5_0.25"
# URL to W5E5 V2 data
obs_url =    "https://data.meteo.unican.es/thredds/dodsC/mirrors/W5E5/W5E5_v2"
# Open dataset
ds = xr.open_dataset(obs_url)
```

The list of available CORDEX-CORE models can be accessed with:

```
import pandas as pd

csv_url = "https://data.meteo.unican.es/inventory.csv"
data = pd.read_csv(csv_url)

# Drop rows with missing values in the 'activity' column
data = data.dropna(subset=['activity'])

filtered_data = data[data['activity'].str.contains("FAO")]

```
A dedicated function to automatically perform these steps will be made available soon. 
