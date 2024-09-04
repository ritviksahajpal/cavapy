#### **Status

**Beta testing**

# Introduction
This project is about creating a Python function to automatically get the required climate data needed to run pyAEZ. This can be an optional feature of the pyAEZ climate module. 

## Data source
The climate data is available at the THREDDS data server of the University of Cantabria as part of the CAVA (Climate and Agriculture Risk Visualization and Assessment) product developed by FAO, the University of Cantabria, the University of Cape Town and Predictia. 
CAVA has available CORDEX-CORE climate models, the high resolution (25 Km) dynamically-downscaled climate models used in the IPCC report AR5. Additionally, CAVA  offers access to state-of-the-art reanalyses datasets, such as W5E5 and ERA5.

The currently available data is:

- CORDEX-CORE simulations (3 GCMs donwscaled with 2 RCMs for two RCPs)
- W5E5 and ERA5 reanalyses datasets
  
Available variables are:

- Daily maximum temperature (tasmax)
- Daily minimum temperature (tasmin)
- Daily precipitation (pr)
- Daily relative humidity (hurs)
- Daily wind speed (sfcWind)
- Daily solar radiation (rsds)



## Usage
The function can be downloaded from the script folder and imported, for example, as follow:

```
import os
os.chdir('/path/to/function')
import climate_data_pyAEZ as cliAEZ
# check documentation
help(cliAEZ.climate_data_pyAEZ)

```
Downloading climate data can be done in a few different ways

```
### Bias-corrected climate projections with the empirical quantile mapping method
Zambia_climate_data = cliAEZ.climate_data_pyAEZ(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=True, historical=False, xlim=None, ylim=None, years_obs=None)
### Non bias-corrected climate projections
Zambia_climate_data = cliAEZ.climate_data_pyAEZ(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=False, historical=False, xlim=None, ylim=None, years_obs=None)
### Bias-corrected climate projections plus the historical run
Zambia_climate_data = cliAEZ.climate_data_pyAEZ(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=True, historical=True, xlim=None, ylim=None, years_obs=None)
### Observations only
Zambia_climate_data = cliAEZ.climate_data_pyAEZ(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=True, bias_correction=True, historical=True, xlim=None, ylim=None, years_obs=range(1980,2019))
```



