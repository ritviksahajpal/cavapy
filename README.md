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
import cavapy
# check documentation
help(cavapy.get_climate_data)

```
Depending on the interest, downloading climate data can be done in a few different ways. Note that GCM stands for General Circulation Model while RCP stands for Regional Climate Model. As the climate data comes from the CORDEX-CORE initiative, users can choose between 3 different GCMs downscaled with two RCMs. In total, there are six simulations for any given domain (except for CAS-22 where only three are available).

Since bias-correction requires both the historical run of the CORDEX model and the observational dataset (in this case ERA5), even when the historical argument is set to False, the historical run will be used for learning the bias-correction factor.

```
### Bias-corrected climate projections
Zambia_climate_data = cavapy.get_climate_data(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=True, historical=False)
### Non bias-corrected climate projections
Zambia_climate_data = cavapy.get_climate_data(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=False, historical=False)
### Bias-corrected climate projections plus the historical run. This is useful when assessing changes in crop yield from the historical period. In this case, we provide the bias-corrected # historical run of the climate models plus the projections. 
Zambia_climate_data = cavapy.get_climate_data(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=True, historical=True)
### Observations only (ERA5)
Zambia_climate_data = cavapy.get_climate_data(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=True, bias_correction=True, historical=True, years_obs=range(1980,2019))
```