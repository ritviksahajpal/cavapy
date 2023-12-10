# Introduction
This project is about creating a Python function to automatically get the required climate data needed to run pyAEZ. This can be an optional feature of the pyAEZ climate module. 

## Data source
The climate data is available at the THREDDS data server of the University of Cantabria as part of the CAVA (Climate and Agriculture Risk Visualization and Assessment) product developed by FAO, the University of Cantabria, the University of Cape Town and Predictia. 
CAVA has available CORDEX-CORE climate models, the high resolution (25 Km) dynamically-downscaled climate models used in the IPCC report AR5. Additionally, CAVA  offers access to state-of-the-art reanalyses datasets, such as W5E5 and ERA5.

The currently available data is:

CORDEX-CORE simulations (3 GCMs donwscaled with 2 RCMs for two RCPs)
W5E5 and ERA5 reanalyses datasets
Available variables are:

Daily maximum temperature (tasmax)
Daily minimum temperature (tasmin)
Daily precipitation (pr)
Daily relative humidity (hurs)
Daily wind speed (sfcWind)
Daily solar radiation (rsds)
