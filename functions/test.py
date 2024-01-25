from typing import Union
import os
os.chdir('/home/riccardo/Dropbox/RIKI/FAO/tools/pyAEZ-climate-module/functions/')
import climate_data_pyAEZ as cliAEZ

help(cliAEZ.climate_data_pyAEZ)

Zambia_climate_data = cliAEZ.climate_data_pyAEZ(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", model=1, years_up_to=2030, obs=False, bias_correction=True)
