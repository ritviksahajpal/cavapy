# Required libraries
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from typing import Union
from tqdm import tqdm, trange
from xclim import sdba
from rich import print
import dask
from dask import delayed
from shapely.geometry import box

# Function to geo-localize based on country or bounding box
def geo_localize(country=None, xlim=None, ylim=None, buffer=0):
    if country is not None and (xlim is not None or ylim is not None):
        raise ValueError("Either select a country or specify a region (xlim and ylim), not both")

    if country is not None:
        # Load the country shapefile manually (download from Natural Earth website)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))  # Update the path
        country_shp = world[world.name == country]

        if country_shp.empty:
            raise ValueError(f"Country '{country}' not found in shapefile.")

        bounds = country_shp.total_bounds  # [minx, miny, maxx, maxy]

    elif xlim is not None and ylim is not None:
        # Create a bounding box based on provided xlim and ylim
        bbox = gpd.GeoSeries([box(min(xlim), min(ylim), max(xlim), max(ylim))])
        bounds = bbox.total_bounds  # [minx, miny, maxx, maxy]
    else:
        raise ValueError("Either country or both xlim and ylim must be provided")

    # Apply buffer
    xlim = [bounds[0] - buffer, bounds[2] + buffer]
    ylim = [bounds[1] - buffer, bounds[3] + buffer]

    return {'xlim': xlim, 'ylim': ylim}



# Function to download climate data
def download_data(url, bbox, variable, obs, years_obs, years_up_to, remote):
    variable_map = {
        "pr": "tp",
        "tasmax": "t2mx",
        "tasmin": "t2mn",
        "hurs": "hurs",
        "sfcWind": "sfcwind",
        "tas": "t2m",
        "rsds": "ssrd"
    }

    if obs:
        var = variable_map[variable]
        if remote:
          ds_var = xr.open_dataset("https://hub.ipcc.ifca.es/thredds/dodsC/fao/observations/ERA5/0.25/ERA5_025.ncml")[var]
        else:
          ds_var = xr.open_dataset("/home/jovyan/shared/data/observations/ERA5/0.25/ERA5_025.ncml")[var]

        # Coordinate normalization and renaming for 'hurs'
        if var == 'hurs':
            ds_var = ds_var.rename({'lat': 'latitude', 'lon': 'longitude'})
            ds_cropped = ds_var.sel(longitude=slice(bbox["xlim"][0], bbox["xlim"][1]), latitude=slice(bbox["ylim"][0], bbox["ylim"][1]))
        else:
            ds_var.coords['longitude'] = (ds_var.coords['longitude'] + 180) % 360 - 180
            ds_var = ds_var.sortby(ds_var.longitude)
            ds_cropped = ds_var.sel(longitude=slice(bbox["xlim"][0], bbox["xlim"][1]), latitude=slice(bbox["ylim"][1], bbox["ylim"][0]))

        # Unit conversion
        if var in ['t2mx', 't2mn', 't2m']:
            ds_cropped -= 273.15  # Convert from Kelvin to Celsius
            ds_cropped.attrs['units'] = '°C'
        elif var == 'tp':
            ds_cropped *= 1000  # Convert precipitation
            ds_cropped.attrs['units'] = 'mm'
        elif var == 'ssrd':
            ds_cropped /= 86400  # Convert from J/m^2 to W/m^2
            ds_cropped.attrs['units'] = 'W m-2'
        elif var == 'sfcwind':
            ds_cropped = ds_cropped * (4.87 / np.log((67.8 * 10) - 5.42))  # Convert wind speed from 10 m to 2 m
            ds_cropped.attrs['units'] = 'm s-1'

        # Select years
        years = [x for x in years_obs]
        time_mask = (ds_cropped['time'].dt.year >= years[0]) & (ds_cropped['time'].dt.year <= years[-1])

    else:
        ds_var = xr.open_dataset(url)[variable]
        ds_cropped = ds_var.sel(longitude=slice(bbox["xlim"][0], bbox["xlim"][1]), latitude=slice(bbox["ylim"][1], bbox["ylim"][0]))

        # Unit conversion
        if variable in ['tas', 'tasmax', 'tasmin']:
            ds_cropped -= 273.15  # Convert from Kelvin to Celsius
            ds_cropped.attrs['units'] = '°C'
        elif variable == 'pr':
            ds_cropped *= 86400  # Convert from kg m^-2 s^-1 to mm/day
            ds_cropped.attrs['units'] = 'mm'
        elif variable == 'rsds':
            ds_cropped.attrs['units'] = 'W m-2'
        elif variable == 'sfcWind':
            ds_cropped = ds_cropped * (4.87 / np.log((67.8 * 10) - 5.42))  # Convert wind speed from 10 m to 2 m
            ds_cropped.attrs['units'] = 'm s-1'

        # Select years based on rcp
        if "rcp" in url:
            years = [x for x in range(2006, years_up_to + 1)]
        else:
            years = [x for x in range(1980, 2006)]

        # Add missing dates
        ds_cropped = ds_cropped.convert_calendar(calendar='gregorian', missing=np.nan, align_on="date")

        time_mask = (ds_cropped['time'].dt.year >= years[0]) & (ds_cropped['time'].dt.year <= years[-1])

    # subset years
    ds_cropped = ds_cropped.sel(time=time_mask)

    return ds_cropped

# Function for climate data processing

def climate_data(country, cordex_domain, rcp, gcm, rcm, years_up_to, variable, years_obs, obs, bias_correction, historical, buffer, xlim, ylim, remote):
    # Validate inputs
    remote=True
    valid_variables = ["rsds", "tasmax", "tasmin", "pr", "sfcWind", "hurs"]
    valid_domains = ["AFR-22", "EAS-22", "SEA-22", "WAS-22", "AUS-22", "SAM-22", "CAM-22"]
    valid_rcps = ["rcp26", "rcp85"]
    valid_gcm = ["MOHC", "MPI", "NCC"]
    valid_rcm = ["REMO", "Reg"]

    if variable not in valid_variables:
        raise ValueError(f"Invalid variable. Must be one of {valid_variables}")
    if cordex_domain not in valid_domains:
        raise ValueError(f"Invalid domain. Must be one of {valid_domains}")
    if rcp not in valid_rcps:
        raise ValueError(f"Invalid RCP. Must be one of {valid_rcps}")
    if gcm not in valid_gcm:
        raise ValueError(f"Invalid GCM. Must be one of {valid_gcm}")
    if rcm not in valid_rcm:
        raise ValueError(f"Invalid RCM. Must be one of {valid_rcm}")
    if years_obs is not None and not (1980 <= min(years_obs) <= max(years_obs) <= 2020):
        raise ValueError("Years in years_obs must be within the range 1980 to 2020")

    # Geo-localize
    bbox = geo_localize(country=country, xlim=xlim, ylim=ylim, buffer=buffer)
    csv_url_remote = "https://hub.ipcc.ifca.es/thredds/fileServer/inventories/cava.csv"
    csv_url_local = "/home/jovyan/shared/inventories/cava/inventory.csv"

    # Read CSV data into a pandas DataFrame
    pd.options.mode.chained_assignment = None
    data = pd.read_csv(csv_url_remote) if remote else pd.read_csv(csv_url_local)
    column_to_use = 'location' if remote else 'hub'  # Decide which column to use based on `remote`
    filtered_data = data[
        (data['activity'].str.contains("FAO", na=False)) &
        (data['domain'] == cordex_domain) &
        (data['model'].str.contains(gcm, na=False)) &
         (data['rcm'].str.contains(rcm, na=False)) &
        (data['experiment'].isin([rcp, 'historical']))
    ]
    filtered_data = filtered_data[['experiment', column_to_use]]  # Use the correct column

    if not obs:
        downloaded_models = []
        # Use tqdm to iterate through the URLs and download the models
        for url in tqdm(filtered_data[column_to_use], desc=f"Downloading selected CORDEX-CORE climate model for {variable} for the specified rcp plus the historical run"):
            model_data = download_data(url=url, bbox=bbox, variable=variable, obs=False, years_up_to=years_up_to, years_obs=years_obs, remote=remote)
            downloaded_models.append(model_data)

        # Add the downloaded models to the DataFrame
        filtered_data['models'] = downloaded_models
        print("[bold yellow]360-calendar converted into Gregorian calendar and missing values linearly interpolated[/bold yellow]")
        hist = filtered_data['models'].iloc[0].interpolate_na(dim='time', method='linear')
        proj = filtered_data['models'].iloc[1].interpolate_na(dim='time', method='linear')

        if bias_correction and historical:
            # Load observations for bias correction
            downloaded_obs = []
            with trange(1, desc=f"Downloading observations (ERA5) for {variable} used for bias correction") as t:
                for _ in t:
                    obs_model = download_data(url="not_needed", bbox=bbox, variable=variable, obs=True, years_up_to=years_up_to, years_obs=range(1980, 2006), remote=remote)
                    downloaded_obs.append(obs_model)
            ref = downloaded_obs[0]
            QM_mo = sdba.EmpiricalQuantileMapping.train(ref, hist, group='time.month', kind='*' if variable in ['pr', 'rsds', 'sfcWind'] else '+')
            print("[bold yellow]Performing bias correction with eqm[/bold yellow]")
            hist_bs = QM_mo.adjust(hist, extrapolation="constant", interp="linear")
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")
            print("[bold yellow]Done![/bold yellow]")
            if variable == 'hurs':
                hist_bs = hist_bs.where(hist_bs <=
                100, 100)
                hist_bs = hist_bs.where(hist_bs >=
                0, 0)
            combined = xr.concat([hist_bs, proj_bs], dim='time')
            return combined

        elif not bias_correction and historical:
            combined = xr.concat([hist, proj], dim='time')
            return combined

        elif bias_correction and not historical:
            downloaded_obs = []
            with trange(1, desc=f"Downloading observations (ERA5) for {variable} used for bias correction") as t:
                for _ in t:
                    obs_model = download_data(url="not_needed", bbox=bbox, variable=variable, obs=True, years_up_to=years_up_to, years_obs=range(1980, 2006), remote=remote)
                    downloaded_obs.append(obs_model)
            ref = downloaded_obs[0]
            print("[bold yellow]Performing bias correction with eqm[/bold yellow]")
            QM_mo = sdba.EmpiricalQuantileMapping.train(ref, hist, group='time.month',  kind='*' if variable in ['pr', 'rsds', 'sfcWind'] else '+') # multiplicative approach for pr, rsds and wind speed
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")
            print("[bold yellow]Done![/bold yellow]")
            if variable == 'hurs':
                proj_bs = proj_bs.where(proj_bs <= 100, 100)
                proj_bs = proj_bs.where(proj_bs >= 0, 0)
            return proj_bs

    else:  # when observations are True
        downloaded_obs = []
        with trange(1, desc=f"Downloading observations (ERA5) for {variable}") as t:
            for _ in t:
                obs_model = download_data(url="not_needed", bbox=bbox, variable=variable, obs=True, years_up_to=years_up_to, years_obs=years_obs, remote=remote)
                downloaded_obs.append(obs_model)
        print("[bold yellow]Done![/bold yellow]")        
        return downloaded_obs[0]


# Function to process climate data for multiple variables
def get_climate_data(country, cordex_domain, rcp, gcm, rcm, years_up_to, years_obs: Union[range, None] = None, bias_correction=False, historical=False, obs=False, buffer=0, xlim=None, ylim=None, remote=True, cores=6):
    """
    Process climate data required by pyAEZ climate module. The function automatically access CORDEX-CORE models at 0.25° and the ERA5 datasets.

    Args:
    country (str): Name of the country for which data is to be processed. Use None if specifying a region using xlim and ylim.
    cordex_domain (str): CORDEX domain of the climate data (e.g. AFR-22, EAS-22, SEA-22). 
    rcp (str): Representative Concentration Pathway (e.g., 'rcp26', 'rcp85').
    gcm (str): GCM name. (one of 'MPI', 'NCC', "MOHC").
    rcm (str): RCM name. (One of 'REMO', 'Reg').
    years_obs (range): Range of years for observational data (ERA5 only). Only used when obs is True. (default: None).
    years_up_to (int): The ending year for the projected data. Projections start in 2006 and ends in 2100. Hence, if years_up_to is set to 2030, data will be downloaded for the 2006-2030 period.
    obs (bool): Flag to indicate if processing observational data (default: False).
    bias_correction (bool): Flag to apply bias correction (default: False).
    historical (bool): Flag to indicate if processing historical data (default: False). If True, historical data is provided together with projections. Historical simulation runs for CORDEX-CORE initiative are provided for the 1980-2005 time period.
    buffer (int): Buffer distance to expand the region of interest (default: 0).
    xlim (list or None): Longitudinal bounds of the region of interest. Specify only when 'country' is None (default: None).
    ylim (list or None): Latitudinal bounds of the region of interest. Specify only when 'country' is None (default: None).
    remote (bool): Flag to work with remote data or not (default: True).
    cores (int): Number of cores to use (default: 6).

    Returns:
    dict: A dictionary containing processed climate data for each variable as an xarray object.
    """
    
    dask.config.set(scheduler='threads', num_workers=cores, threads_per_worker=1)

    results = {}

    # List of variables to process
    
    variables = ["tasmin", "pr", "hurs", "tasmax", "sfcWind", "rsds"]
  
    # Use Dask delayed to parallelize the processing of each variable
    delayed_results = []
    for variable in variables:

        delayed_result = delayed(climate_data)(
            country=country,
            xlim=xlim,
            ylim=ylim,
            cordex_domain=cordex_domain,
            rcp=rcp,
            gcm=gcm,
            rcm=rcm,
            years_obs=years_obs,
            years_up_to=years_up_to,
            bias_correction=bias_correction,
            historical=historical,
            variable=variable,
            obs=obs,
            buffer=buffer,
            remote=remote
        )
        delayed_results.append(delayed_result)

    # Compute all delayed results
    computed_results = dask.compute(*delayed_results)

    # Store results in a dictionary
    for i, variable in enumerate(variables):
        results[variable] = computed_results[i]

    return results
