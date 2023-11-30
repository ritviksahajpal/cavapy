library(magrittr)


# parameters --------------------------------------------------------------


obs=F
cordex_domain= "AFR-22" # "AFR-22,SEA-22, EAS-22
rcp="rcp26" # rcp26 or rcp85
model=1 # 1:6
years_obs <- 2010:2019 # only when obs is TRUE
years_up_to <- 2030 # 2006 to 2100
buffer <- 0 # default
country <- "Togo"
xlim <- NULL # specify only when country is NULL
ylim <- NULL# specify only when country is NULL


# localization ------------------------------------------------------------


# intermediate functions. Geolocalization
geo_localize <- function(country, xlim, ylim, buffer) {
    if (!is.null(country) & !is.null(xlim)) {
      cli::cli_abort(c("x" = "Either select a country or a region of interest, not both"))
    } else {
      country_shp = if (!is.null(country)) {
        suppressMessages(
          rnaturalearth::ne_countries(
            country = country,
            scale = "medium",
            returnclass = "sf"
          ) %>%
            sf::st_set_crs(., NA)
        )
      } else {
        sf::st_bbox(c(
          xmin = min(xlim),
          xmax = max(xlim),
          ymax = max(ylim),
          ymin = min(ylim)
        )) %>%
          sf::st_as_sfc() %>%
          data.frame(geometry = .) %>%
          sf::st_as_sf() %>%
          sf::st_set_crs(., NA)
      }
      
      xlim <-
        c(sf::st_bbox(country_shp)[1] - buffer,
          sf::st_bbox(country_shp)[3] + buffer)
      ylim <-
        c(sf::st_bbox(country_shp)[2] - buffer,
          sf::st_bbox(country_shp)[4] + buffer)
      return(list(
        xlim = xlim,
        ylim = ylim
      ))
    }
}


# load_data_and_bc --------------------------------------------------------

# load_data and bias correct when obs is FALSE
load_data_and_bc <- function(cordex_domain, xlim, ylim, years_up_to, rcp, model) {
csv_url <- "https://data.meteo.unican.es/inventory.csv"

data_list <- purrr::map(c("tasmax", "tasmin", "pr", "rsds", "sfcWind", "hurs"), function(variable) {
  data <- read.csv(url(csv_url)) %>%
  dplyr::filter(stringr::str_detect(activity, "FAO"), domain ==  cordex_domain) %>% 
  dplyr::group_by(experiment) %>%
  dplyr::summarise(location = list(as.character(location))) %>% 
  dplyr::filter(experiment%in%c(rcp, "historical")) %>% 
  dplyr::mutate(location=purrr::map_chr(location, ~ purrr::chuck(.x, model))) %>% 
  dplyr::mutate(models = suppressWarnings(purrr::map(location, function(x)  {
    if (stringr::str_detect(x, "historical")) {
    cli::cli_progress_step(paste0("Downloading ", variable, " for the historical period."))
      data <-
          suppressMessages(loadeR::loadGridData(
            dataset = x,
            var = variable,
            years = 1980:2005,
            lonLim = xlim,
            latLim = ylim
          )) %>% 
        {
          if (stringr::str_detect(variable, "tas")) {
            suppressMessages(transformeR::gridArithmetics(., 273.15, operator = "-"))
          } else if (stringr::str_detect(variable, "pr")) {
            suppressMessages(transformeR::gridArithmetics(., 86400, operator = "*"))} else {
              
            suppressMessages(transformeR::gridArithmetics(., 1, operator = "*"))
            }
          
        }
      cli::cli_process_done()
      return(data)
    } else {
      cli::cli_progress_step(paste0("Downloading ", variable, " for the future period"))
      data <-
          suppressMessages(loadeR::loadGridData(
            dataset = x,
            var = variable,
            years = 2006:years_up_to,
            lonLim = xlim,
            latLim = ylim
          )) %>%
        { 
          
          if (stringr::str_detect(variable, "tas")) {
            suppressMessages(transformeR::gridArithmetics(., 273.15, operator = "-"))
          } else if (stringr::str_detect(variable, "pr")) {
            suppressMessages(transformeR::gridArithmetics(., 86400, operator = "*"))
            
          } else {
            
            suppressMessages(transformeR::gridArithmetics(., 1, operator = "*"))   
          }
          
        } 
      cli::cli_process_done()
      return(data)
    }
    
  }))) 
  
# load obs
  cli::cli_progress_step(paste0("Downloading ", variable, " from ERA5. Used for bias correction"))
  out_obs <- suppressMessages(loadeR::loadGridData(
    "https://data.meteo.unican.es/thredds/dodsC/copernicus/cds/ERA5_0.25",
      var = c(
          "pr" = "tp",
          "tasmax" = "t2mx",
          "tasmin" = "t2mn",
          "hurs" = "hurs",
          "sfcWind" = "sfcwind",
          "tas" = "t2m",
          "rsds"= "ssrd")[variable],
      years = 1980:2005,
      lonLim = xlim,
      latLim = ylim) %>%
      {
          if (stringr::str_detect(variable, "tas")) {
            obs_tr <- transformeR::gridArithmetics(., 273.15, operator = "-")
            obs_tr$Variable$varName = variable
            obs_tr
          } else if (stringr::str_detect(variable, "pr")) {
            obs_tr <- transformeR::gridArithmetics(., 1000, operator = "*")
            obs_tr$Variable$varName = variable
            obs_tr
          } else {
            obs_tr <- transformeR::gridArithmetics(., 1, operator = "*")
            obs_tr$Variable$varName = variable
            obs_tr
          }
        })
  
data$obs <- list(out_obs)
cli::cli_process_done()

cli::cli_progress_step(paste0("Performing bias correction for ", variable))

  invisible(data %>%
  dplyr::mutate(models= purrr::map2(models, location, function(x,y) {
  if (stringr::str_detect(y, "historical")) {
    suppressMessages(
      downscaleR::biasCorrection(
        y = obs[[1]],
        x = x,
        precipitation = ifelse(variable == "pr", TRUE, FALSE),
        method = "eqm",
        window = c(30, 30),
        cross.val = "kfold", 
        folds = 2, 
        extrapolation = "constant"
      )
    )

  } else {
    suppressMessages(
      downscaleR::biasCorrection(
        y = obs[[1]],
        x = data$models[[1]],
        newdata=x,
        precipitation = ifelse(variable == "pr", TRUE, FALSE),
        method = "eqm",
        window = c(30, 30),
        extrapolation = "constant"
      )
    )

  }

 })))

})

names(data_list) <- c("tasmax", "tasmin", "pr", "rsds", "sfcWind", "hurs")
return(data_list)
}

# load_data_obs when obs is TRUE (observations (ERA5))

# load_data_obs -----------------------------------------------------------
load_data_obs <- function(xlim, ylim, years_up_to, years_obs) {

future::plan(future::multisession, workers = 6)
  
data_list <- furrr::future_map(c("tasmax", "tasmin", "pr", "rsds", "sfcWind", "hurs"), function(variable) {
    cli::cli_progress_step(paste0("Downloading ", variable, " from ERA5"))
    out_obs <- suppressMessages(loadeR::loadGridData(
      "https://data.meteo.unican.es/thredds/dodsC/copernicus/cds/ERA5_0.25",
      var = c(
        "pr" = "tp",
        "tasmax" = "t2mx",
        "tasmin" = "t2mn",
        "hurs" = "hurs",
        "sfcWind" = "sfcwind",
        "tas" = "t2m",
        "rsds"= "ssrd")[variable],
      years =  years_obs,
      lonLim = xlim,
      latLim = ylim) %>%
        {
          if (stringr::str_detect(variable, "tas")) {
            obs_tr <- transformeR::gridArithmetics(., 273.15, operator = "-")
            obs_tr$Variable$varName = variable
            obs_tr
          } else if (stringr::str_detect(variable, "pr")) {
            obs_tr <- transformeR::gridArithmetics(., 1000, operator = "*")
            obs_tr$Variable$varName = variable
            obs_tr
          } else {
            obs_tr <- transformeR::gridArithmetics(., 1, operator = "*")
            obs_tr$Variable$varName = variable
            obs_tr
          }
        })
    
    cli::cli_process_done()
    return(out_obs)
 
  })
  
names(data_list) <- c("tasmax", "tasmin", "pr", "rsds", "sfcWind", "hurs")
return(data_list)
  
}

# executing ---------------------------------------------------------------


result <- geo_localize(country="Cambodia", xlim=NULL, ylim=NULL, buffer=0)
xlim <- result$xlim
ylim <- result$ylim

out <- load_data_and_bc(cordex_domain = "SEA-22", xlim = xlim, ylim = ylim, years_up_to = 2010, rcp = "rcp26", model = 1)


