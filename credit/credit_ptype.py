from credit.interp import create_reduced_pressure_grid, geopotential_from_model_vars, interp_hybrid_to_height_agl
from credit.physics_constants import RDGAS, GRAVITY
from metpy.calc import dewpoint_from_specific_humidity
from metpy.units import units
import numpy as np
from numba import njit
import xarray as xr
from tqdm import tqdm
import os
from bridgescaler import load_scaler
import pandas as pd
from keras.models import load_model
from mlguess.keras.models import CategoricalDNN
from mlguess.keras.losses import evidential_cat_loss
import logging
logger = logging.getLogger(__name__)

class CreditPostProcessor:
    def __init__(self, levels=None, interp_var=None, interp_heights_m=None):
        """
        Initialize the CreditPostProcessor with required paths and configurations.
        """
        self.model_params_path = "/glade/work/sakor/miles-credit/credit/metadata/ERA5_Lev_Info.nc"
        self.surface_geopotential_path = "/glade/derecho/scratch/ksha/CREDIT_data/static_norm_old.nc"
        
        self.levels = levels or np.array([10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]) - 1
        self.interp_var = interp_var or ['T', 'DPT', 'U', 'V']
        self.interp_heights_m = interp_heights_m or np.arange(0, 5250, 250)
        
        self.model_params = xr.open_dataset(self.model_params_path)
        self.surface_geopotential_ds = xr.open_dataset(self.surface_geopotential_path)
        self.surface_geopotential = self.surface_geopotential_ds.Z_GDS4_SFC.values
        self.data = None

        self.save_vars = ['ML_u',
                         'ML_rain_ale',
                         'ML_rain_epi',
                         'ML_snow_ale',
                         'ML_snow_epi',
                         'ML_icep_ale',
                         'ML_icep_epi',
                         'ML_frzr_ale',
                         'ML_frzr_epi',
                         'ML_rain',
                         'ML_crain',
                         'ML_snow',
                         'ML_csnow',
                         'ML_icep',
                         'ML_cicep',
                         'ML_frzr',
                         'ML_cfrzr']

    def process_credit_output(self, all_upper_air, all_single_level):

        lat, lon = all_single_level.latitude.values, all_single_level.longitude.values
        time_values = all_single_level.time.values
        
        hgl_data = {}
        
        for t_idx in range(len(time_values)):
            # Compute 3D pressure for current time step
            pressure_3d, pressure_3d_half = create_reduced_pressure_grid(
                all_single_level.SP.values,
                self.model_params.a_model.sel(level=self.levels).values,
                self.model_params.b_model.sel(level=self.levels).values
            )

            # Compute geopotential height
            model_geopotential = geopotential_from_model_vars(
                self.surface_geopotential,
                all_single_level.SP.isel(time=t_idx).values,
                all_upper_air.T.isel(time=t_idx).values,
                all_upper_air.Q.isel(time=t_idx).values,
                pressure_3d_half
            )

            # Compute dew point temperature
            temp_3d = all_upper_air.T.isel(time=t_idx).values
            q_3d = all_upper_air.Q.isel(time=t_idx).values
            dew_point_3d = np.empty_like(temp_3d)
            for i in range(temp_3d.shape[0]):
                dew_point_3d[i, ...] = dewpoint_from_specific_humidity(
                    pressure_3d[i, ...] * units.Pa, q_3d[i, ...] * units.kg / units.kg
                )
           
            for var in self.interp_var:
                var_name_3d = var.lower()
                data_source = dew_point_3d if var == 'DPT' else all_upper_air[var].isel(time=t_idx).values
                
                d_interp = interp_hybrid_to_height_agl(
                    data_source,
                    np.flip(self.interp_heights_m),  # flip to go from  top of the atmosphere to surface
                    model_geopotential,
                    self.surface_geopotential
                )

                if var_name_3d == 't':  # Convert to Celsius
                    d_interp -= 273.15

                if var_name_3d not in hgl_data:
                    hgl_data[var_name_3d] = (
                        ['time', 'height', 'latitude', 'longitude'],
                        np.empty((len(time_values), len(self.interp_heights_m), len(lat), len(lon)))
                    )

                hgl_data[var_name_3d][1][t_idx, ...] = np.flip(d_interp, axis=0)

        var_attributes = {
            't': {'units': 'C', 'long_name': 'Temperature'},
            'dpt': {'units': 'C', 'long_name': 'Dew point temperature'},
            'u': {'units': 'm/s', 'long_name': 'Zonal wind'},
            'v': {'units': 'm/s', 'long_name': 'Meridional wind'},
        }
        ds = xr.Dataset(
            hgl_data,
            coords={
                'time': (['time'], time_values),
                'height': (['height'], self.interp_heights_m),
                'latitude': (['latitude'], lat),
                'longitude': (['longitude'], lon)
            }
        )
        for var in var_attributes:
            if var in ds:
                ds[var].attrs = var_attributes[var]

        return ds

    def convert_longitude(self,lon):
        """ Convert longitude from -180-180 to 0-360"""
        return lon % 360

    def subset_extent(self, nwp_data, extent, data_proj=None):
        """
        Subset data by given extent in projection coordinates
        Args:
            nwp_data: Xr.dataset of NWP data
            extent: List of coordinates for subsetting (lon_min, lon_max, lat_min, lat_max)
            transformer: Pyproj Projection transformer object

        Returns:
            Subsetted Xr.Dataset
        """
        lon_min, lon_max, lat_min, lat_max = extent
        if data_proj is not None:
            x_coords, y_coords = data_proj(np.array([lon_min, lon_max], dtype=np.float64),
                                                       np.array([lat_min, lat_max], dtype=np.float64))
            subset = nwp_data.swap_dims({'y': 'y_projection_coordinate', 'x': 'x_projection_coordinate'}).sel(
                y_projection_coordinate=slice(y_coords[0], y_coords[1]),
                x_projection_coordinate=slice(x_coords[0], x_coords[1])).swap_dims(
                    {'y_projection_coordinate': 'y', 'x_projection_coordinate': 'x'})
        else:
            subset = nwp_data.sel(longitude=slice(self.convert_longitude(lon_min), self.convert_longitude(lon_max)),
                                  latitude=slice(lat_max, lat_min))  
        return subset

    def extract_variable_levels(self, data: xr.Dataset) -> np.ndarray:
        """
        Extracts data from an xarray dataset into a NumPy array of shape (84, lat, lon),
        where each height level is treated as a separate variable.
        
        Parameters:
        - data (xr.Dataset): Input dataset with dimensions (time, height, lat, lon) 
                             and variables (t, dpt, u, v).
        
        Returns:
        - np.ndarray: Extracted data of shape (lat * long, 84).
        """
        n_vars = len(data.data_vars) * 21  # Total variables (4 vars * 21 levels)
        lat_size = data.sizes['latitude']
        lon_size = data.sizes['longitude']
            
        data_array = np.empty((n_vars, lat_size, lon_size), dtype=np.float64)
    
        k = 0
        for var_name in data.data_vars:
            data_array[k:k+21, ...] = data[var_name].isel(time=0).values
            k += 21  
        return data_array.reshape(n_vars, -1).T
        
        
    def load_scalar(self,scaler_path):
        """
        Load bridgescaler object.
        Args:
            scalar_path: Path to scalar object.
    
        Returns:
            Loaded bridgescaler object
        """
        scaler = load_scaler(scaler_path)
        groups = scaler.groups_
        input_features = [x for y in groups for x in y]
    
        return  scaler,input_features
        
    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'loss': evidential_cat_loss})
    
    def transform_data(self,input_data, transformer, input_features):
        """
        Transform data for input into ML model.
        Args:
            input_data: Pandas Dataframe of input data
            transformer: Bridgescaler object used to fit data.
    
        Returns:
            Pandas dataframe of transformed input.
        """
        transformed_data = transformer.transform(pd.DataFrame(input_data, columns=input_features))
    
        return transformed_data


    def grid_predictions(self, data, predictions,output_uncertainties=False):
        """
        Populate gridded xarray dataset with ML probabilities and categorical predictions as separate variables.
        Args:
            data: Xarray dataset of input data.
            preds: Pandas Dataframe of ML predictions.
    
        Returns:
            Xarray dataset of ML predictions and surface variables on model grid.
        """
        if output_uncertainties:
    
            probabilities = predictions[0].numpy()
            ptype = probabilities.argmax(axis=1).reshape(-1, 1)
            u = predictions[1].numpy().reshape(data['latitude'].size, data['longitude'].size)
            data['ML_u'] = (['latitude', 'longitude'], u.astype('float64'))
            data[f"ML_u"].attrs = {"Description": f"Evidential Uncertainty (Dempster-Shafer Theory)"}
            ale = predictions[2].numpy().reshape(data['latitude'].size, data['longitude'].size, probabilities.shape[-1])
            epi = predictions[3].numpy().reshape(data['latitude'].size, data['longitude'].size, probabilities.shape[-1])
            for i, (var, v) in enumerate(zip(["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
                                             ["rain", "snow", "icep", "frzr"])):
                for uncertainty_type, long_name, short_name in zip([ale, epi], ["aleatoric", "epistemic"], ["ale", "epi"]):
                    data[f"ML_{v}_{short_name}"] = (['latitude', 'longitude'], uncertainty_type[:, :, i].astype('float64'))
                    data[f"ML_{v}_{short_name}"].attrs = {"Description": f"Machine Learned {long_name}u ncertainty of {var}"}
        else:
            ptype = predictions.argmax(axis=1).reshape(-1, 1)
            probabilities = predictions
    
        preds = np.hstack([probabilities, ptype])
        reshaped_preds = preds.reshape(data['latitude'].size, data['longitude'].size, preds.shape[-1])
        for i, (long_v, v) in enumerate(zip(
                ['rain', 'snow', 'ice pellets', 'freezing rain'], ['rain', 'snow', 'icep', 'frzr'])):
            data[f"ML_{v}"] = (['latitude', 'longitude'], reshaped_preds[:, :, i].astype('float64'))  # ML probability
            data[f"ML_{v}"].attrs = {"Description": f"Machine Learned Probability of {long_v}"}
            data[f"ML_c{v}"] = (['latitude', 'longitude'], np.where(reshaped_preds[:, :, -1] == i, 1, 0).astype('uint8'))  # ML categorical
            data[f"ML_c{v}"].attrs = {"Description": f"Machine Learned Categorical {long_v}"}
    
    
        for var in ["crain", "csnow", "cicep", "cfrzr"]:
            if var in list(data.data_vars):
                data[var] = data[var].astype('uint8')
    
        for v in data.coords:
            if data[v].dtype == 'float32':
                data[v] = data[v].astype('float64')
    
        return data

    def ptype_classification(self, dataset):
        return dataset[self.save_vars].expand_dims({'time': dataset.time.values})
    
    def write_to_netcdf(self,dataset,nc_filename, forecast_hour,conf):
        """Saves the processed data to a NetCDF file."""

        logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)

        unique_filename = os.path.join(
            save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        dataset.to_netcdf(unique_filename)