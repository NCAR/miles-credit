import torch
import xarray as xr

class dataConverter:
    '''
    utility class for converting from various formats to xarray dataset.
    e.g. in train.py, Tensor to Dataset
    e.g. in predict.py DataArray to Dataset
    '''
    def __init__(self, conf) -> None:
        self.conf = conf
        static_ds = xr.open_dataset(self.conf["loss"]["latitude_weights"])
        self.lat = static_ds.latitude.values
        self.lon = static_ds.longitude.values

    def tensor_to_Dataset(self, tensor, forecast_datetimes):
        return self.dataArrays_to_dataset(*self.tensor_to_dataArray(tensor, forecast_datetimes))

    def concat_and_reshape(self, x1, x2): # will be useful for getting back to tensor
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def split_and_reshape(self, tensor):
        # get the number of levels
        levels = self.conf["model"]["levels"]
        # get number of channels
        channels = len(self.conf["data"]["variables"])
        single_level_channels = len(self.conf["data"]["surface_variables"])

        tensor1 = tensor[:, :int(channels * levels), :, :, :]
        tensor2 = tensor[:, -int(single_level_channels):, :, :, :]
        tensor1 = tensor1.view(tensor1.shape[0], channels, levels, tensor1.shape[2], tensor1.shape[3], tensor1.shape[4])
        return tensor1, tensor2
    
    def tensor_to_dataArray(self, pred, forecast_datetimes):
        '''
        pred: Tensor
            w/ shape (B, C, T, lat, lon)
        forecast_datetimes: array-like
        lat: array-like
        lon: array-like
        conf: dictionary
        '''

        # subset upper air and surface variables
        tensor_upper_air, tensor_single_level = self.split_and_reshape(pred)
        tensor_upper_air = tensor_upper_air.squeeze(3)
        tensor_single_level = tensor_single_level.squeeze(2) # take out time dim=1, keep batch dim
        # upper air variables
        darray_upper_air = xr.DataArray(
            tensor_upper_air,
            dims=["datetime", "vars", "level", "latitude", "longitude"],
            coords=dict(
                datetime=forecast_datetimes,
                vars=self.conf["data"]["variables"],
                level=range(self.conf["model"]["levels"]),
                latitude=self.lat,
                longitude=self.lon,
            ),
        )

        # diagnostics and surface variables
        darray_single_level = xr.DataArray(
            tensor_single_level.squeeze(2),
            dims=["datetime", "vars", "latitude", "longitude"],
            coords=dict(
                datetime=forecast_datetimes,
                vars=self.conf["data"]["surface_variables"],
                latitude=self.lat,
                longitude=self.lon,
            ),
        )
                
        # return dataarrays as outputs
        return darray_upper_air, darray_single_level

    def dataArrays_to_dataset(self, darray_upper_air, darray_single_level):
        ds_x = darray_upper_air.to_dataset(dim="vars")
        ds_surf = darray_single_level.to_dataset(dim="vars")
        ds = xr.merge([ds_x, ds_surf])
        
        # dataset as output
        return ds
