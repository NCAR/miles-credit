import logging

import torch
import xarray as xr
from geocat.comp.interpolation import interp_hybrid_to_pressure

logger = logging.getLogger(__name__)


class dataConverter:
    """
    utility class for converting from various formats to xarray dataset.
    e.g. in train.py, Tensor to Dataset
    e.g. in predict.py DataArray to Dataset
    """

    def __init__(self, conf, new_levels=None) -> None:
        self.conf = conf
        static_ds = xr.open_dataset(self.conf["loss"]["latitude_weights"])
        self.lat = static_ds.latitude.values
        self.lon = static_ds.longitude.values
        self.SP = static_ds.SP
        self.level_info = xr.open_dataset(
            "/glade/derecho/scratch/dkimpara/nwp_files/hy_to_pressure.nc"
        )
        self.new_levels = new_levels  # levels to interpolate to

    def tensor_to_dataset(self, tensor, forecast_datetimes):
        return self.dataArrays_to_dataset(
            *self.tensor_to_dataArray(tensor, forecast_datetimes)
        )

    def tensor_to_pressure_lev_dataset(self, tensor, forecast_datetimes):
        return self.dataset_to_pressure_levels(
            self.tensor_to_dataset(tensor, forecast_datetimes)
        )

    def concat_and_reshape(self, x1, x2):  # will be useful for getting back to tensor
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def split_and_reshape(self, tensor):
        # get the number of levels
        levels = self.conf["model"]["levels"]
        # get number of channels
        channels = len(self.conf["data"]["variables"])
        single_level_channels = len(self.conf["data"]["surface_variables"])

        tensor1 = tensor[:, : int(channels * levels), :, :, :]
        tensor2 = tensor[:, -int(single_level_channels) :, :, :, :]
        tensor1 = tensor1.view(
            tensor1.shape[0],
            channels,
            levels,
            tensor1.shape[2],
            tensor1.shape[3],
            tensor1.shape[4],
        )
        return tensor1, tensor2

    def tensor_to_dataArray(self, pred, forecast_datetimes):
        """
        Convert tensor to DataArray

        Args:
            pred: Tensor with shape (B, C, T, lat, lon)
            forecast_datetimes: array-like
        """

        # subset upper air and surface variables
        tensor_upper_air, tensor_single_level = self.split_and_reshape(pred)

        tensor_upper_air = tensor_upper_air.squeeze(3)
        tensor_single_level = tensor_single_level.squeeze(
            2
        )  # take out time dim=1, keep batch dim

        # upper air variables
        darray_upper_air = xr.DataArray(
            tensor_upper_air.detach().numpy(),  # could put this in top level, this might be faster
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
            tensor_single_level.detach().numpy(),
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
        # dArrays need to have a time dim
        ds_x = darray_upper_air.to_dataset(dim="vars")
        ds_surf = darray_single_level.to_dataset(dim="vars")
        ds = xr.merge([ds_x, ds_surf])

        # dataset as output
        return ds

    def dataset_to_pressure_levels(self, dataset):
        """
        unless specified in class init,
        interpolation defaults to the pressure levels (in Pa):

        [100000., 92500., 85000., 70000., 50000., 40000.,
         30000., 25000., 20000., 15000., 10000., 7000., 5000.,
         3000., 2000., 1000., 700., 500., 300., 200., 100.],

        """
        dataset = dataset.assign_coords(
            {"level": self.level_info.ref_Pa.values}
        )  # these levels are from high to low
        atmos_dataset = dataset[self.conf["data"]["variables"]]
        SP = self.SP.expand_dims(dim={"datetime": dataset.datetime.values})

        interp_kwargs = {
            "lev_dim": "level",
            "extrapolate": True,
            "variable": "other",
        }  # need to build this kwarg dict because interpolation errors when given None
        if self.new_levels:
            interp_kwargs["new_levels"] = self.new_levels

        # modify atmos slice of dataset
        atmos = atmos_dataset.map(
            interp_hybrid_to_pressure,
            args=[
                SP,
                self.level_info.a_model
                / 100000,  # IFS a_coeffs are in Pa units. geocat computes pressure as hya * p0 + hyb * psfc
                self.level_info.b_model,
            ],  # fix this patch if we ever use coeffs that arent IFS
            **interp_kwargs,
        )

        dataset = xr.merge([atmos, dataset[self.conf["data"]["surface_variables"]]])
        return (
            dataset  # original dataset with interpolated atmos vars and plev coordinate
        )


if __name__ == "__main__":
    from os.path import join
    import yaml

    test_dir = (
        "/glade/work/dkimpara/repos/global/miles-credit/results/test_files_quarter"
    )
    config = join(test_dir, "model.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    y_pred = torch.load(join(test_dir, "pred.pt"))
    # y = torch.load(join(test_dir, "y.pt"))

    converter = dataConverter(conf)
    ds = converter.tensor_to_dataset(y_pred, [0])

    ##### test hybrid to pressure ###
    pressure = converter.dataset_to_pressure_levels(ds)
    print(pressure)
    print(f"nulls: {pressure.U.isnull().sum().values}")
