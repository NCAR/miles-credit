import unittest
import os
import numpy as np
import xarray as xr
import datetime
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock

from credit.output import (
    load_metadata,
    split_and_reshape,
    make_xarray,
    make_xarray_diag,
    save_netcdf_increment,
    save_netcdf_clean,
    save_netcdf_diag,
)


class TestOutput(unittest.TestCase):
    def setUp(self):
        """Set up common test data and configurations."""
        self.test_dir = tempfile.mkdtemp()
        self.conf = {
            "model": {"levels": 13},
            "data": {
                "variables": ["T", "U"],
                "surface_variables": ["T2M", "SP"],
                "diagnostic_variables": ["MSLP"],
                "level_ids": list(range(1, 14)),
            },
            "predict": {
                "metadata": "credit/metadata/meta.yaml",
                "save_forecast": self.test_dir,
            },
        }
        self.lat = np.arange(90, -90.1, -1)
        self.lon = np.arange(0, 360, 1)
        self.forecast_dt = datetime.datetime(2023, 1, 1, 0, 0)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)

    def test_load_metadata_success(self):
        """Test loading metadata from a YAML file successfully."""
        mock_yaml_content = "T:\n  long_name: Temperature\n  units: K"
        expected_data = {"T": {"long_name": "Temperature", "units": "K"}}

        with patch(
            "builtins.open", mock_open(read_data=mock_yaml_content)
        ) as mock_file:
            with patch("yaml.load", return_value=expected_data) as mock_yaml_load:
                meta_data = load_metadata(self.conf)
                mock_file.assert_called_once_with(self.conf["predict"]["metadata"])
                mock_yaml_load.assert_called_once()
                self.assertEqual(meta_data, expected_data)

    def test_load_metadata_not_given(self):
        """Test behavior when metadata file is not specified in config."""
        conf_no_meta = {"predict": {"metadata": None}}
        with patch("builtins.print") as mock_print:
            meta_data = load_metadata(conf_no_meta)
            self.assertFalse(meta_data)
            mock_print.assert_called_once_with(
                "conf['predict']['metadata'] not given. Skip."
            )

    def test_split_and_reshape(self):
        """Test splitting and reshaping of the model output tensor."""
        levels = self.conf["model"]["levels"]
        ua_vars = len(self.conf["data"]["variables"])
        sfc_vars = len(self.conf["data"]["surface_variables"])
        diag_vars = len(self.conf["data"]["diagnostic_variables"])

        total_ua_channels = ua_vars * levels
        total_sfc_channels = sfc_vars + diag_vars

        # Shape: (batch, channels, lat, lon)
        tensor = np.random.rand(1, total_ua_channels + total_sfc_channels, 2, 3)

        tensor_upper_air, tensor_single_level = split_and_reshape(tensor, self.conf)

        # Expected shape for upper air: (batch, vars, levels, lat, lon)
        self.assertEqual(tensor_upper_air.shape, (1, ua_vars, levels, 2, 3))
        # Expected shape for single level: (batch, vars, lat, lon)
        self.assertEqual(tensor_single_level.shape, (1, total_sfc_channels, 2, 3))

    def test_make_xarray(self):
        """Test creation of xarray DataArrays for upper-air and surface variables."""
        levels = self.conf["model"]["levels"]
        ua_vars = len(self.conf["data"]["variables"])
        sfc_vars = len(self.conf["data"]["surface_variables"])
        diag_vars = len(self.conf["data"]["diagnostic_variables"])

        total_ua_channels = ua_vars * levels
        total_sfc_channels = sfc_vars + diag_vars

        lat_len, lon_len = 2, 3
        pred = np.random.rand(
            1, total_ua_channels + total_sfc_channels, 1, lat_len, lon_len
        )

        darray_upper_air, darray_single_level = make_xarray(
            pred, self.forecast_dt, np.arange(lat_len), np.arange(lon_len), self.conf
        )

        # Check upper air DataArray
        self.assertIsInstance(darray_upper_air, xr.DataArray)
        self.assertEqual(darray_upper_air.shape, (1, ua_vars, levels, lat_len, lon_len))
        self.assertListEqual(
            list(darray_upper_air.dims),
            ["time", "vars", "level", "latitude", "longitude"],
        )
        self.assertListEqual(
            list(darray_upper_air["vars"].values), self.conf["data"]["variables"]
        )

        # Check single level DataArray
        self.assertIsInstance(darray_single_level, xr.DataArray)
        self.assertEqual(
            darray_single_level.shape, (1, total_sfc_channels, lat_len, lon_len)
        )
        self.assertListEqual(
            list(darray_single_level.dims), ["time", "vars", "latitude", "longitude"]
        )
        expected_sfc_vars = (
            self.conf["data"]["surface_variables"]
            + self.conf["data"]["diagnostic_variables"]
        )
        self.assertListEqual(
            list(darray_single_level["vars"].values), expected_sfc_vars
        )

    def test_make_xarray_no_surface(self):
        """Test make_xarray when there are no surface or diagnostic variables."""
        conf_no_sfc = self.conf.copy()
        conf_no_sfc["data"] = {
            "variables": ["T", "U"],
            "surface_variables": [],
            "diagnostic_variables": [],
        }
        conf_no_sfc["model"] = {"levels": 13}

        levels = conf_no_sfc["model"]["levels"]
        ua_vars = len(conf_no_sfc["data"]["variables"])
        total_ua_channels = ua_vars * levels
        lat_len, lon_len = 2, 3
        pred = np.random.rand(1, total_ua_channels, lat_len, lon_len)

        result = make_xarray(
            pred, self.forecast_dt, np.arange(lat_len), np.arange(lon_len), conf_no_sfc
        )

        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.shape, (1, ua_vars, levels, lat_len, lon_len))

    def test_make_xarray_diag(self):
        """Test creation of a diagnostic xarray DataArray."""
        diag_vars = len(self.conf["data"]["diagnostic_variables"])
        lat_len, lon_len = 2, 3
        # Shape: (batch, vars, 1, lat, lon)
        pred = np.random.rand(1, diag_vars, 1, lat_len, lon_len)

        darray_diag = make_xarray_diag(
            pred, self.forecast_dt, np.arange(lat_len), np.arange(lon_len), self.conf
        )

        self.assertIsInstance(darray_diag, xr.DataArray)
        self.assertEqual(darray_diag.shape, (1, diag_vars, lat_len, lon_len))
        self.assertListEqual(
            list(darray_diag.dims), ["time", "vars", "latitude", "longitude"]
        )
        self.assertListEqual(
            list(darray_diag["vars"].values), self.conf["data"]["diagnostic_variables"]
        )

    def _create_test_dataarrays(self):
        """Helper to create dummy DataArrays for saving functions."""
        ua_vars = self.conf["data"]["variables"]
        sfc_vars = (
            self.conf["data"]["surface_variables"]
            + self.conf["data"]["diagnostic_variables"]
        )
        levels = self.conf["model"]["levels"]
        lat_len, lon_len = 4, 5

        darray_upper_air = xr.DataArray(
            np.random.rand(1, len(ua_vars), levels, lat_len, lon_len),
            dims=["time", "vars", "level", "latitude", "longitude"],
            coords={
                "time": [self.forecast_dt],
                "vars": ua_vars,
                "level": np.arange(levels),
                "latitude": np.arange(lat_len),
                "longitude": np.arange(lon_len),
            },
        )

        darray_single_level = xr.DataArray(
            np.random.rand(1, len(sfc_vars), lat_len, lon_len),
            dims=["time", "vars", "latitude", "longitude"],
            coords={
                "time": [self.forecast_dt],
                "vars": sfc_vars,
                "latitude": np.arange(lat_len),
                "longitude": np.arange(lon_len),
            },
        )
        return darray_upper_air, darray_single_level

    @patch("credit.output.full_state_pressure_interpolation")
    def test_save_netcdf_increment_basic(self, mock_interp):
        """Test basic saving functionality of save_netcdf_increment."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 6
        nc_filename = "test_run"

        # Mock interpolation to return an empty dataset
        mock_interp.return_value = xr.Dataset()

        save_netcdf_increment(
            darray_upper_air,
            darray_single_level,
            nc_filename,
            forecast_hour,
            meta_data=False,
            conf=self.conf,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        self.assertTrue(os.path.exists(expected_path))

        with xr.open_dataset(expected_path) as ds:
            self.assertIn("T", ds.data_vars)
            self.assertIn("T2M", ds.data_vars)
            self.assertEqual(ds["forecast_hour"], forecast_hour)

    @patch("credit.output.full_state_pressure_interpolation")
    def test_save_netcdf_increment_with_interp(self, mock_interp):
        """Test save_netcdf_increment with pressure interpolation enabled."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 12
        nc_filename = "test_interp"

        interp_conf = self.conf.copy()
        interp_conf["predict"]["interp_pressure"] = {
            "pres_ending": "_pl",
            "static_fields": "dummy_path",
        }

        # Mock interpolation result
        interp_ds = xr.Dataset({"T_pl": (("level_pl",), np.random.rand(5))})
        mock_interp.return_value = interp_ds

        # Mock opening static fields file
        static_ds = xr.Dataset(
            {"Z_GDS4_SFC": (("latitude", "longitude"), np.random.rand(4, 5))}
        )
        mock_static_ds_context = MagicMock()
        mock_static_ds_context.__enter__.return_value = static_ds
        mock_static_ds_context.__exit__.return_value = None

        with patch("xarray.open_dataset", return_value=mock_static_ds_context):
            save_netcdf_increment(
                darray_upper_air,
                darray_single_level,
                nc_filename,
                forecast_hour,
                meta_data=False,
                conf=interp_conf,
            )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        self.assertTrue(os.path.exists(expected_path))

        with xr.open_dataset(expected_path) as ds:
            self.assertIn("T_pl", ds.data_vars)
            mock_interp.assert_called_once()

    def test_save_netcdf_increment_with_metadata(self):
        """Test save_netcdf_increment applies metadata correctly."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 3
        nc_filename = "test_meta"

        meta_data = {
            "T": {"long_name": "Temperature", "units": "K"},
            "T2M": {"long_name": "2m Temperature", "units": "K"},
            "time": {"calendar": "gregorian"},
        }

        save_netcdf_increment(
            darray_upper_air,
            darray_single_level,
            nc_filename,
            forecast_hour,
            meta_data=meta_data,
            conf=self.conf,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        with xr.open_dataset(expected_path) as ds:
            self.assertEqual(ds["T"].attrs["long_name"], "Temperature")
            self.assertEqual(ds["T2M"].attrs["units"], "K")
            self.assertIn("calendar", ds.time.encoding)

    def test_save_netcdf_increment_drop_vars(self):
        """Test save_netcdf_increment with save_vars to drop variables."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 9
        nc_filename = "test_drop"

        conf_with_save_vars = self.conf.copy()
        conf_with_save_vars["predict"]["save_vars"] = ["T", "MSLP"]

        save_netcdf_increment(
            darray_upper_air,
            darray_single_level,
            nc_filename,
            forecast_hour,
            meta_data=False,
            conf=conf_with_save_vars,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        with xr.open_dataset(expected_path) as ds:
            self.assertIn("T", ds.data_vars)
            self.assertIn("MSLP", ds.data_vars)
            self.assertNotIn("U", ds.data_vars)
            self.assertNotIn("T2M", ds.data_vars)

    def test_save_netcdf_clean_basic(self):
        """Test basic saving with save_netcdf_clean."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 1
        nc_filename = "test_clean"

        save_netcdf_clean(
            darray_upper_air,
            darray_single_level,
            nc_filename,
            forecast_hour,
            meta_data=False,
            conf=self.conf,
            use_logger=False,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        self.assertTrue(os.path.exists(expected_path))
        with xr.open_dataset(expected_path) as ds:
            self.assertIn("T", ds.data_vars)
            self.assertIn("T2M", ds.data_vars)
            self.assertEqual(ds["forecast_hour"], forecast_hour)

    def test_save_netcdf_clean_no_surface(self):
        """Test save_netcdf_clean with only upper-air data."""
        darray_upper_air, _ = self._create_test_dataarrays()
        forecast_hour = 2
        nc_filename = "test_clean_no_sfc"

        save_netcdf_clean(
            darray_upper_air,
            None,
            nc_filename,
            forecast_hour,
            meta_data=False,
            conf=self.conf,
            use_logger=False,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        self.assertTrue(os.path.exists(expected_path))
        with xr.open_dataset(expected_path) as ds:
            self.assertIn("T", ds.data_vars)
            self.assertNotIn("T2M", ds.data_vars)

    def test_save_netcdf_clean_with_metadata(self):
        """Test save_netcdf_clean applies metadata correctly."""
        darray_upper_air, darray_single_level = self._create_test_dataarrays()
        forecast_hour = 4
        nc_filename = "test_clean_meta"
        meta_data = {
            "U": {"long_name": "Zonal Wind", "units": "m/s"},
            "time": {"units": "hours since 2000-01-01"},
        }

        save_netcdf_clean(
            darray_upper_air,
            darray_single_level,
            nc_filename,
            forecast_hour,
            meta_data=meta_data,
            conf=self.conf,
            use_logger=False,
        )

        expected_path = os.path.join(
            self.test_dir, nc_filename, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        with xr.open_dataset(expected_path) as ds:
            self.assertEqual(ds["U"].attrs["long_name"], "Zonal Wind")
            self.assertEqual(ds.time.encoding["units"], "hours since 2000-01-01")

    def test_save_netcdf_diag(self):
        """Test saving diagnostic variables with save_netcdf_diag."""
        _, darray_single_level = self._create_test_dataarrays()
        # Let's imagine darray_single_level only contains diagnostic vars for this test
        darray_diag = darray_single_level.sel(
            vars=self.conf["data"]["diagnostic_variables"]
        )

        forecast_hour = 0
        nc_foldername = "diagnostics"
        nc_filename = "diag_run"

        save_netcdf_diag(
            darray_diag,
            nc_foldername,
            nc_filename,
            forecast_hour,
            meta_data=False,
            conf=self.conf,
        )

        expected_path = os.path.join(
            self.test_dir, nc_foldername, f"pred_{nc_filename}.nc"
        )
        self.assertTrue(os.path.exists(expected_path))
        with xr.open_dataset(expected_path) as ds:
            self.assertIn("MSLP", ds.data_vars)
            self.assertNotIn(
                "T2M", ds.data_vars
            )  # Assuming T2M is not a diagnostic var
            self.assertEqual(ds["forecast_hour"], forecast_hour)


if __name__ == "__main__":
    unittest.main()
