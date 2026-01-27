from fastapi import FastAPI
import xarray as xr
import base64
from os.path import exists, join
import xesmf as xe
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


data_path = "/glade/derecho/scratch/dgagne/CREDIT/RAW_OUTPUT/wxformer_1h_gfs_demo/"


gaussian_grid = xr.open_dataset("/glade/work/dgagne/global_gaussian_grid.nc")
latlon_grid = xr.open_dataset("/glade/work/dgagne/global_latlon_grid.nc")
regridder = xe.Regridder(
    gaussian_grid,
    latlon_grid,
    method="conservative",
    weights="/glade/work/dgagne/gaussian_to_latlon_weights.nc",
)


@app.get("/")
async def get_inference_field(
    run_date: str = "2026-01-21T00Z",
    forecast_hour: int = 1,
    variable: str = "Z500",
    level: int = 137,
    height: float = 100.0,
    pressure: float = 500.0,
):
    file_path = join(data_path, run_date, f"pred_{run_date}_{forecast_hour:03d}.nc")
    if not exists(file_path):
        return {
            "status": f"File {file_path} not found.",
            "data": "",
            "dtype": "<f4",
            "shape": [0, 0],
        }

    with xr.open_dataset(file_path) as ds:
        var_dims = ds[variable].dims
        if "level" in var_dims:
            var_data = ds[variable][0].loc[level][::-1]
        elif "pressure" in var_dims:
            var_data = ds[variable][0].loc[pressure][::-1]
        elif "height" in var_dims:
            var_data = ds[variable][0].loc[height][::-1]
        else:
            var_data = ds[variable][0][::-1]
        var_data_uniform = regridder(var_data).values.astype("float32")
    b64_var_data = base64.b64encode(var_data_uniform.tobytes()).decode("utf-8")
    out_dict = {
        "status": "ok",
        "data": b64_var_data,
        "dtype": var_data_uniform.dtype.str,
        "shape": var_data_uniform.shape,
    }
    return out_dict
