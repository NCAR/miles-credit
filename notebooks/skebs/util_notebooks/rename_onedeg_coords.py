from credit.data import get_forward_data
from glob import glob
import os

# renaming coordinates in zarr for onedeg data

# all_ERA_files = sorted(glob('/glade/derecho/scratch/wchapman/y_ONEdeg*.zarr'))
# all_ERA_files = sorted(glob('/glade/derecho/scratch/wchapman/SixHourly_y*_ONEdeg*.zarr'))
all_files = sorted(glob('/glade/derecho/scratch/wchapman/credit_solar_1h_1deg/*.nc'))

for file in all_files:
    filename = os.path.basename(file)
    dataset = get_forward_data(file)
    dataset = dataset.rename({'lat': 'latitude',
                              'lon': 'longitude'})
    if filename[-3:] == '.nc' or filename[-4:] == '.nc4':
        dataset.to_netcdf(os.path.join("/glade/derecho/scratch/dkimpara/", filename))
    else:
        dataset.to_zarr(os.path.join("/glade/derecho/scratch/dkimpara/", filename))
    print(f"wrote {filename}")

