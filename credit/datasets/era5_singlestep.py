import numpy as np
from credit.data import (
    ERA5_and_Forcing_Dataset,
    Sample,
    find_key_for_number,
    extract_month_day_hour,
    find_common_indices,
)


class ERA5_and_Forcing_SingleStep(ERA5_and_Forcing_Dataset):

    def __getitem__(self, index):
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.ERA5_indices)

        # get the ind within the current file
        ind_start = self.ERA5_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.all_files[int(ind_file)]["time"]) - (
            self.history_len + self.forecast_len + 1
        )
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        # ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        ERA5_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)
        )  # .load() NOT load into memory

        # merge surface into the dataset

        if self.surface_files:
            # subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )

            # merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(
                surface_subset
            )

        # split ERA5_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static
        # the ind_end of the ERA5_subset
        ind_end_time = len(ERA5_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype("datetime64[s]").astype(int)

        # xarray dataset as input
        # historical_ERA5_images: the final input
        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, self.history_len, self.skip_periods)
        ).load()

        # merge dynamic forcing inputs
        if self.dyn_forcing_files:
            dyn_forcing_subset = self.dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, self.history_len, self.skip_periods)
            ).load()  # <-- load into memory

            historical_ERA5_images = historical_ERA5_images.merge(dyn_forcing_subset)

        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(
                np.array(self.xarray_forcing["time"])
            )
            month_day_inputs = extract_month_day_hour(
                np.array(historical_ERA5_images["time"])
            )  # <-- upper air
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(
                time=ind_forcing
            )
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = historical_ERA5_images["time"]

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(
                dim={"time": N_time_dims}
            )
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords(
                {"time": ERA5_subset["time"]}
            )

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(
                time=slice(0, self.history_len, self.skip_periods)
            )  # .load() # <-- loaded in init

            # update
            static_subset_input["time"] = historical_ERA5_images["time"]

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)

        # xarray dataset as target
        # target_ERA5_images: the final target

        if self.one_shot is not None:
            # one_shot is True (on), go straight to the last element
            target_ERA5_images = ERA5_subset.isel(
                time=slice(-1, None)
            ).load()

            # merge diagnoisc input here:
            if self.diagnostic_files:
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(-1, None)
                ).load()

                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        else:
            # one_shot is None (off), get the full target length based on forecast_len
            target_ERA5_images = ERA5_subset.isel(
                time=slice(self.history_len, ind_end_time, self.skip_periods)
            ).load()  # <-- load into memory

            # merge diagnoisc input here:
            if self.diagnostic_files:
                # subset diagnostic variables
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(self.history_len, ind_end_time, self.skip_periods)
                ).load()

                # merge into the target dataset
                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        # sst forcing operations
        if self.sst_forcing is not None:
            # get xr.dataset keys
            varname_skt = self.sst_forcing["varname_skt"]
            varname_ocean_mask = self.sst_forcing["varname_ocean_mask"]

            # get xr.dataarray from the dataset
            ocean_mask = historical_ERA5_images[varname_ocean_mask]
            input_skt = historical_ERA5_images[varname_skt]
            target_skt = target_ERA5_images[varname_skt]

            # for multi-input cases, use time=-1 ocean mask for all times
            if self.history_len > 1:
                ocean_mask[: self.history_len - 1] = ocean_mask.isel(time=-1)

            # get ocean mask
            ocean_mask_bool = ocean_mask.isel(time=-1) == 0

            # for multi-input cases, use time=-1 ocean SKT for all times
            if self.history_len > 1:
                input_skt[: self.history_len - 1] = input_skt[
                    : self.history_len - 1
                ].where(~ocean_mask_bool, input_skt.isel(time=-1))

            # for target skt, replace ocean values using time=-1 input SKT
            target_skt = target_skt.where(~ocean_mask_bool, input_skt.isel(time=-1))

            # Update the target_ERA5_images dataset with the modified target_skt
            historical_ERA5_images[varname_ocean_mask] = ocean_mask
            historical_ERA5_images[varname_skt] = input_skt
            target_ERA5_images[varname_skt] = target_skt

        # pipe xarray datasets to the sampler
        sample = Sample(
            historical_ERA5_images=historical_ERA5_images,
            target_ERA5_images=target_ERA5_images,
            datetime_index=datetime_as_number,
        )

        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # add datetime for convenice
        sample["datetime"] = [
            int(
                historical_ERA5_images.time.values[0]
                .astype("datetime64[s]")
                .astype(int)
            ),
            int(target_ERA5_images.time.values[0].astype("datetime64[s]").astype(int)),
        ]

        # assign sample index
        sample["index"] = index
        # These are here for consistency with trainers.
        # Forecast_step is always 1
        sample["forecast_step"] = 1
        # Hence stop_forecast is always true
        sample["stop_forecast"] = True
        return sample


if __name__ == "__main__":

    import yaml
    from torch.utils.data import DataLoader
    from credit.transforms import load_transforms
    from credit.parser import credit_main_parser, training_data_check
    from credit.datasets import setup_data_loading, set_globals

    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)

    data_config = setup_data_loading(conf)

    data_config["forecast_len"] = 1

    set_globals(data_config, namespace=globals())

    dataset_multi = ERA5_and_Forcing_SingleStep(
        varname_upper_air=data_config['varname_upper_air'],
        varname_surface=data_config['varname_surface'],
        varname_dyn_forcing=data_config['varname_dyn_forcing'],
        varname_forcing=data_config['varname_forcing'],
        varname_static=data_config['varname_static'],
        varname_diagnostic=data_config['varname_diagnostic'],
        filenames=data_config['all_ERA_files'],
        filename_surface=data_config['surface_files'],
        filename_dyn_forcing=data_config['dyn_forcing_files'],
        filename_forcing=data_config['forcing_files'],
        filename_static=data_config['static_files'],
        filename_diagnostic=data_config['diagnostic_files'],
        history_len=data_config['history_len'],
        forecast_len=data_config['forecast_len'],
        skip_periods=data_config['skip_periods'],
        one_shot=False,
        max_forecast_len=data_config['max_forecast_len'],
        sst_forcing=data_config['sst_forcing'],
        transform=load_transforms(conf)
    )

    dataloader = DataLoader(
        dataset_multi,
        batch_size=2,  # Adjust the batch size as needed
        shuffle=True,   # Shuffle the dataset if needed
        num_workers=4,  # Number of subprocesses to use for data loading (adjust as needed)
        drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
        prefetch_factor=4
    )

    for (k, sample) in enumerate(dataloader):
        print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'], sample["x"].shape, sample["x_surf"].shape)
        if k == 20:
            break
