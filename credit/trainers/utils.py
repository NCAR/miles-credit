import os

import torch
from torch.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist

from credit.scheduler import load_scheduler
from credit.samplers import DistributedMultiStepBatchSampler
from credit.datasets.multi_source import MultiSourceDataset
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def inject_flat_var_keys(conf: dict) -> None:
    """Inject Gen1-compatible variable keys into conf["data"] for metrics/loss.

    ``LatWeightedMetrics`` and ``VariableTotalLoss2D`` expect flat lists at:
    - ``conf["data"]["variables"]``
    - ``conf["data"]["surface_variables"]``
    - ``conf["data"]["diagnostic_variables"]``

    These are derived from the nested Gen2 source config.
    """
    if "variables" in conf["data"]:
        return

    source_conf = next(iter(conf["data"]["source"].values()))
    vars_conf = source_conf.get("variables", {})

    prog = vars_conf.get("prognostic") or {}
    diag = vars_conf.get("diagnostic") or {}

    conf["data"]["variables"] = prog.get("vars_3D", [])
    conf["data"]["surface_variables"] = prog.get("vars_2D", [])
    conf["data"]["diagnostic_variables"] = (diag.get("vars_3D", []) + diag.get("vars_2D", [])) if diag else []


def inject_postblock_info(conf: dict) -> None:
    """Inject post-block indices into conf["postblock"] for proper initialization of Gen 1 postblocks."""
    conf["model"].setdefault("post_conf", {"activate": False})

    # set defaults for post modules
    post_list = [
        "skebs",
        "tracer_fixer",
        "global_mass_fixer",
        "global_water_fixer",
        "global_energy_fixer",
        "global_energy_fixer_updown",
    ]

    # if activate is false, set all post modules to false
    if not conf["model"]["post_conf"]["activate"]:
        for post_module in post_list:
            conf["model"]["post_conf"][post_module] = {"activate": False}

    # set defaults for post modules
    for post_module in post_list:
        conf["model"]["post_conf"].setdefault(post_module, {"activate": False})

    # see if any of the postconfs want to be activated
    post_conf = conf["model"]["post_conf"]
    activate_any = any([post_conf[post_module]["activate"] for post_module in post_list])
    if post_conf["activate"] and not activate_any:
        raise ("post_conf is set activate, but no post modules specified")

    if conf["model"]["post_conf"]["activate"]:
        # copy only model configs to post_conf subdictionary
        conf["model"]["post_conf"]["model"] = {k: v for k, v in conf["model"].items() if k != "post_conf"}
        # copy data configs to post_conf (for de-normalize variables)
        conf["model"]["post_conf"]["data"] = {k: v for k, v in conf["data"].items()}
        conf["model"]["post_conf"].setdefault("grid", "legendre-gauss")

        # --------------------------------------------------------------------- #
        # get the full list of input / output variables for post_conf
        # the list is ordered based on the tensor channels inside credit.models

        # upper air vars on all levels
        varname_input = []
        varname_output = []
        for var in conf["data"]["variables"]:
            for i_level in range(conf["data"]["levels"]):
                varname_input.append(var)
                varname_output.append(var)

        varname_input += conf["data"]["surface_variables"]

        # handle the order of input-only variables
        if conf["data"]["static_first"]:
            varname_input += (
                conf["data"]["static_variables"]
                + conf["data"]["dynamic_forcing_variables"]
                + conf["data"]["forcing_variables"]
            )
        else:
            varname_input += (
                conf["data"]["dynamic_forcing_variables"]
                + conf["data"]["forcing_variables"]
                + conf["data"]["static_variables"]
            )

        varname_output += conf["data"]["surface_variables"] + conf["data"]["diagnostic_variables"]

        # # debug only
        conf["model"]["post_conf"]["varname_input"] = varname_input
        conf["model"]["post_conf"]["varname_output"] = varname_output

        # --------------------------------------------------------------------- #

    # SKEBS
    if conf["model"]["post_conf"]["skebs"]["activate"]:
        assert "freeze_base_model_weights" in conf["model"]["post_conf"]["skebs"], (
            "need to specify freeze_base_model_weights in skebs config"
        )

        assert conf["trainer"]["train_batch_size"] == conf["trainer"]["valid_batch_size"], (
            "train and valid batch sizes need to be the same for skebs"
        )

        # setup backscatter writing
        conf["model"]["post_conf"]["predict"] = {k: v for k, v in conf["predict"].items()}

        conf["model"]["post_conf"]["skebs"].setdefault("lmax", None)
        conf["model"]["post_conf"]["skebs"].setdefault("mmax", None)

        if conf["model"]["post_conf"]["skebs"]["lmax"] in ["none", "None"]:
            conf["model"]["post_conf"]["skebs"]["lmax"] = None
        if conf["model"]["post_conf"]["skebs"]["mmax"] in ["none", "None"]:
            conf["model"]["post_conf"]["skebs"]["mmax"] = None

        U_inds = [i_var for i_var, var in enumerate(varname_output) if var == "U"]

        V_inds = [i_var for i_var, var in enumerate(varname_output) if var == "V"]
        T_inds = [i_var for i_var, var in enumerate(varname_output) if var == "T"]
        Q_inds = [i_var for i_var, var in enumerate(varname_output) if var in ["Q", "Qtot"]]

        conf["model"]["post_conf"]["skebs"]["U_inds"] = U_inds
        conf["model"]["post_conf"]["skebs"]["V_inds"] = V_inds
        conf["model"]["post_conf"]["skebs"]["Q_inds"] = Q_inds
        conf["model"]["post_conf"]["skebs"]["T_inds"] = T_inds

        if "SP" in varname_output:
            conf["model"]["post_conf"]["skebs"]["SP_ind"] = varname_output.index("SP")
        else:
            conf["model"]["post_conf"]["skebs"]["SP_ind"] = varname_output.index("PS")

        static_inds = [i_var for i_var, var in enumerate(varname_input) if var in conf["data"]["static_variables"]]
        conf["model"]["post_conf"]["skebs"]["static_inds"] = static_inds

        ###### debug mode setup #######
        conf["model"]["post_conf"]["skebs"]["save_loc"] = conf["save_loc"]

    # --------------------------------------------------------------------- #
    # tracer fixer
    flag_tracer = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["tracer_fixer"]["activate"]

    if flag_tracer:
        # when tracer fixer is on, get tensor indices of tracers
        # tracers must be outputs (either prognostic or output only)

        # tracer fixer runs on de-normalized variables by default
        conf["model"]["post_conf"]["tracer_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["tracer_fixer"].setdefault("tracer_thres", [])

        varname_tracers = conf["model"]["post_conf"]["tracer_fixer"]["tracer_name"]
        tracers_thres_input = conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"]
        tracers_thres_maximum = conf["model"]["post_conf"]["tracer_fixer"].get("tracer_thres_max", None)

        # create a mapping from tracer variable names to their thresholds
        tracer_threshold_dict = dict(zip(varname_tracers, tracers_thres_input))
        if tracers_thres_maximum is not None:
            tracer_threshold_dict_max = dict(zip(varname_tracers, tracers_thres_maximum))

        # Iterate over varname_output to find tracer indices and thresholds
        tracer_inds = []
        tracer_thres = []
        tracer_thres_max = []
        for i_var, var in enumerate(varname_output):
            if var in tracer_threshold_dict:
                tracer_inds.append(i_var)
                tracer_thres.append(float(tracer_threshold_dict[var]))
                if tracers_thres_maximum is not None:
                    tracer_thres_max.append(float(tracer_threshold_dict_max[var]))

        conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = tracer_inds
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = tracer_thres
        if tracers_thres_maximum is not None:
            conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres_max"] = tracer_thres_max

    # --------------------------------------------------------------------- #
    # global mass fixer

    flag_mass = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_mass_fixer"]["activate"]

    if flag_mass:
        # when global mass fixer is on, get tensor indices of q, precip, evapor
        # these variables must be outputs

        # global mass fixer defaults
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("midpoint", False)
        conf["model"]["post_conf"]["global_mass_fixer"].setdefault("grid_type", "pressure")

        assert "fix_level_num" in conf["model"]["post_conf"]["global_mass_fixer"], (
            "Must specifiy what level to fix on specific total water"
        )

        if conf["model"]["post_conf"]["global_mass_fixer"]["simple_demo"] is False:
            assert "lon_lat_level_name" in conf["model"]["post_conf"]["global_mass_fixer"], (
                "Must specifiy var names for lat/lon/level in physics reference file"
            )

        if conf["model"]["post_conf"]["global_mass_fixer"]["grid_type"] == "sigma":
            assert "surface_pressure_name" in conf["model"]["post_conf"]["global_mass_fixer"], (
                "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"
            )

        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_mass_fixer"]["specific_total_water_name"]
        ]
        conf["model"]["post_conf"]["global_mass_fixer"]["q_inds"] = q_inds

        if conf["model"]["post_conf"]["global_mass_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_mass_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_mass_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------------------- #
    # global water fixer
    flag_water = conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_water_fixer"]["activate"]

    if flag_water:
        # when global water fixer is on, get tensor indices of q, precip, evapor
        # these variables must be outputs

        # global water fixer defaults
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_water_fixer"].setdefault("midpoint", False)

        conf["model"]["post_conf"]["global_water_fixer"].setdefault("grid_type", "pressure")

        if conf["model"]["post_conf"]["global_water_fixer"]["simple_demo"] is False:
            assert "lon_lat_level_name" in conf["model"]["post_conf"]["global_water_fixer"], (
                "Must specifiy var names for lat/lon/level in physics reference file"
            )

        if conf["model"]["post_conf"]["global_water_fixer"]["grid_type"] == "sigma":
            assert "surface_pressure_name" in conf["model"]["post_conf"]["global_water_fixer"], (
                "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"
            )
        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["specific_total_water_name"]
        ]

        precip_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["precipitation_name"]
        ]

        evapor_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_water_fixer"]["evaporation_name"]
        ]

        conf["model"]["post_conf"]["global_water_fixer"]["q_inds"] = q_inds
        conf["model"]["post_conf"]["global_water_fixer"]["precip_ind"] = precip_inds[0]
        conf["model"]["post_conf"]["global_water_fixer"]["evapor_ind"] = evapor_inds[0]

        if conf["model"]["post_conf"]["global_water_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_water_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_water_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------------------- #
    # global energy fixer
    flag_energy = (
        conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_energy_fixer"]["activate"]
    )

    if flag_energy:
        # when global energy fixer is on, get tensor indices of energy components
        # geopotential at surface is input, others are outputs

        # global energy fixer defaults
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("activate_outside_model", False)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("denorm", True)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("simple_demo", False)
        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("midpoint", False)

        conf["model"]["post_conf"]["global_energy_fixer"].setdefault("grid_type", "pressure")

        if conf["model"]["post_conf"]["global_energy_fixer"]["simple_demo"] is False:
            assert "lon_lat_level_name" in conf["model"]["post_conf"]["global_energy_fixer"], (
                "Must specifiy var names for lat/lon/level in physics reference file"
            )

        if conf["model"]["post_conf"]["global_energy_fixer"]["grid_type"] == "sigma":
            assert "surface_pressure_name" in conf["model"]["post_conf"]["global_energy_fixer"], (
                "Must specifiy surface pressure var name when using hybrid sigma-pressure coordinates"
            )

        T_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["air_temperature_name"]
        ]

        q_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["specific_total_water_name"]
        ]

        U_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["u_wind_name"]
        ]

        V_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["v_wind_name"]
        ]

        TOA_rad_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["TOA_net_radiation_flux_name"]
        ]

        surf_rad_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_net_radiation_flux_name"]
        ]

        surf_flux_inds = [
            i_var
            for i_var, var in enumerate(varname_output)
            if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_energy_flux_name"]
        ]

        conf["model"]["post_conf"]["global_energy_fixer"]["T_inds"] = T_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["q_inds"] = q_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["U_inds"] = U_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["V_inds"] = V_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["TOA_rad_inds"] = TOA_rad_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["surf_rad_inds"] = surf_rad_inds
        conf["model"]["post_conf"]["global_energy_fixer"]["surf_flux_inds"] = surf_flux_inds

        if conf["model"]["post_conf"]["global_energy_fixer"]["grid_type"] == "sigma":
            sp_inds = [
                i_var
                for i_var, var in enumerate(varname_output)
                if var in conf["model"]["post_conf"]["global_energy_fixer"]["surface_pressure_name"]
            ]
            conf["model"]["post_conf"]["global_energy_fixer"]["sp_inds"] = sp_inds[0]

    # --------------------------------------------------------------------- #
    # global energy fixer (up/down flux version)
    flag_energy_updown = (
        conf["model"]["post_conf"]["activate"] and conf["model"]["post_conf"]["global_energy_fixer_updown"]["activate"]
    )

    if flag_energy_updown:
        cfg_ud = conf["model"]["post_conf"]["global_energy_fixer_updown"]

        cfg_ud.setdefault("activate_outside_model", False)
        cfg_ud.setdefault("denorm", True)
        cfg_ud.setdefault("simple_demo", False)
        cfg_ud.setdefault("midpoint", True)
        cfg_ud.setdefault("grid_type", "sigma")

        if cfg_ud["simple_demo"] is False:
            assert "lon_lat_level_name" in cfg_ud, "Must specify var names for lat/lon/level in physics reference file"

        if cfg_ud["grid_type"] == "sigma":
            assert "surface_pressure_name" in cfg_ud, (
                "Must specify surface pressure var name when using hybrid sigma-pressure coordinates"
            )

        def _find_ind(name_key, single=False):
            inds = [i for i, v in enumerate(varname_output) if v in cfg_ud[name_key]]
            return inds[0] if single else inds

        cfg_ud["T_inds"] = _find_ind("air_temperature_name")
        cfg_ud["q_inds"] = _find_ind("specific_total_water_name")
        cfg_ud["U_inds"] = _find_ind("u_wind_name")
        cfg_ud["V_inds"] = _find_ind("v_wind_name")
        cfg_ud["TOA_down_solar_ind"] = _find_ind("TOA_down_solar_name", single=True)
        cfg_ud["TOA_up_solar_ind"] = _find_ind("TOA_up_solar_name", single=True)
        cfg_ud["TOA_up_OLR_ind"] = _find_ind("TOA_up_OLR_name", single=True)
        cfg_ud["surf_down_solar_ind"] = _find_ind("surf_down_solar_name", single=True)
        cfg_ud["surf_up_solar_ind"] = _find_ind("surf_up_solar_name", single=True)
        cfg_ud["surf_down_LW_ind"] = _find_ind("surf_down_LW_name", single=True)
        cfg_ud["surf_up_LW_ind"] = _find_ind("surf_up_LW_name", single=True)
        cfg_ud["surf_SH_ind"] = _find_ind("surf_SH_name", single=True)
        cfg_ud["surf_LH_ind"] = _find_ind("surf_LH_name", single=True)

        if cfg_ud["grid_type"] == "sigma":
            cfg_ud["sp_inds"] = _find_ind("surface_pressure_name", single=True)


def load_dataset(conf: dict, is_train: bool) -> MultiSourceDataset:
    """Build a MultiSourceDataset for train or validation."""
    if is_train:
        data_conf = conf["data"]
    else:
        data_conf = {**conf["data"], **conf.get("validation_data", {})}
        data_conf["source"] = conf["data"]["source"]

    return MultiSourceDataset(data_conf, return_target=True)


def load_dataloader(
    conf: dict,
    dataset: MultiSourceDataset,
    rank: int,
    world_size: int,
    is_train: bool,
) -> DataLoader:
    """Build a DataLoader with DistributedMultiStepBatchSampler."""
    if is_train:
        batch_size = conf["trainer"]["train_batch_size"]
        shuffle = True
        seed = conf.get("seed", 42) + rank
    else:
        batch_size = conf["trainer"]["valid_batch_size"]
        shuffle = False
        seed = conf.get("seed", 42)

    forecast_len = conf["data"]["forecast_len"]
    num_workers = conf["trainer"].get("thread_workers" if is_train else "valid_thread_workers", 4)
    prefetch_factor = conf["trainer"].get("prefetch_factor", 2) if num_workers > 0 else None

    sampler = DistributedMultiStepBatchSampler(
        dataset,
        batch_size=batch_size,
        num_forecast_steps=forecast_len,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )


def load_model_states_and_optimizer(conf, model, device):
    """Load model weights, optimizer, scheduler, and gradient scaler."""
    conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

    learning_rate = float(conf["trainer"]["learning_rate"])
    weight_decay = float(conf["trainer"]["weight_decay"])
    amp = conf["trainer"]["amp"]

    load_weights = conf["trainer"].get("load_weights", False)
    load_optimizer_conf = conf["trainer"].get("load_optimizer", False)
    load_scaler_conf = conf["trainer"].get("load_scaler", False)
    load_scheduler_conf = conf["trainer"].get("load_scheduler", False)

    def _make_optimizer(model):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            opt = FSDPOptimizerWrapper(opt, model)
        return opt

    def _make_scaler():
        return ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    if not load_weights:
        optimizer = _make_optimizer(model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

    elif not (load_optimizer_conf or load_scaler_conf or load_scheduler_conf):
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = _make_optimizer(model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
        else:
            ckpt = os.path.join(save_loc, "checkpoint.pt")
            checkpoint = torch.load(ckpt, map_location=device)
            if conf["trainer"]["mode"] == "ddp":
                load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            load_state_dict_error_handler(load_msg)
            optimizer = _make_optimizer(model)

        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

        if conf["trainer"].get("reload_epoch", False) and os.path.exists(os.path.join(save_loc, "training_log.csv")):
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        if conf["trainer"]["mode"] == "fsdp":
            optimizer = _make_optimizer(model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if load_optimizer_conf:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))
        else:
            if conf["trainer"]["mode"] == "ddp":
                load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            load_state_dict_error_handler(load_msg)
            optimizer = _make_optimizer(model)
            if load_optimizer_conf:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

        if conf["trainer"].get("reload_epoch", False):
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

        if conf["trainer"]["start_epoch"] > 0:
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if conf["trainer"].get("update_learning_rate", False):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler
