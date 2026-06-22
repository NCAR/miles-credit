"""
SKEBS                 — full backscatter scheme; owns a SpectralNoisePattern
                        and a backscatter neural network.
"""

import logging
import os
from os.path import join
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.cuda.amp import custom_fwd
import segmentation_models_pytorch as smp

import xarray as xr

# project-local imports (unchanged from original)
logger = logging.getLogger(__name__)

from sht_noise import SpectralNoisePattern



# ---------------------------------------------------------------------------
# Full SKEBS scheme
# ---------------------------------------------------------------------------

class SKEBS(nn.Module):
    """
    Stochastic Kinetic-Energy Backscatter Scheme.

    post_conf: configuration dictionary (see config examples).

    The scheme perturbs the streamfunction of horizontal wind (U, V) with a
    nondivergent perturbation drawn from a red-noise spectral pattern.  The
    amplitude of the perturbation is predicted by a neural network
    (``backscatter_network``).  The ``SpectralNoisePattern`` module handles
    everything related to spectral noise generation.

    References:
        Berner et al. (2009). J. Atmos. Sci., 66(3), 603-626.
    """

    def __init__(self, post_conf: dict,):
        super().__init__()

        self.post_conf   = post_conf
        self.retain_graph = post_conf["data"].get("retain_graph", False)

        # Grid / dimension metadata
        self.nlon             = post_conf["model"]["image_width"]
        self.nlat             = post_conf["model"]["image_height"]
        self.channels         = post_conf["model"]["channels"]
        self.levels           = post_conf["model"]["levels"]
        self.surface_channels = post_conf["model"]["surface_channels"]
        self.output_only_channels = post_conf["model"]["output_only_channels"]
        self.input_only_channels  = post_conf["model"]["input_only_channels"]
        self.frames           = post_conf["model"]["frames"]

        self.forecast_len       = post_conf["data"]["forecast_len"] + 1
        self.valid_forecast_len = post_conf["data"]["valid_forecast_len"] + 1
        self.multistep          = self.forecast_len > 1

        self.U_var = "ERA5/prognostic/3d/U"
        self.V_var = "ERA5/prognostic/3d/V"
        self.T_var = "ERA5/prognostic/3d/T"
        self.Q_var = "ERA5/prognostic/3d/Qtot"

        # Cosine-latitude weight (expanded to device / ensemble size on first forward pass)
        # cos_lat = np.cos(
        #     np.deg2rad(xr.open_dataset(post_conf["data"]["save_loc_static"])["latitude"])
        # ).values
        # self.cos_lat = (
        #     torch.tensor(cos_lat)
        #     .reshape(1, 1, 1, cos_lat.shape[0], 1)
        #     .expand(1, 1, 1, cos_lat.shape[0], 288)
        # )

        self.eps = 1e-12

        # ------------------------------------------------------------------
        # Spectral noise pattern (reusable component)
        # ------------------------------------------------------------------
        alpha_init                   = post_conf["skebs"].get("alpha_init", 0.125)
        freeze_pattern_weights       = post_conf["skebs"].get("freeze_pattern_weights", False)
        max_pattern_wavenum          = post_conf["skebs"].get("max_pattern_wavenum", 60)
        pattern_filter_anneal_start  = post_conf["skebs"].get("pattern_filter_anneal_start", 40)

        # Ansatz vs standard perturbation
        ansatz_perturb = post_conf["skebs"].get("ansatz_perturb", False)
        if ansatz_perturb:
            self.apply_perturb = self.apply_ansatz
            logger.warning("using Ansatz Perturb")
        else:
            self.apply_perturb = self.apply_skebs

        self.noise_pattern = SpectralNoisePattern(
            nlat=self.nlat,
            nlon=self.nlon,
            multistep=self.multistep,
            alpha_init=alpha_init,
            max_pattern_wavenum=max_pattern_wavenum,
            pattern_filter_anneal_start=pattern_filter_anneal_start,
            freeze_pattern_weights=freeze_pattern_weights,
            mode="scalar" if ansatz_perturb else "streamfunction"
        )

        # Expose lmax / mmax from the pattern for convenience
        self.lmax = self.noise_pattern.lmax
        self.mmax = self.noise_pattern.mmax

        # Pattern initialisation flag (controls re-initialisation between train steps)
        self.spec_coef_is_initialized = False

        # ------------------------------------------------------------------
        # Backscatter  filters spectral and top of model
        # ------------------------------------------------------------------
        zero_out_levels = post_conf["skebs"].get("zero_out_levels_top_of_model", 3)
        self.register_buffer('backscatter_filter', # filter to zero out backscatter at top of model
                        torch.cat([
                            torch.zeros(zero_out_levels),
                            torch.ones(self.levels - zero_out_levels)
                        ]).view(1, self.levels, 1, 1, 1),
                        persistent=False) 

        b_max    = post_conf["skebs"].get("max_backscatter_wavenum", 100)
        b_anneal = post_conf["skebs"].get("spectral_backscatter_filter_anneal_start", 90)
        spectral_backscatter_filter = torch.cat([
            torch.ones(b_anneal),
            torch.linspace(1.0, 0.1, b_max - b_anneal),
            torch.zeros(self.lmax - b_max),
        ]).view(1, 1, 1, self.lmax, 1)
        self.spectral_backscatter_filter = Parameter(spectral_backscatter_filter, requires_grad=False)

        # ------------------------------------------------------------------
        # Backscatter neural network and scaling coefs
        # ------------------------------------------------------------------
        self.relu1 = nn.ReLU()
        self.r = Parameter(torch.tensor(0.02, requires_grad=False)) # see berner 2009, section 4a
        self.r.requires_grad = False

        self.dissipation_type  = post_conf["skebs"]["dissipation_type"]
        self.filter_backscatter = (
            post_conf["skebs"].get("filter_backscatter", True)
            and self.dissipation_type not in ["prescribed", "uniform"]
        )
        logger.info(f"backscatter filter: {self.filter_backscatter}")

        num_channels = self._compute_num_channels()
        self.backscatter_network = self._build_backscatter_network(post_conf, num_channels)
        logger.info(f"using dissipation type: {self.dissipation_type}")

        # ------------------------------------------------------------------
        # Training / freezing controls
        # ------------------------------------------------------------------
        if post_conf["skebs"].get("freeze_dissipation_weights", False):
            logger.warning("freezing all dissipation predictor weights")
            for param in self.backscatter_network.parameters():
                param.requires_grad = False
    
        if not post_conf["skebs"].get("freeze_noise", True):
            logger.warning("freezing all noise parameters")
            for param in self.noise_pattern.parameters():
                param.requires_grad = False

        # multistep turns this on by default unless frozen by the above
        if post_conf["skebs"].get("freeze_alpha", True):
            self.noise_pattern.alpha.requires_grad = False
            logger.info("training alpha")

        # these are off by default
        if post_conf["skebs"].get("train_pattern_filter", False):
            self.noise_pattern.spectral_pattern_filter.requires_grad = True
            logger.info("training pattern filter")

        if post_conf["skebs"].get("train_backscatter_filter", False):
            self.spectral_backscatter_filter.requires_grad = True
            logger.info("training backscatter filter")

        logger.info(
            f"trainable params: {[name for name, p in self.named_parameters() if p.requires_grad]}"
        )


        ### initialization flags
        self.is_training  = False
        self.iteration    = 0

        # ------------------------------------------------------------------
        # Debugging / analysis
        # ------------------------------------------------------------------


        # self.write_rollout_debug_files = post_conf["skebs"].get("write_rollout_debug_files", True)
        # self.write_train_debug_files   = post_conf["skebs"].get("write_train_debug_files", False)
        # self.write_every               = post_conf["skebs"].get("write_train_every", 999)

        # save_loc = post_conf["skebs"]["save_loc"]
        # self.debug_save_loc = join(save_loc, suffix)

        # if self.write_train_debug_files or self.write_rollout_debug_files:
        #     os.makedirs(self.debug_save_loc, exist_ok=True)
        #     logger.info("writing SKEBS debugging files")

        # # Early stop
        # self.iteration_stop = post_conf["skebs"].get("iteration_stop", 0)
        # if self.iteration_stop:
        #     logger.info(f"SKEBS is STOPPING at iteration {self.iteration_stop}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_num_channels(self) -> int:
        """Derive the number of input channels for the backscatter network."""
        post_conf = self.post_conf
        num = (
            self.channels * self.levels
            + post_conf["model"]["surface_channels"]
            + post_conf["model"]["output_only_channels"]
        )
        if post_conf["tracer_fixer"].get("activate", False):
            num -= 1
        # if self.use_statics:
        #     num += len(self.static_inds) + 1  # +1 for cos_lat
        return num

    def _build_backscatter_network(self, post_conf: dict, num_channels: int) -> nn.Module:
        """Instantiate the appropriate backscatter network from config."""
        dtype = self.dissipation_type
        skebs_conf = post_conf["skebs"]

        if dtype == "prescribed":
            return BackscatterPrescribed(
                self.nlat, self.nlon, self.levels,
                post_conf["data"]["std_path"], skebs_conf["sigma_max"],
            )
        if dtype == "uniform_by_latitude":
            return BackscatterPrescribed(
                self.nlat, self.nlon, self.levels,
                post_conf["data"]["std_path"], skebs_conf["sigma_max"],
                vary_by_latitude=True,
            )
        if dtype == "uniform":
            return BackscatterFixedCol(self.levels)
        if dtype == "FCNN":
            return BackscatterFCNN(num_channels, self.levels)
        if dtype == "FCNN_wide":
            return BackscatterFCNNWide(num_channels, self.levels)
        if dtype == "CNN":
            return BackscatterCNN(num_channels, self.levels, self.nlat, self.nlon)
        # if dtype == "unet":
        #     return BackscatterUnet(
        #         num_channels, self.levels, self.nlat, self.nlon,
        #         skebs_conf.get("architecture", None),
        #         skebs_conf.get("padding", 48),
        #     )
        raise RuntimeError(
            f"{dtype} is not a valid dissipation type, please modify config"
        )

    # ------------------------------------------------------------------
    # Perturbation strategies
    # ------------------------------------------------------------------

    def apply_skebs(
        self,
        pred_dict: dict,
        backscatter_filtered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard non-divergent (u, v) streamfunction perturbation."""
        u_chi, v_chi = self.noise_pattern(not self.retain_graph)
        
        backscatter_term = torch.sqrt(self.r * backscatter_filtered)

        u_perturb = backscatter_term * u_chi
        v_perturb = backscatter_term * v_chi

        U = pred_dict[self.U_var] + u_perturb
        V = pred_dict[self.V_var] + v_perturb

        return U, V

    def apply_ansatz(
        self,
        pred_dict: dict,
        backscatter_filtered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ansatz perturbation: scale the noise by the local wind magnitude.
        Note: this is the 'wrong' SKEBS perturbation retained for experimentation.
        """
        noise = self.noise_pattern(not self.retain_graph) * self.r * torch.sqrt(backscatter_filtered)

        U = pred_dict[self.U_var]
        V = pred_dict[self.V_var]
        wind_magnitude = torch.sqrt(U**2 + V**2)

        return U / wind_magnitude * noise, V / wind_magnitude * noise

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input_dict: dict) -> dict:
        """
        Apply the SKEBS perturbation to ``input_dict["y_pred"]`` that is already scaled

        Args:
            x: dict with keys ``"x"`` (previous timestep), ``"y_pred"``
               (current model output), and ``"forecast_step"``.

        Returns:
            The same dict with ``"y_pred"`` replaced by the perturbed field.

        Note:
            The inverse SHT requires float32 or higher precision.
        """
        # ------------------------------------------------------------------
        # setup inputs
        # ------------------------------------------------------------------
        pred_dict  = input_dict["prediction"]
        pred_array = input_dict["y_pred"] 

        if not self.retain_graph:
            pred_array = pred_array.detach()
            pred_dict = {k: v.detach() for k,v in pred_dict.items()}

        # ------------------------------------------------------------------
        # manual overrides of skebs parameters for tuning purposes
        # ------------------------------------------------------------------
        if self.iteration == 0:
            if "override_r" in self.post_conf["skebs"]:
                self.noise_pattern.r.data = torch.tensor(self.post_conf["skebs"]["override_r"])
                logger.warning(f"manually setting r to {self.noise_pattern.r}")
            if "override_dE" in self.post_conf["skebs"]:
                self.noise_pattern.dE.data = torch.tensor(float(self.post_conf["skebs"]["override_dE"]))
                logger.warning(f"manually setting dE to {self.noise_pattern.dE}")

        # ------------------------------------------------------------------
        # initialize pattern
        # ------------------------------------------------------------------
        if self.iteration == 0:
            self.noise_pattern.initialize_pattern(pred_array)
            # self.cos_lat = self.cos_lat.to(x.device).expand(x.shape[0], *self.cos_lat.shape[1:])

        # TODO statics?
        # Optionally concatenate static fields and cos_lat for backscatter input
        # x_for_backscatter = x
        # if self.use_statics:
        #     x_input_statics = x_input[:, self.static_inds]
        #     x = torch.cat([x, x_input_statics, self.cos_lat], dim=1)


        # ------------------------------------------------------------------
        # Backscatter prediction
        # ------------------------------------------------------------------
        backscatter_pred = self.backscatter_filter * self.backscatter_network(pred_array)
        # filter
        if self.filter_backscatter:
            self.spectral_backscatter_filter.data = self.spectral_backscatter_filter.data.clamp(0., 1.)
            backscatter_spec = self.noise_pattern.sht(backscatter_pred)
            backscatter_spec = self.spectral_backscatter_filter * backscatter_spec
            backscatter_filtered = self.noise_pattern.isht(backscatter_spec)
        else:
            backscatter_filtered = backscatter_pred
        # clip
        backscatter_filtered = self.relu1(backscatter_filtered)

        # ------------------------------------------------------------------
        # Apply perturbation (ansatz or skebs)
        # ------------------------------------------------------------------

        u_perturbed, v_perturbed = self.apply_perturb(pred_dict, backscatter_filtered)
        pred_dict[self.U_var] = u_perturbed + 1
        pred_dict[self.V_var] = v_perturbed + 1

        # ------------------------------------------------------------------
        # book keeping
        # ------------------------------------------------------------------
        assert not torch.isnan(u_perturbed).any(), "NaN detected in SKEBS U output"
        assert not torch.isnan(v_perturbed).any(), "NaN detected in SKEBS V output"
        self.iteration += 1
    
        # ------------------------------------------------------------------
        # write debug files
        # ------------------------------------------------------------------
        # logger.debug(f"max backscatter: {torch.max(torch.abs(backscatter_filtered))}")

        # if self.write_train_debug_files and self.iteration % self.write_every == 0:
        #     torch.save(
        #         self.noise_pattern.spectral_pattern_filter,
        #         join(self.debug_save_loc, f"spectral_filter_{self.iteration}"),
        #     )
        #     torch.save(
        #         self.spectral_backscatter_filter,
        #         join(self.debug_save_loc, f"spectral_backscatter_filter_{self.iteration}"),
        #     )

        # if self._should_write_debug():
        #     logger.info(f"writing raw backscatter file for iter {self.iteration}")
        #     torch.save(backscatter_pred, join(self.debug_save_loc, f"backscatter_raw_{self.iteration}"))
        # if self._should_write_debug():
        #     logger.info(f"writing filtered backscatter file for iter {self.iteration}")
        #     torch.save(backscatter_filtered, join(self.debug_save_loc, f"backscatter_{self.iteration}"))

        input_dict = input_dict | {"prediction": pred_dict}
        return input_dict

    # ------------------------------------------------------------------
    # Tiny helpers
    # ------------------------------------------------------------------

    def _should_write_debug(self) -> bool:
        return (
            (self.write_rollout_debug_files and not self.is_training)
            or (self.write_train_debug_files and self.iteration % self.write_every == 0)
        )
    
    
class BackscatterFCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 levels):
        """
        A small two layer full connected neural net (multilayer perceptron) to predict
        the backscatter rate
        """
        # could also predict with x_prev and y
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 2, self.levels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # put channels last

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)

        x = x.permute(0, -1, 1, 2, 3) # put channels back to 1st dim

        return x

class BackscatterFCNNWide(nn.Module):
    def __init__(self,
                 in_channels,
                 levels):
        """
        A wide four layer full connected neural net (multilayer perceptron) to predict
        the backscatter rate
        """
        # could also predict with x_prev and y
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels

        self.fc1 = nn.Linear(in_channels, in_channels * 2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_channels * 2, in_channels * 4)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_channels * 4, in_channels * 2)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(in_channels * 2, levels)
        self.relu4 = nn.ReLU()


    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # put channels last

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = torch.clamp(x, max=1000.)

        x = x.permute(0, -1, 1, 2, 3) # put channels back to 1st dim
        return x

class BackscatterCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 levels,
                 nlat,
                 nlon):
        """
        A small 3x3 convolutional layer to predict
        the backscatter rate.
        Padding 1 on each edge of the input array
        """
        # could also predict with x_prev and y
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.in_channels = in_channels
        self.levels = levels

        # setup padding functions
        self.pad_lon = torch.nn.CircularPad2d((1,1,0,0))
        self.pad_lat = torch.nn.ReplicationPad2d((0,0,1,1))

        # setup conv layer
        self.conv = torch.nn.Conv2d(self.in_channels, self.levels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()
        

    def pad(self, x):
        x = self.pad_lat(x) #reflection padding
        x[..., [0,-1], :] = torch.roll(x[..., [0,-1], :], self.nlon // 2, -1) # shift reflection by 180
        x = self.pad_lon(x) #padding across lon
        return x
    
    def unpad(self, x):
        return x[..., 1:-1, 1:-1]

    def forward(self, x):
        x = x.squeeze(2) # squeeze out time dim (see above)
        x = self.pad(x)
        # (b,c,lat+2,lon+2)
        x = self.conv(x) # should take out the pad
        x = self.sigmoid(x)

        x = x.unsqueeze(2)
        return x

supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
}

def load_premade_encoder_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}"
        )

# class BackscatterUnet(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  levels,
#                  nlat,
#                  nlon,
#                  architecture,
#                  padding):
#         """
#         configurable unet to predict the backscatter rate. architectures and weights can be loaded
#         """
#         # could also predict with x_prev and y
#         super().__init__()
#         self.nlat = nlat
#         self.nlon = nlon
#         self.in_channels = in_channels
#         self.levels = levels
        
#         # setup padding functions
#         self.pad = padding
#         if self.pad:
#             logger.info(f"padding size {self.pad} inside unet")
#             self.boundary_padding = TensorPadding(pad_lat=(self.pad, self.pad),
#                                                 pad_lon=(self.pad, self.pad))

#         self.relu = nn.ReLU()

#         if architecture is None:
#             architecture = {
#                             "name": "unet++",
#                             "encoder_name": "resnet34",
#                             "encoder_weights": "imagenet",
#                         }
#         if architecture["name"] == "unet":
#             architecture["decoder_attention_type"] = "scse"

#         architecture["in_channels"] = in_channels
#         architecture["classes"] = levels

        # self.model = load_premade_encoder_model(architecture)
    
    # def forward(self, x):
    #     x = x.squeeze(2) # squeeze out time dim (see above)

    #     if self.pad:
    #         x = self.boundary_padding.pad(x)
        
    #     x = self.model(x)

    #     if self.pad:
    #         x = self.boundary_padding.unpad(x)

    #     x = self.relu(x)


    #     x = x.unsqueeze(2)
    #     return x

class BackscatterFixedCol(nn.Module):
    def __init__(self,
                 levels,):
        """
        An array for each column for a uniform (across space) backscatter rate.
        the array weights are trainable
        """
        super().__init__()
        self.backscatter_array = Parameter(torch.full((1,levels,1,1,1), 10.))

    def forward(self, x):
        logger.debug(torch.flatten(self.backscatter_array))
        return self.backscatter_array  # this will be inside sqrt
    

class BackscatterPrescribed(nn.Module):
    def __init__(self,
                 nlat,
                 nlon,
                 levels,
                 std_path,
                 sigma_max,
                 vary_by_latitude=False):
        """
        An prescribed array for each column for a uniform (across space) backscatter rate.
        the array weights are trainable. 
        The array is initialized based on 1% of sigma_max * std_dev of wind
        """
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon

        std = xr.open_dataset(std_path)
        std_wind_avg = (std.U.values + std.V.values) / 2.
        backscatter = torch.tensor(0.01 * (std_wind_avg * sigma_max) ** 2).reshape(1,levels,1,1,1)
        if vary_by_latitude:
            self.backscatter_array = Parameter(backscatter.repeat_interleave(self.nlat, dim=-2), requires_grad=True)
        else:
            self.backscatter_array = Parameter(backscatter, requires_grad=True)
        # total perturb will be 10% of the sqrt of the backscatter rate

    def forward(self, x):
        return self.backscatter_array

if __name__ == "__main__":

    import numpy as np
    import torch
    import yaml
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    with open("/glade/work/dkimpara/credit-camulator/config/gen_2/examples/skebs_dev.yml") as f:
        conf = yaml.safe_load(f)

    # ---------------------------------------------------------------------------
    # 2. Stubs for external I/O
    # ---------------------------------------------------------------------------

    nlat = conf["model"]["image_height"]
    nlon = conf["model"]["image_width"]

    # Fake xarray dataset returning evenly-spaced latitudes
    fake_lat    = np.linspace(-90, 90, nlat)
    fake_ds     = MagicMock()
    fake_ds.__getitem__ = MagicMock(return_value=MagicMock(values=fake_lat))

    # Identity transforms (no normalisation)
    fake_transforms              = MagicMock()
    fake_transforms.inverse_transform = lambda x: x
    fake_transforms.transform_array   = lambda x: x

    # ---------------------------------------------------------------------------
    # 4. Construct full SKEBS
    # ---------------------------------------------------------------------------

    skebs = SKEBS(conf)

    print(f"\n[SKEBS] noise_pattern lmax : {skebs.noise_pattern.lmax}")
    print(f"[SKEBS] dissipation_type   : {skebs.dissipation_type}")
    print(f"[SKEBS] trainable params   : "
        f"{[n for n, p in skebs.named_parameters() if p.requires_grad]}")

    # ---------------------------------------------------------------------------
    # 5. Build a dummy input dict and run one forward pass
    # ---------------------------------------------------------------------------

    total_channels = (
        conf["model"]["channels"] * conf["model"]["levels"]
        + conf["model"]["surface_channels"]
        + conf["model"]["output_only_channels"]
    )  # 3*8 + 2 + 0 = 26

    nlevels = conf["model"]["levels"]
    print(f"num levels: {nlevels}")

    y_pred = torch.randn(2, total_channels, 1, nlat, nlon)
    dummy_input = {
        "y_pred":        y_pred.clone(),
        "forecast_step": 1,
        "prediction": {
            "ERA5/prognostic/3d/U": y_pred[:, : nlevels].clone(),
            "ERA5/prognostic/3d/V": y_pred[:, nlevels : 2 * nlevels].clone(),
        }
    }

    print(f"[dummy_input] y_pred shape : {dummy_input['y_pred'].shape}")

    output = skebs(dummy_input)

    print((output["prediction"]["ERA5/prognostic/3d/U"] - dummy_input["prediction"]["ERA5/prognostic/3d/U"]).mean())

    print(f"[SKEBS forward] iteration     : {skebs.iteration}")