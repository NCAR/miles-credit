import copy
import os
from os.path import join

import torch
from torch import nn
import torch_harmonics as harmonics
import segmentation_models_pytorch as smp

from torch.amp import custom_fwd

import numpy as np
import xarray as xr

from credit.boundary_padding import TensorPadding
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from credit.transforms import load_transforms
from credit.physics_constants import (
    RAD_EARTH,
    PI
)

import logging
logger = logging.getLogger(__name__)


def concat_for_inplace_ops(y_orig, y_inplace_slice, ind_start, ind_end):
    """
    alternate way to concat tensors along first dim, 
    given a set of indices to replace that are contiguous 
    """
    tensors = [
        y_orig[:, :ind_start],
        y_inplace_slice,
        y_orig[:, ind_end + 1:]
    ]
    new_tensor = torch.concat(tensors, dim=1) # concat on channel dim
    return new_tensor


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

class BackscatterUnet(nn.Module):
    def __init__(self,
                 in_channels,
                 levels,
                 nlat,
                 nlon,
                 architecture,
                 padding):
        """
        configurable unet to predict the backscatter rate. architectures and weights can be loaded
        """
        # could also predict with x_prev and y
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.in_channels = in_channels
        self.levels = levels
        
        # setup padding functions
        self.pad = padding
        if self.pad:
            logger.info(f"padding size {self.pad} inside unet")
            self.boundary_padding = TensorPadding(pad_lat=(self.pad, self.pad),
                                                pad_lon=(self.pad, self.pad))

        self.relu = nn.ReLU()

        if architecture is None:
            architecture = {
                            "name": "unet++",
                            "encoder_name": "resnet34",
                            "encoder_weights": "imagenet",
                        }
        if architecture["name"] == "unet":
            architecture["decoder_attention_type"] = "scse"

        architecture["in_channels"] = in_channels
        architecture["classes"] = levels

        self.model = load_premade_encoder_model(architecture)
    
    def forward(self, x):
        x = x.squeeze(2) # squeeze out time dim (see above)

        if self.pad:
            x = self.boundary_padding.pad(x)
        
        x = self.model(x)

        if self.pad:
            x = self.boundary_padding.unpad(x)

        x = self.relu(x)


        x = x.unsqueeze(2)
        return x

class BackscatterFixedCol(nn.Module):
    def __init__(self,
                 levels,):
        """
        An array for each column for a uniform (across space) backscatter rate.
        the array weights are trainable
        """
        super().__init__()
        self.backscatter_array = Parameter(torch.full((1,levels,1,1,1), 2.5))

    def forward(self, x):
        logger.debug(torch.flatten(self.backscatter_array))
        return self.backscatter_array  # this will be inside sqrt
    

class BackscatterPrescribed(nn.Module):
    def __init__(self,
                 nlat,
                 nlon,
                 levels,
                 std_path,
                 sigma_max):
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
        self.backscatter_array = Parameter(torch.tensor(1e-2 * (std_wind_avg * sigma_max) ** 2).reshape(1,levels,1,1,1), requires_grad=True)
        # formula to convert to 1% of sigma * std_wind_avg

    def forward(self, x):
        return self.backscatter_array
    

class SKEBS(nn.Module):
    """
    post_conf: dictionary with config options for PostBlock.
                if post_conf is not specified in config,
                defaults are set in the parser

    """

    def __init__(self, post_conf):
        """
        A model error correction scheme inspired by [1]. The scheme perturbs the streamfunction of horizontal wind (U,V) and is a nondivergent perturbation.
        Randomness is drawn from a red-noise spectral pattern. The amplitude of the perturbations are predicted by a neural net (calling it backscatter to
        align with [1]). 

        The streamfunction perturbation in grid space is

        Phi(lat, lon, z, t) = sqrt( r * D(x,y,z,t) / ΔE ) * phi(lat, lon, t)

        where D is the "backscatter rate" and phi is spectral stochastic red noise. See [1] for further details.

        [1] Berner, J., Shutts, G. J., Leutbecher, M., & Palmer, T. N. (2009). A spectral stochastic kinetic energy backscatter scheme 
        and its impact on flow-dependent predictability in the ECMWF ensemble prediction system. Journal of the Atmospheric Sciences, 66(3), 603-626.
        """
        super().__init__()

        self.post_conf = post_conf
        self.retain_graph = post_conf["data"].get("retain_graph", False)

        self.nlon = post_conf["model"]["image_width"]
        self.nlat = post_conf["model"]["image_height"]
        self.channels = post_conf["model"]["channels"]
        self.levels = post_conf["model"]["levels"]
        self.surface_channels = post_conf["model"]["surface_channels"]
        self.output_only_channels = post_conf["model"]["output_only_channels"]
        self.input_only_channels = post_conf["model"]["input_only_channels"]
        self.frames = post_conf["model"]["frames"]

        self.forecast_len = post_conf["data"]["forecast_len"] + 1
        self.valid_forecast_len = post_conf["data"]["valid_forecast_len"] + 1
        self.multistep = self.forecast_len > 1
        self.lmax = post_conf["skebs"]["lmax"]
        self.mmax = post_conf["skebs"]["mmax"]
        self.grid = post_conf["grid"]
        self.U_inds = post_conf["skebs"]["U_inds"]
        self.V_inds = post_conf["skebs"]["V_inds"]
        self.T_inds = post_conf["skebs"]["T_inds"]
        self.Q_inds = post_conf["skebs"]["Q_inds"]
        self.sp_index = post_conf["skebs"]["SP_ind"]
        self.static_inds = post_conf["skebs"]["static_inds"]
        
        # setup coslat, gets expanded to ensemble size and to device in first forward pass
        cos_lat = np.cos(np.deg2rad(xr.open_dataset(post_conf["data"]["save_loc_static"])["latitude"])).values
        self.cos_lat = torch.tensor(cos_lat).reshape(1, 1, 1, cos_lat.shape[0], 1).expand(1,1,1,cos_lat.shape[0], 288)
        
        self.state_trans = load_transforms(post_conf, scaler_only=True)
        self.eps = 1e-12

        # check for contiguous indices, need this for concat operation
        assert np.all(np.diff(self.U_inds) == 1) and np.all(self.U_inds[:-1] <= self.U_inds[1:])
        assert np.all(np.diff(self.V_inds) == 1) and np.all(self.V_inds[:-1] <= self.V_inds[1:])

        # initialize specific params    
        self.alpha_init = post_conf["skebs"].get("alpha_init", 0.125)
        self.zero_out_levels_top_of_model = post_conf["skebs"].get("zero_out_levels_top_of_model", 3)
        logger.info(f"filtering out top {self.zero_out_levels_top_of_model} levels of skebs perturbation")

        self.tropics_only_dissipation = post_conf["skebs"].get("tropics_only_dissipation", False)


        ### initialize filters, pattern, and spherical harmonic transforms
        self.initialize_sht()
        self.initialize_skebs_parameters()   
        self.initialize_filters()
   

        # coeffs havent been spun up yet (indicates need to init the coeffs)
        self.spec_coef_is_initialized = False

        # freeze pattern weights before init backscatter
        self.freeze_pattern_weights = post_conf["skebs"].get("freeze_pattern_weights", False)
        if self.freeze_pattern_weights:
            logger.warning("freezing all skebs pattern weights")
            for param in self.parameters():
                param.requires_grad = False

        ############### initialize backscatter prediction ###############
        #################################################################
        self.use_statics = post_conf["skebs"].get("use_statics", True)
        num_channels = (self.channels * self.levels 
                            + post_conf["model"]["surface_channels"]
                            + post_conf["model"]["output_only_channels"])
        if self.use_statics:
            num_channels += len(self.static_inds) + 1 # this one for static vars and coslat
        
        self.relu1 = nn.ReLU() # use this to gaurantee positive backscatter

        self.dissipation_scaling_coefficient = torch.tensor(post_conf["skebs"].get("dissipation_scaling_coefficient", 1.0))
        self.dissipation_type = post_conf["skebs"]["dissipation_type"]
        if self.dissipation_type == "prescribed":
            self.backscatter_network = BackscatterPrescribed(self.nlat, self.nlon, self.levels,
                                                             post_conf["data"]["std_path"],
                                                             post_conf["skebs"]["sigma_max"])
        elif self.dissipation_type == "uniform":
            self.backscatter_network = BackscatterFixedCol(self.levels)
        elif self.dissipation_type == "FCNN":
            self.backscatter_network = BackscatterFCNN(num_channels, self.levels)
        elif self.dissipation_type == "FCNN_wide":
            self.backscatter_network = BackscatterFCNNWide(num_channels, self.levels)
        elif self.dissipation_type == "CNN":
            self.backscatter_network = BackscatterCNN(num_channels, self.levels, self.nlat, self.nlon)
        elif self.dissipation_type == "unet":
            architecture = post_conf["skebs"].get("architecture", None)
            padding = post_conf["skebs"].get("padding", 48)
            self.backscatter_network = BackscatterUnet(num_channels,
                                                        self.levels,
                                                        self.nlat,
                                                        self.nlon,
                                                        architecture,
                                                        padding)
        else:
            raise RuntimeError(f"{self.dissipation_type} is a not a valid dissipation type, please modify config")
        
        logger.info(f"using dissipation type: {self.dissipation_type}")

        # freeze backscatter weights if needed
        if post_conf["skebs"].get("freeze_dissipation_weights", False):
            logger.warning("freezing all dissipation predictor weights")
            for param in self.backscatter_network.parameters():
                param.requires_grad = False

        # turn off training for all skebs params
        if not post_conf["skebs"].get("trainable", True):
            logger.warning("freezing all SKEBS parameters due to skebs config")
            for param in self.parameters():
                param.requires_grad = False

        # reset training for certain params
        self.train_alpha = post_conf["skebs"].get("train_alpha", False)
        if self.train_alpha:
            self.alpha.requires_grad = True
            logger.info("training alpha")
        
        train_backscatter_filter = post_conf["skebs"].get("train_backscatter_filter", False)
        if train_backscatter_filter:
            self.spectral_backscatter_filter.requires_grad = True
            logger.info("training backscatter filter")

        train_pattern_filter = post_conf["skebs"].get("train_pattern_filter", False)
        if train_pattern_filter: 
            self.spectral_pattern_filter.requires_grad = True
            logger.info("training pattern filter")

        logger.info(f"trainable params{[name for name, param in self.named_parameters() if param.requires_grad]}")


        ########### debugging and analysis features #############
        self.is_training = False
        self.iteration = 0

        self.write_rollout_debug_files = post_conf["skebs"].get("write_rollout_debug_files", True)

        self.write_train_debug_files = post_conf['skebs'].get('write_train_debug_files', False)
        self.write_every = post_conf['skebs'].get('write_train_every', 999)

        save_loc = post_conf['skebs']["save_loc"]
        self.debug_save_loc = join(save_loc, "debug_skebs")

        if self.write_train_debug_files or self.write_rollout_debug_files:
            os.makedirs(self.debug_save_loc, exist_ok=True)
            logger.info("writing SKEBS debugging files")


        ############# early shutoff ###################
        self.iteration_stop = post_conf['skebs'].get("iteration_stop", 0)
        if self.iteration_stop:
            logger.info(f"SKEBS is STOPPING at iteration {self.iteration_stop}")

    def initialize_sht(self):
        """
        Initialize spherical harmonics and inverse spherical harmonics transformations
        for both scalar and vector fields.
        """
        # Initialize spherical harmonics transformation objects
        self.sht = harmonics.RealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        # self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        )
        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

        # Compute quadrature weights and cosine of latitudes for the grid
        # cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(
        #     self.nlat, -1, 1
        # )

        ## equiangular grid
        cost, w = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)
        self.lats = -torch.as_tensor(np.arcsin(cost))
        self.lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[
            : self.nlon
        ]

        l_arr = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l_arr = l_arr.expand(self.lmax, self.mmax)
        self.register_buffer("lap", -l_arr * (l_arr + 1) / RAD_EARTH**2,
                             persistent=False)
        self.register_buffer("invlap", -(RAD_EARTH**2) / l_arr / (l_arr + 1),
                             persistent=False)
        self.invlap[0] = 0.0  # Adjusting the first element to avoid division by zero

        logging.info(f"lmax: {self.lmax}, mmax: {self.mmax}")

    def initialize_skebs_parameters(self):
        """ initialize the trainable parameters of skebs (everything excluding the backscatter) """
        self.register_buffer('backscatter_filter', # filter to zero out backscatter at top of model
                             torch.cat([
                                 torch.zeros(self.zero_out_levels_top_of_model),
                                 torch.ones(self.levels - self.zero_out_levels_top_of_model)
                             ]).view(1, self.levels, 1, 1, 1),
                             persistent=False) 
        if self.tropics_only_dissipation:
            self.register_buffer('tropics_backscatter_filter',
                                 torch.cat([
                                 torch.zeros(70),
                                 torch.ones(52),
                                 torch.zeros(70)
                                ]).view(1, 1, 1, self.nlat, 1),
                                persistent=False) 
        else:
            self.register_buffer('tropics_backscatter_filter',
                        torch.ones(self.nlat).view(1, 1, 1, self.nlat, 1),
                        persistent=False)
            

        self.register_buffer('lrange', 
                             torch.arange(1, self.lmax + 1).unsqueeze(1),
                             persistent=False) # (lmax, 1)
        # assume (b, c, t, ,lat,lon)
        # parameters we want to learn: (default init to berner 2009 values)
        if self.multistep:
            logger.info("multi-step skebs")
            self.alpha = Parameter(torch.tensor(self.alpha_init, requires_grad=True))
        else:
            logger.info("single-step skebs")
            self.alpha = Parameter(torch.tensor(1.0, requires_grad=False))
            self.alpha.requires_grad = False 
        self.variance = Parameter(torch.tensor(0.083, requires_grad=True))
        self.p = Parameter(torch.tensor(-1.27, requires_grad=True))
        self.dE = Parameter(torch.tensor(1e-4, requires_grad=True))
        self.r = Parameter(torch.tensor(0.02, requires_grad=False)) # see berner 2009, section 4a
        self.r.requires_grad = False
        # initialize spectral filters

    def initialize_filters(self):
        """ initialize the spectral filters for the backscatter prediction, and the spectral pattern """
        def filter_init(max_wavenum, anneal_start):
            filter = torch.cat([
                                torch.ones(anneal_start),
                                torch.linspace(1., 0.2, max_wavenum - anneal_start),
                                torch.zeros(self.lmax - max_wavenum)
                                ])
            return filter.view(1,1,1,self.lmax, 1)
        
        p_max =  self.post_conf["skebs"].get("max_pattern_wavenum", 60)
        p_anneal = self.post_conf["skebs"].get("pattern_filter_anneal_start", 40)

        self.spectral_pattern_filter = Parameter(filter_init(p_max, p_anneal),
                                         requires_grad=False)
        
        b_max =  self.post_conf["skebs"].get("max_backscatter_wavenum", 100)
        b_anneal = self.post_conf["skebs"].get("backscatter_filter_anneal_start", 90)

        self.spectral_backscatter_filter = Parameter(filter_init(b_max, b_anneal),
                                         requires_grad=False)


    def clip_parameters(self):
        """ clip the trainable parameters so that they are always physical"""
        self.alpha.data = self.alpha.data.clamp(self.eps, 1.)
        self.variance.data = self.variance.clamp(self.eps, 10.)
        self.p.data = self.p.data.clamp(-10, -self.eps)
        self.dE.data = self.dE.data.clamp(self.eps, 1.)
        self.r.data = self.r.data.clamp(self.eps, 1.)
        self.spectral_pattern_filter.data = self.spectral_pattern_filter.data.clamp(0., 1.)
        self.spectral_backscatter_filter.data = self.spectral_backscatter_filter.data.clamp(0., 1.)


    def initialize_pattern(self, y_pred):
        """
        initialize the random red noise pattern.
        in Berner et al
            m is zonal wavenumber -> mmax
            n is total wavenumber -> lmax
        """
        if self.iteration > 0:
            self.spec_coef = self.spec_coef.detach()
        y_shape = y_pred.shape

        self.spec_coef = torch.zeros(
                                 (y_shape[0], 1, 1, self.lmax, self.mmax),  # b, 1, 1, lmax, mmax
                                 dtype=torch.cfloat,
                                 device=y_pred.device)
        self.multivariateNormal = MultivariateNormal(torch.zeros(2, device=y_pred.device), 
                                                     torch.eye(2, device=y_pred.device))
        # initialize pattern todo: how many iters?
        iters = 5
        logger.debug(f"initializing pattern with {iters} iterations")
        for i in range(iters):
            self.spec_coef = self.cycle_pattern(self.spec_coef)

    def cycle_pattern(self, spec_coef):
        """
        cycle the random red noise pattern that is temporally correlated
        in Berner et al
            m is zonal wavenumber -> mmax
            n is total wavenumber -> lmax
        """
        Gamma = torch.sum(self.lrange * (self.lrange + 1.0) * (2 * self.lrange + 1.0) * self.lrange ** (2.0 * self.p))  # scalar
        self.b = torch.sqrt((4.0 * PI * RAD_EARTH**2.0) / (self.variance * Gamma) * self.alpha * self.dE)  # scalar
        self.g_n = self.b * self.lrange ** self.p  # (lmax, 1)

        cmplx_noise = torch.view_as_complex(self.multivariateNormal.sample(spec_coef.shape))
        noise = self.variance * cmplx_noise
        new_coef = (1.0 - self.alpha) * spec_coef + self.g_n * torch.sqrt(self.alpha) * noise  # (lmax, mmax)
        return new_coef * self.spectral_pattern_filter 
    
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x):
        """ the inverse sht operation requires float32 or greater """
 
        if self.is_training: # this checks if we are in a training script
            # self.training is a torch level thing that checks if we are in train/validation mode of training
            # for inference, we don't need to reset the pattern
            if self.training and self.steps >= self.forecast_len:
                self.spec_coef_is_initialized = False
                self.spec_coef = self.spec_coef.detach()
                logger.debug(f"pattern is reset after train step {self.steps} total iter {self.iteration}")
            elif not self.training and self.steps >= self.valid_forecast_len:
                self.spec_coef_is_initialized = False
                self.spec_coef = self.spec_coef.detach()
                logger.debug(f"pattern is reset after valid step {self.steps} total iter {self.iteration}")
        
        if not self.is_training and torch.is_grad_enabled():
            self.is_training = True # set and forget to detect if we are in a training script

        # shutoff skebs if specified
        if self.iteration_stop > 0 and self.iteration > self.iteration_stop:
            if self.iteration == self.iteration_stop + 1:
                logger.info(f"skebs stopped at {self.iteration_stop} steps")
                
            return x
        

        ################### SKEBS ################### 
        #############################################
        #############################################
        #############################################

        ################### BACKSCATTER ################### 
        ######## setup input data for backscatter #########

        x_input_statics = x["x"][:, self.static_inds]
        x = x["y_pred"]

        if not self.retain_graph:
            x = x.detach()

        if self.iteration == 0:
            self.cos_lat = self.cos_lat.to(x.device).expand(x.shape[0], *self.cos_lat.shape[1:])

        if self.use_statics: # compatibility with old fcnn
            x = torch.cat([x, x_input_statics, self.cos_lat], dim=1) # add in static vars and coslat

        # takes in raw (transformed) model output
        # filter out model top and tropics if specified
        backscatter_pred = (self.dissipation_scaling_coefficient 
                            * self.backscatter_filter 
                            * self.tropics_backscatter_filter 
                            * self.backscatter_network(x)) 

        if ((self.write_rollout_debug_files and not self.is_training) # save out raw all backscatter prediction when not training
            or (self.write_train_debug_files and self.iteration % self.write_every == 0)):
            logger.info(f"writing backscatter file for iter {self.iteration}")
            torch.save(backscatter_pred, join(self.debug_save_loc, f"backscatter_raw_{self.iteration}"))
            
        if self.dissipation_type not in ["prescribed", "uniform"]:
            # spatially filter the backscatter 
            backscatter_spec = self.sht(backscatter_pred) # b, levels, t, lmax, mmax
            backscatter_spec = self.spectral_backscatter_filter * backscatter_spec
            backscatter_pred = self.isht(backscatter_spec)

        backscatter_pred = self.relu1(backscatter_pred) 

        logger.debug(f"max backscatter: {torch.max(torch.abs(backscatter_pred))}")

        if ((self.write_rollout_debug_files and not self.is_training) # save out filtered backscatter
            or (self.write_train_debug_files and self.iteration % self.write_every == 0)):
            logger.info(f"writing backscatter file for iter {self.iteration}")
            torch.save(backscatter_pred, join(self.debug_save_loc, f"backscatter_{self.iteration}"))
        
        ################### SKEBS pattern ####################

        # inverse transform to get real output
        x = self.state_trans.inverse_transform(x)

        # (re)-initialize the pattern or clip the updated parameters
        if not self.spec_coef_is_initialized: #hacky way of doing lazymodulemixin
            self.steps = 0
            self.initialize_pattern(x)
            self.spec_coef_is_initialized = True

        else:
            self.clip_parameters()

        # cycle the pattern
        self.spec_coef = self.cycle_pattern(self.spec_coef) # b, 1, 1, lmax, mmax

        # transform pattern to grid space
        spec_coef = self.spec_coef.squeeze()
        u_chi, v_chi = self.getgrad(spec_coef) # spec_coef represent vrt (non-divergent) coeffs so the perturbation will be non-divergent
        u_chi, v_chi = u_chi.unsqueeze(1).unsqueeze(1), v_chi.unsqueeze(1).unsqueeze(1)
        logger.debug(f"max u_chi: {torch.max(torch.abs(u_chi))}")
        logger.debug(f"max v_chi: {torch.max(torch.abs(v_chi))}")
        # compute the dissipation term
        dissipation_term = torch.sqrt(self.r * backscatter_pred / self.dE) # shape (b, levels, 1, lat, lon)
        # sqrt(2e-2 * 1e1 * 1e4) * 0.5e-3 (pattern)
        # 1.4e1.5 * 0.5e-3 = 0.7e-1.5 = 0.2
        # pattern: 1e-3

        #############################################################
        ################### DEBUG perturbations #####################


        ## debug skebs, write out physical values 
        if self.write_train_debug_files and self.iteration % self.write_every == 0:
            torch.save(self.spectral_pattern_filter, join(self.debug_save_loc, f"spectral_filter_{self.iteration}"))
            torch.save(self.spectral_backscatter_filter, join(self.debug_save_loc, f"spectral_backscatter_filter_{self.iteration}"))
            # add_wind_magnitude = torch.sqrt(dissipation_term ** 2 * (u_chi ** 2 + v_chi ** 2))
            # logger.debug(f"perturb max/min: {add_wind_magnitude.max():.2f}, {add_wind_magnitude.min():.2f}")
            # torch.save(add_wind_magnitude, join(self.debug_save_loc, f"perturb_{self.iteration}"))
            # torch.save(pattern_on_grid, join(self.debug_save_loc, f"pattern_{self.iteration}"))
            # torch.save(x, join(self.debug_save_loc, f"x_{self.iteration}"))

        #############################################################
        ##################### perturb fields ########################

        u_perturb = dissipation_term * u_chi
        v_perturb = dissipation_term * v_chi

        if ((self.write_rollout_debug_files and not self.is_training) # save out raw all backscatter prediction when not training
            or (self.write_train_debug_files and self.iteration % self.write_every == 0)):
            torch.save(u_perturb, join(self.debug_save_loc, f"u_perturb_{self.iteration}"))
            torch.save(v_perturb, join(self.debug_save_loc, f"v_perturb_{self.iteration}"))

        logger.debug(f"max u perturb: {torch.max(torch.abs(u_perturb))}")
        logger.debug(f"max v perturb: {torch.max(torch.abs(v_perturb))}")

        x_u_wind = x[:, self.U_inds] + u_perturb
        x_v_wind = x[:, self.V_inds] + v_perturb
        
        x = concat_for_inplace_ops(x, x_u_wind, min(self.U_inds), max(self.U_inds))
        x = concat_for_inplace_ops(x, x_v_wind, min(self.V_inds), max(self.V_inds))
        
        # transform back to model (transformed) output space
        x = self.state_trans.transform_array(x)

        # check for nans
        assert not torch.isnan(x).any()

        #############################################################
        ################### setup next iteration #####################
        self.iteration += 1 # this one for total iterations
        self.steps += 1  # this one for skebs/model state
        
        return x
    
    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)
    
    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht(self.invlap * vrtdivspec / RAD_EARTH)

    def getgrad(self, chispec):
        """
        compute vector gradient on grid given complex spectral coefficients.

        Args:
            chispec: rank 1 or 2 or 3 tensor complex array with shape
        `(ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt)` containing
        complex spherical harmonic coefficients (where ntrunc is the
        triangular truncation limit and nt is the number of spectral arrays
        to be transformed). If chispec is rank 1, nt is assumed to be 1.

        Returns:
            C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
        gridded zonal and meridional components of the vector gradient.
        Shapes are either (nlat,nlon) or (nlat,nlon,nt).
        """
        idim = chispec.ndim

        if (
            len(chispec.shape) != 1
            and len(chispec.shape) != 2
            and len(chispec.shape) != 3
        ):
            msg = "getgrad needs rank one or two arrays!"
            raise ValueError(msg)

        ntrunc = int(
            -1.5
            + 0.5
            * torch.sqrt(
                9.0 - 8.0 * (1.0 - torch.tensor(self.nlat))
            )
        )

        if len(chispec.shape) == 1:
            chispec = torch.view(chispec, ((ntrunc + 1) * (ntrunc + 2) // 2, 1))

        divspec2 = self.lap * chispec

        if idim == 1:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]),
                        divspec2,
                    )
                ).to(divspec2.device)
            )
            return torch.squeeze(uchi), torch.squeeze(vchi)
        elif idim == 2:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]).to(divspec2.device),
                        divspec2,
                    )
                )
            )
            return uchi, vchi
        elif idim == 3:
            # new_shape = (divspec2.shape[0], 2, *divspec2.shape[1:])
            # stacked_divspec = torch.zeros(
            #     new_shape, dtype=torch.complex64
            # ).to(divspec2.device)
            # # Copy the original data into the second slice of the new dimension
            # stacked_divspec[:, 1, :, :] = divspec2
            stacked_divspec = torch.concat([divspec2.unsqueeze(1),
                                            divspec2.unsqueeze(1)],
                                            dim=1)

            backy = self.getuv(stacked_divspec)
            uchi = backy[:, 0, :, :]
            vchi = backy[:, 1, :, :]
            return uchi, vchi
        else:
            print("nothing happening here")