
import xarray as xr
import torch


import logging
logger = logging.getLogger(__name__)

class ERA5StandardTransform:
    def __init__(self, conf, device):
        """
        only transform prognostic for now
        """
        self.device = device

        source_conf = conf["data"]["source"]["ERA5"]
        vars_3D = source_conf["prognostic"]["vars_3D"]
        num_levels = source_conf["prognostic"]["num_levels"]
        vars_2D = source_conf["prognostic"]["vars_2D"]
        self.num_prognostic_variables = len(vars_3D) * num_levels + len(vars_2D)

        # load dataset, sortby levels to enforce correct order
        self.mean_ds = xr.open_dataset(source_conf["prognostic"]["transform"]["mean_path"]).sortby("level", ascending=True)
        self.std_ds = xr.open_dataset(source_conf["prognostic"]["transform"]["std_path"]).sortby("level", ascending=True)

        # setup tensor scaling, use variables list from config to enforce order
        mean_3D_tensor = torch.tensor(self.mean_ds[vars_3D].to_array().stack({'level_var':['variable', 'level']}).values)
        mean_2D_tensor = torch.tensor(self.mean_ds[vars_2D].to_array().values)
        self.mean_tensor = torch.cat([mean_3D_tensor, mean_2D_tensor], axis = 0).to(device)
        
        std_3D_tensor = torch.tensor(self.std_ds[vars_3D].to_array().stack({'level_var':['variable', 'level']}).values)
        std_2D_tensor = torch.tensor(self.std_ds[vars_2D].to_array().values)
        std_tensor = torch.cat([std_3D_tensor, std_2D_tensor], axis = 0)
        self.std_tensor = std_tensor.view(1, self.num_prognostic_variables, 1, 1, 1).to(device)

    def transform_xarray(self, ds):
        variables = list(ds.keys())
        return (ds - self.mean_ds[variables]) / self.std_ds[variables]
    
    def inverse_transform_xarray(self, ds):
        variables = list(ds.keys())
        return ds * self.std_ds[variables] + self.mean_ds[variables]
    
    def transform_tensor(self, tensor):
        """
        input tensor: b,c,t,lat,lon
        """
        prognostic_tensor = tensor[ : self.num_prognostic_variables]
        prognostic_transformed = (prognostic_tensor - self.mean_tensor) / self.std_tensor

        return torch.cat([prognostic_transformed,
                          tensor[-self.num_prognostic_variables : ]
                          ])

    def inverse_transform_tensor(self, tensor):
        prognostic_tensor = tensor[ : self.num_prognostic_variables]
        prog_inv_transformed = prognostic_tensor * self.std_tensor - self.mean_tensor

        return torch.cat([prog_inv_transformed,
                          tensor[-self.num_prognostic_variables : ]
                          ])    