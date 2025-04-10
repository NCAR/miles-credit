# How to add a new postblock

In this example we will be going over adding a new postblock, `Foo` to CREDIT.
The postblocks are named such because they are applied after the forward pass of the model itself, but before the loss calculations. For a more detailed explaination of the currently support post-blocks see **Sha et al. 2025**. 

## Reference 

- Sha, Yingkai, et al. "[Improving AI weather prediction models using global mass and energy conservation schemes.](https://arxiv.org/abs/2501.05648)" arXiv preprint arXiv:2501.05648 (2025).


### Currently supported postblocks

#### Non-Negative Filter 

This example shows how to prevent 7 variables from going below zero

```yaml
tracer_fixer:
    activate: True
    denorm: True
    tracer_name: ['Qtot','PRECT','U10','CLDTOT','CLDHGH','CLDLOW','CLDMED']
    tracer_thres: [0, 0, 0, 0, 0, 0, 0]
```

#### Global Mass Fixer 

This example shows how to fix the model global mass  

```yaml
global_mass_fixer:
    activate: True
    activate_outside_model: True
    fix_level_num: 14  #specify presure or sigma level 
    simple_demo: False
    denorm: True
    grid_type: 'sigma'  #specify presure or sigma level 
    midpoint: True
    lon_lat_level_name: ['lon2d', 'lat2d', 'hyai', 'hybi']
    surface_pressure_name: ['PS']
    specific_total_water_name: ['Qtot']
```

#### Global Water Fixer 

This example shows how to fix the model water balance  


```yaml
global_water_fixer:
    activate: True
    activate_outside_model: True
    simple_demo: False
    denorm: True
    grid_type: 'sigma'  #specify presure or sigma level 
    midpoint: True
    lon_lat_level_name: ['lon2d', 'lat2d', 'hyai', 'hybi']
    surface_pressure_name: ['PS']
    specific_total_water_name: ['Qtot']
    precipitation_name: ['PRECT']
    evaporation_name: ['QFLX']
```

#### Global Energy Fixer

```yaml
global_energy_fixer:
    activate: True
    activate_outside_model: True
    simple_demo: False
    denorm: True
    grid_type: 'sigma'
    midpoint: True
    lon_lat_level_name: ['lon2d', 'lat2d', 'hyai', 'hybi']
    surface_pressure_name: ['PS']
    air_temperature_name: ['T']
    specific_total_water_name: ['Qtot']
    u_wind_name: ['U']
    v_wind_name: ['V']
    surface_geopotential_name: ['PHIS']
    TOA_net_radiation_flux_name: ['FSNT', 'FLNT']
    surface_net_radiation_flux_name: ['FSNS', 'FLNS']
    surface_energy_flux_name: ['SHFLX', 'LHFLX',]
```

## Create code for new postblock

One can add a new class to `credit/postblock.py` or define a new module and import it into `credit/postblock.py`. See `credit/skebs.py` for an example of the latter.

The parser will add the `data` and `model` fields from the main config to `post_conf`. Inside the class `Foo` you will be able to access these.

```python
from torch import nn

class Foo(nn.Module):
    def __init__(self, post_conf):
        super().__init__()
        self.bar = post_conf["foo"]["bar"]
        
        # accessing data or model conf
        lead_time_periods = post_conf["data"]["lead_time_periods"] 

    def forward(self, x):
        # x will be a dictionary of the previous state x and y_pred
        # of the model up to this point
        # both tensors will be in the transformed space

        y_pred = x["y_pred"]
        x_prev = x["x"]

        # do stuff ...

        x["y_pred"] = y_pred   # pack back into the dictionary
        return x

```

## Define config fields

Inside of your config you will need to add a new field for your postblock. 

```yaml
model:
    ...
    post_conf:
        ...
        foo:
            activate: True
            bar: 1.0
            ...
```

## Add to postblock module

Inside `credit/postblock.py`, append your postblock to the list of postblock operations `self.operations`, the order that you want it.

```python
from credit.skebs import SKEBS

class PostBlock(nn.Module):
    def __init__(self, post_conf):
        ...
        # SKEBS
        if post_conf["skebs"]["activate"]:
            logger.info("SKEBS registered")
            opt = SKEBS(post_conf)
            self.operations.append(opt)
        ...
        if post_conf["foo"]["activate"]:
            logger.info("foo registered")
            opt = Foo(post_conf)
            self.operations.append(opt)
        ...
```


