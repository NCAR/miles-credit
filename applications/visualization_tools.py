'''
A collection of functions for visualizing the forecasts 
-------------------------------------------------------
Functions:
    - cmap_combine(cmap1, cmap2)
    - get_projection(proj_name)
    - get_colormap(cmap_strings)
    - get_colormap_extend(var_range)
    - get_variable_range(data)
    - figure_panel_planner(var_num, proj)
    - cartopy_single_panel(figsize=(13, 6.5), proj=ccrs.EckertIII())
    - cartopy_panel2(figsize=(13, 8), proj=ccrs.EckertIII())
    - cartopy_panel4(var_num, figsize=(13, 6.5), proj=ccrs.EckertIII())
    - cartopy_panel6(var_num, figsize=(13, 9.75), proj=ccrs.EckertIII())
    - map_gridline_opt(AX)
    - colorbar_opt(fig, ax, cbar, cbar_extend)
    - draw_sigma_level(data, conf=None, times=None, forecast_count=None, save_location=None)
    - draw_diagnostics(data, conf=None, times=None, forecast_count=None, save_location=None)
    - draw_surface(data, conf=None, times=None, forecast_count=None, save_location=None)
    
Yingkai Sha
ksha@ucar.edu
'''

# ---------- #
# System
from os.path import join

# ---------- #
# Numerics
import datetime
import numpy as np
import xarray as xr
# ---------- #
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import colormaps as plt_cmaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import cartopy.feature as cfeature


def cmap_combine(cmap1, cmap2):
    '''
    combine two matplotlib colormaps as one.
    '''
    colors1 = cmap1(np.linspace(0., 1, 256))
    colors2 = cmap2(np.linspace(0, 1, 256))
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list('temp_cmap', colors)

def get_projection(proj_name):
    '''
    returns a cartopy projection obj
    '''
    if proj_name == 'PlateCarree':
        return ccrs.PlateCarree(central_longitude=0.0)
    elif proj_name == 'LambertCylindrical':
        return ccrs.LambertCylindrical(central_longitude=0.0)
    elif proj_name == 'Miller':
        return ccrs.Miller(central_longitude=0.0)
    elif proj_name == 'Mollweide':
        return ccrs.Mollweide(central_longitude=0.0)
    elif proj_name == 'Robinson':
        return ccrs.Robinson(central_longitude=0.0)
    elif proj_name == 'InterruptedGoodeHomolosine':
        return ccrs.InterruptedGoodeHomolosine(central_longitude=0)
    elif proj_name == 'EckertIII':
        return ccrs.EckertIII()
    else:
        print('Porjection name unkown')
        raise

def get_colormap(cmap_strings):
    '''
    returns a list of colormaps from input strings.
    '''
    colormap_obj = []    
    for cmap_name in cmap_strings:
        if cmap_name == 'viridis_plasma':
            colormap_obj.append(cmap_combine(plt.cm.viridis, plt.cm.plasma_r))
        else:
            colormap_obj.append(plt_cmaps[cmap_name])
    return colormap_obj

def get_colormap_extend(var_range):
    '''
    return colorbar extend options based on the given value range.
    '''
    if var_range[0] == 0.0:
        return 'max'
    elif var_range[1] == 0.0:
        return 'min'
    else:
        return 'both'

def get_variable_range(data):
    '''
    Estimate pcolor value ranges based on the input data.
    '''
    data_ravel = data.ravel()
    
    data_max = np.quantile(data_ravel, 0.98)
    if np.abs(np.min(data)) < 1e-2:
        data_min = 0
    else:
        data_min = np.quantile(data_ravel, 0.02)
        
    # rounding
    if data_max > 1000 or -data_min > 1000:
        round_val = 100
    elif data_max > 100 or -data_min > 100:
        round_val = 50
    elif data_max > 40 or -data_min > 40:
        round_val = 20
    elif data_max > 10 or -data_min > 10:
        round_val = 10
    elif data_max > 1.0 or -data_min > 1.0:
        round_val = 2.0
    elif data_max > 0.1 or -data_min > 0.1:
        round_val = 0.2
    elif data_max > 0.01 or -data_min > 0.01:
        round_val = 0.02
    elif data_max > 0.001 or -data_min > 0.001:
        round_val = 0.002
    elif data_max > 0.0001 or -data_min > 0.0001:
        round_val = 0.0002
    else:
        round_val = 0.00002
        
    data_max = int(np.ceil(data_max / round_val)) * round_val
    if data_min != 0:
        data_min = int(np.floor(data_min / round_val)) * round_val
        
    # 0 in the middle
    if data_min < 0 and data_max > 0:
        data_limit = np.max([-data_min, data_max])
        data_min = -data_limit
        data_max = data_limit
    
    return [data_min, data_max]

def figure_panel_planner(var_num, proj):
    '''
    Choose a figure layout based on the number of variables to plot.
    ! Handles up to 6 variables
    '''
    if var_num == 1:
        return cartopy_single_panel(figsize=(13, 6.5), proj=proj)
    elif var_num == 2:
        return cartopy_panel2(figsize=(13, 8), proj=proj)
    elif var_num == 3 or var_num == 4:
        return cartopy_panel4(var_num, figsize=(13, 6.5), proj=proj)
    elif var_num == 5 or var_num == 6:
        return cartopy_panel6(var_num, figsize=(13, 9.75), proj=proj)
    else:
        print('Built-in visualization tools do cannot plot more than 6 variables at once.')
        raise
        
def cartopy_single_panel(figsize=(13, 6.5), proj=ccrs.EckertIII()):
    '''
    Single panel figure layout
    '''
    fig = plt.figure(figsize=figsize)
    proj_ = proj
    ax = plt.axes(projection=proj_)
    AX = [ax,]
    AX = map_gridline_opt(AX)
    return fig, AX

def cartopy_panel2(figsize=(13, 8), proj=ccrs.EckertIII()):
    '''
    Two-panel figure layout
    '''
    # Figure
    fig = plt.figure(figsize=figsize)
            
    # 3-by-2 subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1,])
    proj_ = proj
    
    # subplot ax
    ax0 = plt.subplot(gs[0, 0], projection=proj_)
    ax1 = plt.subplot(gs[1, 0], projection=proj_)
    
    AX = [ax0, ax1,]
    plt.subplots_adjust(0, 0, 1, 1, hspace=0.2, wspace=0.00)
    
    # lat/lon gridlines and labeling
    AX = map_gridline_opt(AX)

    return fig, AX

def cartopy_panel4(var_num, figsize=(13, 6.5), proj=ccrs.EckertIII()):
    '''
    Four-panel figure layout
    '''
    assert (var_num > 2 and var_num <= 4)
    
    fig = plt.figure(figsize=figsize)
    
    # 2-by-2 subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    proj_ = proj
    
    # subplot ax
    ax0 = plt.subplot(gs[0, 0], projection=proj_)
    ax1 = plt.subplot(gs[0, 1], projection=proj_)
    ax2 = plt.subplot(gs[1, 0], projection=proj_)
    AX = [ax0, ax1, ax2]
    
    # if there are 4 vars to plot
    if var_num == 4:
        ax3 = plt.subplot(gs[1, 1], projection=proj_)
        AX = [ax0, ax1, ax2, ax3]
        
    plt.subplots_adjust(0, 0, 1, 1, hspace=0.2, wspace=0.05)
    
    # lat/lon gridlines and labeling
    AX = map_gridline_opt(AX)
    
    return fig, AX

def cartopy_panel6(var_num, figsize=(13, 9.75), proj=ccrs.EckertIII()):
    '''
    Six-panel figure layout
    '''
    assert (var_num > 4 and var_num <= 6)
    
    # Figure
    fig = plt.figure(figsize=figsize)
            
    # 3-by-2 subplotsvar_num, figsize=(13, 9.75), proj=ccrs.EckertIII()
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    proj_ = proj
    
    # subplot ax
    ax0 = plt.subplot(gs[0, 0], projection=proj_)
    ax1 = plt.subplot(gs[0, 1], projection=proj_)
    ax2 = plt.subplot(gs[1, 0], projection=proj_)
    ax3 = plt.subplot(gs[1, 1], projection=proj_)
    ax4 = plt.subplot(gs[2, 0], projection=proj_)
    AX = [ax0, ax1, ax2, ax3, ax4,]

    if var_num == 6:
        ax5 = plt.subplot(gs[2, 1], projection=proj_)
        AX = [ax0, ax1, ax2, ax3, ax4, ax5]
    
    plt.subplots_adjust(0, 0, 1, 1, hspace=0.2, wspace=0.05)
    
    # lat/lon gridlines and labeling
    AX = map_gridline_opt(AX)
    
    return fig, AX

def map_gridline_opt(AX):
    '''
    Customize cartopy map gridlines
    '''
    # lat/lon gridlines and labeling
    for ax in AX:
        GL = ax.gridlines(crs=ccrs.PlateCarree(), 
                          draw_labels=False, x_inline=False, y_inline=False, 
                          color='k', linewidth=0.5, linestyle=':', zorder=5)
        GL.top_labels = None; GL.bottom_labels = None
        GL.right_labels = None; GL.left_labels = None
        GL.xlabel_style = {'size': 14}; GL.ylabel_style = {'size': 14}
        GL.rotate_labels = False
    
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='k', linewidth=1.0, zorder=5)
        ax.spines['geo'].set_linewidth(2.5)
    return AX

def colorbar_opt(fig, ax, cbar, cbar_extend):
    '''
    Customize the colorbar
    '''
    CBar = fig.colorbar(cbar, location='right', orientation='vertical', 
                        pad=0.02, fraction=0.025, shrink=0.6, aspect=15, extend=cbar_extend, ax=ax)
    CBar.ax.tick_params(axis='y', labelsize=14, direction='in', length=0)
    CBar.outline.set_linewidth(2.5)
    return CBar

def draw_sigma_level(data, conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces figures for sigma-level variables. 
    '''
    # ------------------------------ #
    # visualization settings
    ## colormap
    colormaps = get_colormap(conf['visualization']['sigma_level_visualize']['colormaps'])
    
    ## variable names
    var_names = conf['visualization']['sigma_level_visualize']['variable_names']
    title_string = '{}; level {}\ntime: {}; step: {}' #.format(var_name, level, datetime, step)

    ## indices of diagnostic variables
    var_inds = conf['visualization']['sigma_level_visualize']['variable_indices']
    
    ## variable factors
    var_factors = conf['visualization']['sigma_level_visualize']['variable_factors']

    ## variable range
    var_range = conf['visualization']['sigma_level_visualize']['variable_range']
    
    ## number of levels
    ## !<--- the weights of U is applied to determine the total number of levels
    N_levels = int(len(conf['loss']['variable_weights']['U']))
    
    ## number of variables to plot
    var_num = int(len(var_inds))
    level_nums = conf['visualization']['sigma_level_visualize']['visualize_levels']
    
    ## output figure options and names
    save_options = conf['visualization']['save_options']
    save_name_head = conf['visualization']['sigma_level_visualize']['file_name_prefix']
    
    ## collect figure names
    filenames = []
    
    # ------------------------------ #
    # get forecast step and file name
    k, fn = data
    t = times[k]
    pred = np.load(fn)
    
    # ------------------------------ #
    # get lat/lon grids
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    longitude = lat_lon_weights["longitude"]
    latitude = lat_lon_weights["latitude"]
    
    for level_num in level_nums:
        # ------------------------------ #
        # Figure
        fig, AX = figure_panel_planner(var_num, get_projection(conf['visualization']['map_projection']))
        
        # pcolormesh / colorbar / title in loops
        for i_var, ind_var in enumerate(var_inds):
            # get the current axis
            ax = AX[i_var]
            
            # get the current variable
            var_ind_pick = ind_var*N_levels + level_num
            pred_draw = pred[var_ind_pick]
            pred_draw = pred_draw * var_factors[i_var]
            
            ## variable range
            var_lim = var_range[i_var]
            if var_lim == 'auto':
                var_lim = get_variable_range(pred_draw)

            ## colorbar settings
            cbar_extend = get_colormap_extend(var_lim)
            colormap = colormaps[i_var]
            
            # pcolormesh
            cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                                 cmap=colormap, transform=ccrs.PlateCarree())
            # colorbar operations
            CBar = colorbar_opt(fig, ax, cbar, cbar_extend)
            
            # title
            var_name = var_names[i_var]
            ax.set_title(title_string.format(var_name, level_num, t, k), fontsize=14)
    
        save_name = '{}_level{:02d}_fcst{:03d}_step{:03d}.png'.format(save_name_head, level_num, forecast_count, k)
        filename = join(save_location, save_name)
        plt.savefig(filename, **save_options)
        plt.close()
        filenames.append(filename)
    return k, filenames


def draw_diagnostics(data, conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces figures for diagnostics.
    '''
    # ------------------------------ #
    # visualization settings
    ## colormap
    colormaps = get_colormap(conf['visualization']['diagnostic_variable_visualize']['colormaps'])
    
    ## variable names
    var_names = conf['visualization']['diagnostic_variable_visualize']['variable_names']
    title_string = '{}\ntime: {}; step: {}' #.format(var_name, datetime, step)
    
    ## indices of diagnostic variables
    var_inds = conf['visualization']['diagnostic_variable_visualize']['variable_indices']

    ## variable range
    var_range = conf['visualization']['diagnostic_variable_visualize']['variable_range']
    
    ## variable factors
    var_factors = conf['visualization']['diagnostic_variable_visualize']['variable_factors']
    
    ## number of variables to plot
    var_num = int(len(var_inds))
    
    ## output figure options and names
    save_options = conf['visualization']['save_options']
    save_name_head = conf['visualization']['diagnostic_variable_visualize']['file_name_prefix']
    
    
    # ------------------------------ #
    # get forecast step and file name
    k, fn = data
    t = times[k]
    pred = np.load(fn)

    # ------------------------------ #
    # get lat/lon grids
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    longitude = lat_lon_weights["longitude"]
    latitude = lat_lon_weights["latitude"]
    
    # ------------------------------ #
    # Figure
    fig, AX = figure_panel_planner(var_num, get_projection(conf['visualization']['map_projection']))
    
    for i_var, ind_var in enumerate(var_inds):
        # get the current axis
        ax = AX[i_var]
        pred_draw = pred[ind_var]
        pred_draw = pred_draw * var_factors[i_var]
        
        ## variable range
        var_lim = var_range[i_var]
        if var_lim == 'auto':
            var_lim = get_variable_range(pred_draw)

        ## colorbar settings
        cbar_extend = get_colormap_extend(var_lim)
        colormap = colormaps[i_var]
    
        # pcolormesh
        cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                             cmap=colormap, transform=ccrs.PlateCarree())
    
        # colorbar operations
        CBar = colorbar_opt(fig, ax, cbar, cbar_extend)
        
        # title
        var_name = var_names[i_var]
        ax.set_title(title_string.format(var_name, t, k), fontsize=14)

    save_name = '{}_fcst{:03d}_step{:03d}.png'.format(save_name_head, forecast_count, k)
    filename = join(save_location, save_name)
    
    plt.savefig(filename, **save_options)
    plt.close()
    return k, filename


def draw_surface(data, conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces figures for surface variables
    '''
    # ------------------------------ #
    # visualization settings
    ## colormap
    colormaps = get_colormap(conf['visualization']['surface_visualize']['colormaps'])
    
    ## variable names
    var_names = conf['visualization']['surface_visualize']['variable_names']
    title_string = '{}\ntime: {}; step: {}' #.format(var_name, datetime, step)
    
    ## indices of diagnostic variables
    var_inds = conf['visualization']['surface_visualize']['variable_indices']
    
    ## variable factors
    var_factors = conf['visualization']['surface_visualize']['variable_factors']

    ## variable range
    var_range = conf['visualization']['surface_visualize']['variable_range']
    
    ## number of variables to plot
    var_num = int(len(var_inds))
    
    ## output figure options and names
    save_options = conf['visualization']['save_options']
    save_name_head = conf['visualization']['surface_visualize']['file_name_prefix']
    
    # ------------------------------ #
    # get forecast step and file name
    k, fn = data
    t = times[k]
    pred = np.load(fn)
    
    # ------------------------------ #
    # get lat/lon grids
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    longitude = lat_lon_weights["longitude"]
    latitude = lat_lon_weights["latitude"]
    
    # ------------------------------ #
    # Figure
    fig, AX = figure_panel_planner(var_num, get_projection(conf['visualization']['map_projection']))
    
    for i_var, ind_var in enumerate(var_inds):
        # get the current axis
        ax = AX[i_var]
        pred_draw = pred[ind_var]
        pred_draw = pred_draw * var_factors[i_var]
        
        ## variable range
        var_lim = var_range[i_var]
        if var_lim == 'auto':
            var_lim = get_variable_range(pred_draw)

        ## colorbar settings
        cbar_extend = get_colormap_extend(var_lim)
        colormap = colormaps[i_var]
    
        # pcolormesh
        cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                             cmap=colormap, transform=ccrs.PlateCarree())
    
        # colorbar operations
        CBar = colorbar_opt(fig, ax, cbar, cbar_extend)
        
        # title
        var_name = var_names[i_var]
        ax.set_title(title_string.format(var_name, t, k), fontsize=14)
    
    save_name = '{}_fcst{:03d}_step{:03d}.png'.format(save_name_head, forecast_count, k)
    filename = join(save_location, save_name)
    
    plt.savefig(filename, **save_options)
    plt.close()
    return k, filename
