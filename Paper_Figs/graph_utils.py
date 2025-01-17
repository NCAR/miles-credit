'''
A collection of helper functions for the data visualization of CREDIT forecasts
--------------------------------------------------------------------------------
Content:
    - lg_box
    - lg_clean
    - ax_decorate
    - ax_decorate_box
    - string_partial_format
    - ksha_color_set_summon
    
Yingkai Sha
ksha@ucar.edu
'''

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import transforms

def lg_box(LG):
    '''
    legned block with white background and boundary lines
    '''
    LG.get_frame().set_facecolor('white')
    LG.get_frame().set_edgecolor('k')
    LG.get_frame().set_linewidth(0)
    return LG

def lg_clean(LG):
    '''
    transparent legned block without boundary lines
    '''
    LG.get_frame().set_facecolor('none')
    LG.get_frame().set_linewidth(0)
    LG.get_frame().set_alpha(1.0)

def string_partial_format(fig, ax, x_start, y_start, ha, va, string_list, color_list, fontsize_list, fontweight_list):
    '''
    String partial formatting (experimental).
    
    handles = string_partial_format(fig, ax, 0., 0.5, 'left', 'bottom',
                                    string_list=['word ', 'word ', 'word'], 
                                    color_list=['r', 'g', 'b'], 
                                    fontsize_list=[12, 24, 48], 
                                    fontweight_list=['normal', 'bold', 'normal'])
    Input
    ----------
        fig: Matplotlib Figure instance. Must contain a `canvas` subclass. e.g., `fig.canvas.get_renderer()`
        ax: Matplotlib Axis instance.
        x_start: horizonal location of the text, scaled in [0, 1] 
        y_start: vertical location of the text, scale in [0, 1]
        ha: horizonal alignment of the text, expected to be either "left" or "right" ("center" may not work correctly).
        va: vertical alignment of the text
        string_list: a list substrings, each element can have a different format.
        color_list: a list of colors that matches `string_list`
        fontsize_list: a list of fontsizes that matches `string_list`
        fontweight_list: a list of fontweights that matches `string_list`
    
    Output
    ----------
        A list of Matplotlib.Text instance.
    
    * If `fig` is saved, then the `dpi` keyword must be fixed (becuase of canvas). 
      For example, if `fig=plt.figure(dpi=100)`, then `fig.savefig(dpi=100)`.
      
    '''
    L = len(string_list)
    Handles = []
    relative_loc = ax.transAxes
    renderer = fig.canvas.get_renderer()
    
    for i in range(L):
        handle_temp = ax.text(x_start, y_start, '{}'.format(string_list[i]), ha=ha, va=va,
                              color=color_list[i], fontsize=fontsize_list[i], 
                              fontweight=fontweight_list[i], transform=relative_loc)
        loc_shift = handle_temp.get_window_extent(renderer=renderer)
        relative_loc = transforms.offset_copy(handle_temp._transform, x=loc_shift.width, units='dots')
        Handles.append(handle_temp)
        
    return Handles

def ax_decorate(ax, left_flag, bottom_flag, bottom_spline=False):
    '''
    "L" style panel axis format
    '''
    ax.grid(linestyle=':'); ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(bottom_spline)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, \
               labelbottom=bottom_flag, left=False, right=False, labelleft=left_flag)
    return ax

def ax_decorate_box(ax):
    '''
    "Box" style panel axis format
    '''
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, \
               labelbottom=False, left=False, right=False, labelleft=False)
    return ax

def ksha_color_set_summon(color_set=0):
    if color_set == 0:
        rgb_array = np.array([[0.85      , 0.85      , 0.85      , 1.        ],
                              [0.66666667, 1.        , 1.        , 1.        ],
                              [0.33333333, 0.62745098, 1.        , 1.        ],
                              [0.11372549, 0.        , 1.        , 1.        ],
                              [0.37647059, 0.81176471, 0.56862745, 1.        ],
                              [0.10196078, 0.59607843, 0.31372549, 1.        ],
                              [0.56862745, 0.81176471, 0.37647059, 1.        ],
                              [0.85098039, 0.9372549 , 0.54509804, 1.        ],
                              [1.        , 1.        , 0.4       , 1.        ],
                              [1.        , 0.8       , 0.4       , 1.        ],
                              [1.        , 0.53333333, 0.29803922, 1.        ],
                              [1.        , 0.09803922, 0.09803922, 1.        ],
                              [0.8       , 0.23921569, 0.23921569, 1.        ],
                              [0.64705882, 0.19215686, 0.19215686, 1.        ],
                              [0.55      , 0.        , 0.        , 1.        ]])
        SET0 = {'blue': rgb_array[3, :],
                'cyan': rgb_array[2, :],
                'lgreen': rgb_array[4, :],
                'green': rgb_array[5, :],
                'yellow': rgb_array[8, :],
                'orange': rgb_array[-6, :],
                'red': rgb_array[-3, :]}
        return SET0
        
    elif color_set == 1:
        SET1 = {
            'reds': ['#DB444B', '#A81829', '#C7303C', '#E64E53', '#FF6B6C', '#FF8785', '#FFA39F'],
            'blues': ['#006BA2', '#00588D', '#1270A8', '#3D89C3', '#5DA4DF', '#7BBFFC', '#98DAFF'],
            'cyans': ['#3EBCD2', '#005F73', '#00788D', '#0092A7', '#25ADC2', '#4EC8DE', '#6FE4FB'],
            'greens': ['#379A8B', '#005F52', '#00786B', '#2E9284', '#4DAD9E', '#69C9B9', '#86E5D4'],
            'oranges': ['#EBB434', '#714C00', '#8D6300', '#AA7C00', '#C89608', '#E7B030', '#FFCB4D'],
            'browns': ['#B4BA39', '#4C5900', '#667100', '#818A00', '#9DA521', '#BAC03F', '#D7DB5A'],
            'purples': ['#9A607F', '#78405F', '#925977', '#AD7291', '#C98CAC', '#E6A6C7', '#FFC2E3'],
            'yellows': ['#D1B07C', '#674E1F', '#826636', '#9D7F4E', '#B99966', '#D5B480', '#F2CF9A'],
            'grays': ['#758D99', '#3F5661', '#576E79', '#6F8793', '#89A2AE', '#A4BDC9', '#BFD8E5']
        }
        
        return SET1
