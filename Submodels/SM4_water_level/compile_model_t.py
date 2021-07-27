import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import os
import json
from datetime import datetime
import pysmile
import pysmile_license
import sys
import json
sys.path.append('/src/python_classes')
import rpy2
# os.environ['R_HOME'] = 'C:\ProgramData\Anaconda3\Lib\R'
# %load_ext rpy2.ipython
from matplotlib.animation import FuncAnimation
import geojson
import folium
from colormap import rgb2hex


from BNModel import BNModel

from preprocessing_all_points import *
from preprocessing_points_spatially import *
from compile_model_t import *


def create_BN_time_t(lagoon_model_dict,
                     lagoon_data_dict,
                     df_lagoon,
                     ocean_model_dict,
                     df_ocean,
                     ocean_data_dict):
    '''
    
    '''
    
    # Bootstrap the data, and add it to the model_dict
    lagoon_model_dict = BNModel().bootstrap_data(lagoon_model_dict,lagoon_data_dict,df_lagoon)
    ocean_model_dict = BNModel().bootstrap_data(ocean_model_dict,ocean_data_dict,df_ocean)
    
    # Discretise the data
    lagoon_file_label = "lagoon"
    ocean_file_label = "ocean"

    # Discretise the data
    lagoon_model_dict = BNModel().discretiser(lagoon_model_dict)
    ocean_model_dict = BNModel().discretiser(ocean_model_dict)

    # Save the dataset
    BNModel().save_dataset(lagoon_model_dict,lagoon_file_label)
    BNModel().save_dataset(ocean_model_dict,ocean_file_label)

    # Create the BN
    lagoon_model_dict = BNModel().create_SM(lagoon_model_dict,lagoon_file_label)
    ocean_model_dict = BNModel().create_SM(ocean_model_dict,ocean_file_label)

    return(lagoon_model_dict,ocean_model_dict)
