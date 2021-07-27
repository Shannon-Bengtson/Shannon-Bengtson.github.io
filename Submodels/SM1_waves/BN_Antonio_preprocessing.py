# Load required modules
import pandas as pd 
from datetime import datetime
import numpy as np

def BN_Antonio_preprocessing_lagoon():
    '''
        Function used for preprocessing the data provided by Antonio (DS2)
        Reads in the files (lagoon side and ocean side) and formates the data
        
        Returns:
         - Two pandas.DataFrames; lagoon and ocean side data
    '''
    
    # Download the data 
    df_Pto_lagoon = pd.read_csv(r'C:\Users\shannonb\Documents\Model_and_data\Dataset\D2_SPC_inundation_model\Pto_110_lagoonside.csv') # Lagoon side data

    # Convert the date time to a proper datetime format
    df_Pto_lagoon['Time'] = [datetime.strptime(x,'%m/%d/%y %H:%M:%S') for x in df_Pto_lagoon['Time']]

    # Convert datetime to to number of days and hours count
    df_Pto_lagoon['Time'] = df_Pto_lagoon['Time']-np.min(df_Pto_lagoon['Time'])

    # Remove the tide from the total water level in both files
    # (since there is a very obvious relationship between them)
    df_Pto_lagoon['TWL_point_110_less_Tide'] = df_Pto_lagoon['Tide']-df_Pto_lagoon['TWL_point_110']

    # Make a lists of the independent vars (excluding time)
    ind_vars_lagoon = [x for x in list(df_Pto_lagoon) if x not in ['Time','TWL_point_110','TWL_point_110_less_Tide']]

    # Make a lists of the all vars (excluding time)
    all_vars_lagoon = [x for x in list(df_Pto_lagoon) if x not in ['Time']]
    
    # Get variables for SM1
    Wind_array = np.array(df_Pto_lagoon['Wind'])
    WindDir_array = np.array(df_Pto_lagoon['WindDir'])
    Hs_offshore_array = np.array(df_Pto_lagoon['Hs_offshore'])
    Tm_offshore_array = np.array(df_Pto_lagoon['Tm_offshore'])
    Dir_offshore_array = np.array(df_Pto_lagoon['Dir_offshore'])
    Tide_array = np.array(df_Pto_lagoon['Tide'])
    TWL_point_110_array = np.array(df_Pto_lagoon['TWL_point_110'])
    Hs_point_110_array = np.array(df_Pto_lagoon['Hs_point_110'])
    MSL_array = np.array(df_Pto_lagoon['MSL'])
    
    # Put all variables for BN into a dictionary
    variables_dict = {
        'Wind':Wind_array,
        'WindDir':WindDir_array,
        'Hs_offshore':Hs_offshore_array,
        'Tm_offshore':Tm_offshore_array,
        'Dir_offshore':Dir_offshore_array,
        'Tide':Tide_array,
        'TWL_point_110':TWL_point_110_array,
        'Hs_point_110':Hs_point_110_array,
        'MSL':MSL_array,
    }
    
    return(df_Pto_lagoon,variables_dict)

def BN_Antonio_preprocessing_ocean():
    '''
        Function used for preprocessing the data provided by Antonio (DS2)
        Reads in the files (lagoon side and ocean side) and formates the data
        
        Returns:
         - Two pandas.DataFrames; lagoon and ocean side data
    '''
    
    # Download the data 
    df_Pto_ocean = pd.read_csv(r'C:\Users\shannonb\Documents\Model_and_data\Dataset\D2_SPC_inundation_model\Pto_218_oceanside.csv') # Ocean side data 

    # Convert the date time to a proper datetime format
    df_Pto_ocean['Time'] = [datetime.strptime(x,'%m/%d/%y %H:%M:%S') for x in df_Pto_ocean['Time']]

    # Convert datetime to to number of days and hours count
    df_Pto_ocean['Time'] = df_Pto_ocean['Time']-np.min(df_Pto_ocean['Time'])

    # Remove the tide from the total water level in both files
    # (since there is a very obvious relationship between them)
    df_Pto_ocean['TWL_point_218_less_Tide'] = df_Pto_ocean['Tide']-df_Pto_ocean['TWL_point_218']

    # Make a lists of the independent vars (excluding time)
    ind_vars_ocean = [x for x in list(df_Pto_ocean) if x not in ['Time','TWL_point_218','TWL_point_218_less_Tide']]

    # Make a lists of the all vars (excluding time)
    all_vars_ocean = [x for x in list(df_Pto_ocean) if x not in ['Time']]
    
    # Get variables for SM1
    MSL_array = np.array(df_Pto_ocean['MSL'])
    Tide_array = np.array(df_Pto_ocean['Tide'])
    TWL_point_218_array = np.array(df_Pto_ocean['TWL_point_218'])
    Hs_point_218_array = np.array(df_Pto_ocean['Hs_point_218'])
    Tm_point_218_array = np.array(df_Pto_ocean['Tm_point_218'])
    
    # Put all variables for BN into a dictionary
    variables_dict = {
        'MSL':MSL_array,
        'Tide':Tide_array,
        'TWL_point_218':TWL_point_218_array,
        'Hs_point_218':Hs_point_218_array,
        'Tm_point_218':Tm_point_218_array
    }
    
    return(df_Pto_ocean,variables_dict)