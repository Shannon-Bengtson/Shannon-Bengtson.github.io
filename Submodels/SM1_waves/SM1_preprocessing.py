# Load required modules
import pandas as pd 
from datetime import datetime
import numpy as np

def SM1_preprocessing_lagoon():
    '''
        Function used for preprocessing the data provided by Antonio (DS2)
        Reads in the files (lagoon side and ocean side) and formates the data
        
        Returns:
         - Two pandas.DataFrames; lagoon and ocean side data
    '''
    
    # Download the data 
    df_Pto_lagoon = pd.read_csv(r'C:\Users\Administrator\GNS Science\Pacific Island Sea Level Team - Documents\General\Model_and_data\Dataset\D2_SPC_inundation_model\Pto_110_lagoonside.csv') # Lagoon side data
    df_Pto_ocean = pd.read_csv(r'C:\Users\Administrator\GNS Science\Pacific Island Sea Level Team - Documents\General\Model_and_data\Dataset\D2_SPC_inundation_model\Pto_218_oceanside.csv') # Ocean side data 

    # Convert the date time to a proper datetime format
    df_Pto_lagoon['Time'] = [datetime.strptime(x,'%m/%d/%y %H:%M:%S') for x in df_Pto_lagoon['Time']]
    df_Pto_ocean['Time'] = [datetime.strptime(x,'%m/%d/%y %H:%M:%S') for x in df_Pto_ocean['Time']]

    # Convert datetime to to number of days and hours count
    df_Pto_lagoon['Time'] = df_Pto_lagoon['Time']-np.min(df_Pto_lagoon['Time'])
    df_Pto_ocean['Time'] = df_Pto_ocean['Time']-np.min(df_Pto_ocean['Time'])

    # Remove the tide from the total water level in both files
    # (since there is a very obvious relationship between them)
    df_Pto_lagoon['TWL_point_110_less_Tide'] = df_Pto_lagoon['Tide']-df_Pto_lagoon['TWL_point_110']
    df_Pto_ocean['TWL_point_218_less_Tide'] = df_Pto_ocean['Tide']-df_Pto_ocean['TWL_point_218']

    # Make a lists of the independent vars (excluding time)
    ind_vars_lagoon = [x for x in list(df_Pto_lagoon) if x not in ['Time','TWL_point_110','TWL_point_110_less_Tide']]
    ind_vars_ocean = [x for x in list(df_Pto_ocean) if x not in ['Time','TWL_point_218','TWL_point_218_less_Tide']]

    # Make a lists of the all vars (excluding time)
    all_vars_lagoon = [x for x in list(df_Pto_lagoon) if x not in ['Time']]
    all_vars_ocean = [x for x in list(df_Pto_ocean) if x not in ['Time']]
    
    # Get variables for SM1
    wind_speed_array = np.array(df_Pto_lagoon['Wind'])
    wind_dir_array = np.array(df_Pto_lagoon['WindDir'])
    H0_array = np.array(df_Pto_lagoon['Hs_offshore'])
    L0_array = np.array(df_Pto_lagoon['Tm_offshore'])
    H0L0_array = H0_array/L0_array
    
    # Put all variables for BN into a dictionary
    variables_dict = {
        'Wind':wind_speed_array,
        'WindDir':wind_dir_array,
        'H0':H0_array,
        'H0L0':H0L0_array
    }
    
    return(df_Pto_lagoon,variables_dict)