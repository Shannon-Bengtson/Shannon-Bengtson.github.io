# Load required modules
import pandas as pd 
from datetime import datetime
import numpy as np
import json

def BN_Antonio_preprocessing_ocean():
    '''
        Function used for preprocessing the data provided by Antonio (DS2)
        Reads in the files (ocean side) and formates the data
        Adds MEI data (ENSO indicator) to the file (DS5)
        
        Returns:
         - Two pandas.DataFrames; lagoon and ocean side data
    '''
    
    # Download the data 
    df_Pto_ocean = pd.read_csv(r'C:\Users\shannonb\Documents\Model_and_data\Dataset\D2_SPC_inundation_model\Pto_218_oceanside.csv') # Ocean side data 

    # Convert the date time to a proper datetime format
    df_Pto_ocean['Time'] = [datetime.strptime(x,'%m/%d/%y %H:%M:%S') for x in df_Pto_ocean['Time']]

    # Remove the tide from the total water level in both files
    # (since there is a very obvious relationship between them)
    df_Pto_ocean['TWL_point_218_less_Tide_less_MSL'] = df_Pto_ocean['Tide']-df_Pto_ocean['TWL_point_218']-df_Pto_ocean['MSL']

    ### Add DS5 (ENSO)
    # Load MEI (ENSO indicator)
    data_location = "C:\\Users\\shannonb\\Documents\\Model_and_data\\Dataset\\D5_ENSO\\"
    file_name = "MEI_preprocessed"
    with open("{}{}.json".format(data_location,file_name), 'r') as fp:
        MEI_dict = json.load(fp)
    # Add MEI to dataframe
    df_Pto_ocean['MEI'] = [float(MEI_dict[str(x.month)][str(x.year)]) for x in df_Pto_ocean.Time]
    
    # Get variables for SM1
    MSL_array = np.array(df_Pto_ocean['MSL'])
    Tide_array = np.array(df_Pto_ocean['Tide'])
    TWL_point_218_array = np.array(df_Pto_ocean['TWL_point_218'])
    Hs_point_218_array = np.array(df_Pto_ocean['Hs_point_218'])
    Tm_point_218_array = np.array(df_Pto_ocean['Tm_point_218'])
    TWL_point_218_less_Tide_less_MSL_array = np.array(df_Pto_ocean['TWL_point_218_less_Tide_less_MSL'])
    MEI_array = np.array(df_Pto_ocean['MEI'])
    
    # Put all variables for BN into a dictionary
    variables_dict = {
        'MSL':MSL_array,
        'Tide':Tide_array,
        'TWL_point_218_less_Tide_less_MSL':TWL_point_218_less_Tide_less_MSL_array,
        'Hs_point_218':Hs_point_218_array,
        'Tm_point_218':Tm_point_218_array,
        'MEI':MEI_array
    }
    
    return(df_Pto_ocean,variables_dict)