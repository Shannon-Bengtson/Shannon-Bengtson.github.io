import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import pysmile
import pysmile_license
import itertools

def create_SM1(wind_speed_array,wind_dir_array,H0_array,H0L0_array):
    '''
        Function for creating submodel BN 1 (SM1.xdsl)
        Args:
            - wind_speed_array:
            - wind_dir_array:
            - H0_array:
            - H0L0_array:
        Returns:
            - ds_SM1_input:
            - df_SM1_input_disc:
            - net_SM1:
            - SM1_node_ids:
        
    '''
    
    # Discretise the data using kmeans clustering
    # Set up discretising function
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
    
    # Combine into a single array
    SM1_input_array = np.vstack((wind_speed_array,wind_dir_array,H0_array,H0L0_array)).T
    
    # Use the discretiser on the array and gen output
    est.fit(SM1_input_array)
    SM1_input_disc_array = est.transform(SM1_input_array)
    
    # Turn the resulting discretised array back to a dataframe
    df_SM1_input_disc = pd.DataFrame(SM1_input_disc_array).astype(str)

    # Relabel the columns of the new dataframe
    df_SM1_input_disc.columns = ['wind_speed','wind_dir','H0','H0L0']

    # Save back as csv
    df_SM1_input_disc.to_csv('SM1_input_disc.csv',index=False)
    
    # Structure Learning using pysmile
    ds_SM1_input = pysmile.learning.DataSet()
    ds_SM1_input.read_file('SM1_input_disc.csv')

    # Use Bayesian Search algorithm to learn the network from the data
    bay_search = pysmile.learning.BayesianSearch()
    net_SM1 = bay_search.learn(ds_SM1_input)
    
    # Get all the node ids
    SM1_nodes = net_SM1.get_all_nodes()

    # Loop over every combination of the nodes and, if there is an arc, remove it
    for node1,node2 in itertools.product(SM1_nodes,SM1_nodes):
        try:
            net_SM1.delete_arc(node1,node2)
        except:
            continue

    # Create a dictionary of all node and node ids
    SM1_node_ids = {}
    for node, node_id in zip(net_SM1.get_all_node_ids(), SM1_nodes):
        SM1_node_ids.update({
            node:node_id
        })
        
    # Add arcs according to causal_diagram.drawio
    net_SM1.add_arc(SM1_node_ids['wind_dir'],SM1_node_ids['H0']) 
    net_SM1.add_arc(SM1_node_ids['wind_speed'],SM1_node_ids['H0']) 
    net_SM1.add_arc(SM1_node_ids['wind_dir'],SM1_node_ids['H0L0']) 
    net_SM1.add_arc(SM1_node_ids['wind_speed'],SM1_node_ids['H0L0']) 
    
    # Setup  the validator
    matching_SM1 = ds_SM1_input.match_network(net_SM1)
    validator_SM1 = pysmile.learning.Validator(net_SM1, ds_SM1_input, matching_SM1)

    # Validate the results on the total water level node 
    classNodeHandle = net_SM1.get_node("H0")

    # Add total water level node to the validator
    validator_SM1.add_class_node(classNodeHandle)

    # Get the expectation maximisation (EM) function for parameter estimation
    em = pysmile.learning.EM()

    # Using expectation-maximisation, determine the accuracy using k_fold (5) cross validation
    validator_SM1.k_fold(em, 5)
    acc = validator_SM1.get_accuracy(classNodeHandle, 0)

    print("Accuracy for predicting H0: ",acc)

    # Validate the results on the total water level node 
    classNodeHandle = net_SM1.get_node("H0L0")

    # Add total water level node to the validator
    validator_SM1.add_class_node(classNodeHandle)

    # Get the expectation maximisation (EM) function for parameter estimation
    em = pysmile.learning.EM()

    # Using expectation-maximisation, determine the accuracy using k_fold (5) cross validation
    validator_SM1.k_fold(em, 5)
    acc = validator_SM1.get_accuracy(classNodeHandle, 0)

    print("Accuracy for predicting H0L0: ",acc)
    
    #########################
    
    # Learn the paramters using EM
    em.learn(data=ds_SM1_input, net=net_SM1, matching=matching_SM1)

    # Save the network
    net_SM1.write_file("SM1.xdsl")
    
    return(ds_SM1_input,df_SM1_input_disc,net_SM1,SM1_node_ids)