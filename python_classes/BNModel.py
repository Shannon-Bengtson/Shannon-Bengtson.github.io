import pysmile
import pysmile_license
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import itertools
import graphviz
import os
import glob
import pylab
import math
from scipy.interpolate import splrep, splev

class BNModel():
    
    def __init__(name):
        
        # Name the BN model
        self = name
        
        # Create descretised array
#         self.df_input_disc = self.discretiser(variables_dict)
        
#         # Create the submodel
#         self.ds_input,self.net,self.node_ids_dict = self.create_SM()

    def bootstrap_data(self,model_dict,data_dict,df):
        '''
        This function bootstraps the data and adds it to the model_dict
        inputs:
            - The model dictionary containing all the variable metadata (but not data)
            - The data dictionary, produced by preprocessing, that has all dat ain it
            - df: Dataframe of the data
        returns:
            - Model dictionary with the raw data added, divided into bootstrap reps
        '''

        # calculate the length of the training dataset
        len_training = int(model_dict['training_frac']*len(df))

        # Initialise dictionary of training and testing preprocessed arrays in model_dict each var
        for var_key in model_dict['variables'].keys():
            model_dict['variables'][var_key]['training_data_preprocessed'] = {}
            model_dict['variables'][var_key]['testing_data_preprocessed'] = {}

        # Repeat bootstrapping the same number of times as the bootstrap count    
        for rep in np.arange(0,model_dict['bootstrap_reps'],1):

            # Randomly select data from all data to make training dataset
            df_training = df.sample(len_training)
            # Get the remaining data as the testing data
            df_testing = df[~df.index.isin(df_training.index)]

            # order the indecies
            training_index = np.sort(df_training.index)
            testing_index = np.sort(df_testing.index)

            # Get the arrays for testing and training to add to the model_dict
            for var_key, data_array in data_dict.items():

                var_training_array = data_array[training_index]
                var_testing_array = data_array[testing_index]

                # Add arrays to dictionary
                model_dict['variables'][var_key]['training_data_preprocessed'].update({
                    rep:var_training_array
                })
                model_dict['variables'][var_key]['testing_data_preprocessed'].update({
                    rep:var_testing_array
                })

        return(model_dict)
        
    def discretiser(self,model_dict,exclusion_list):
        '''
        This function goes over all the variables in the model_dict, and discretises them using KBinsDiscretizer.
        Importantly, it also adds all data to the model_dict (even if it doesn't need discretisation)
        input:
            - Model dictionary, with the variable discretisation meta data
        returns:
            - Model dictionary with the data discretised and variables added        
        '''
        
        # Just get the variables dictionary
        var_dict = model_dict['variables']
        
        # Create an empty dictionary to put the discretised variables arrays into
        discretised_arrays_dict = {}

        # loop over each parameter
        for var_key,var_params in var_dict.items():
    
            # See if this parameter needs to be discretised
            if 'discretisation' in var_params.keys():
                
                # Get just the discretisation parameters dictionary
                disc_params = var_params['discretisation']
                
                # See if the descritisation edges have already been set.
                if 'bin_edges' in disc_params.keys():

                    # Get dictionary of all the iterations of the training data (bootstrapped)
                    training_data_dict = var_dict[var_key]['training_data_preprocessed']

                    # Create new dictionaries for discretised data
                    discretised_training_dict = {}
                    discretised_testing_dict = {}
                    bin_edges_dict = {}

                    # Loop over each iteration
                    for rep, training_array in training_data_dict.items():

                        # Create an array of strings to make as binned data
                        discretised_training_array = np.empty(len(training_array)).astype(str)

                        # Using the bounds already found, discretise the data
                        for lower, upper, bin_name in zip(
                            disc_params['bin_edges'][:-1],
                            disc_params['bin_edges'][1:],
                            disc_params['bin_names']):
                            discretised_training_array[(training_array>lower)&(training_array<=upper)] = bin_name

                        # Now Include out of bounds values in the upper and lower bins
                        discretised_training_array[training_array<=np.min(disc_params['bin_edges'])] = disc_params['bin_names'][0]
                        discretised_training_array[training_array>=np.max(disc_params['bin_edges'])] = disc_params['bin_names'][-1]

                        # Add discretised data to a dict
                        discretised_training_dict.update({
                            rep:discretised_training_array
                        })

                        ### Use the same bins to discretise the testing_data
                        # Get testing data for this rep
                        testing_array = var_dict[var_key]['testing_data_preprocessed'][rep]

                        # Create an array of strings to make as binned data
                        discretised_testing_array = np.empty(len(testing_array)).astype(str)

                        # Using the bounds already found, discretise the data
                        for lower, upper, bin_name in zip(
                            disc_params['bin_edges'][:-1],
                            disc_params['bin_edges'][1:],
                            disc_params['bin_names']):
                            discretised_testing_array[(testing_array>lower)&(testing_array<=upper)] = bin_name

                        # Now Include out of bounds values in the upper and lower bins
                        discretised_testing_array[testing_array<=np.min(disc_params['bin_edges'])] = disc_params['bin_names'][0]
                        discretised_testing_array[testing_array>=np.max(disc_params['bin_edges'])] = disc_params['bin_names'][-1]

                        # Add discretised data to a dict
                        discretised_testing_dict.update({
                            rep:discretised_testing_array
                        })

                        # Add bin edge data to a dict
                        bin_edges_dict.update({
                            rep:np.array(disc_params['bin_edges'])
                        })

                    # Add details of bin to the var dict
                    var_dict.update({
                        var_key:{**var_dict[var_key],
                                 **{'training_data':discretised_training_dict},
                                 **{'testing_data':discretised_testing_dict},
                                 **{'bin_edges':bin_edges_dict}}
                    })

                
                # Use descritisation algorithm
                else:

                    # Get dictionary of all the iterations of the training data (bootstrapped)
                    training_data_dict = var_dict[var_key]['training_data_preprocessed']

                    # Create new dictionaries for discretised data
                    discretised_training_dict = {}
                    discretised_testing_dict = {}
                    bin_edges_dict = {}

                    # Loop over each iteration
                    for rep, training_array in training_data_dict.items():

                        # Reshape array
                        training_array = training_array.reshape(-1, 1)

                        # Discretise the data using kmeans clustering
                        # Set up discretising function
                        est = KBinsDiscretizer(n_bins=disc_params['n_bins'], encode='ordinal', strategy=disc_params['strategy'])

                        # Use the discretiser on the array and gen output
                        est.fit(training_array)
                        discretised_array = est.transform(training_array)

                        # Reshape to be x,
                        discretised_array = discretised_array.reshape(len(discretised_array),)

                        # Make array a float of integers
                        discretised_array = (discretised_array.astype(int)).astype(str)

                        # For each bin, rename them as per var_dict bin_names
                        for bin_no,bin_name in zip(np.arange(0,disc_params['n_bins'],1),disc_params['bin_names']):
                            discretised_array[discretised_array==str(int(bin_no))] = bin_name

                        # Add discretised data to a dict
                        discretised_training_dict.update({
                            rep:discretised_array
                        })

                        ### Use the same bins to discretise the testing_data

                        # Get testing data for this rep
                        testing_array = var_dict[var_key]['testing_data_preprocessed'][rep]

                        # Create an array of strings to make as binned data
                        discretised_testing_array = np.empty(len(testing_array)).astype(str)

                        # Using the bounds already found, discretise the data
                        for lower, upper, bin_name in zip(
                            est.bin_edges_[0][:-1],
                            est.bin_edges_[0][1:],
                            disc_params['bin_names']):
                            discretised_testing_array[(testing_array>lower)&(testing_array<=upper)] = bin_name

                        # Now Include out of bounds values in the upper and lower bins
                        discretised_testing_array[testing_array<=np.min(est.bin_edges_[0])] = disc_params['bin_names'][0]
                        discretised_testing_array[testing_array>=np.max(est.bin_edges_[0])] = disc_params['bin_names'][-1]

                        
                        
                        # Add discretised data to a dict
                        discretised_testing_dict.update({
                            rep:discretised_testing_array
                        })

                        # Add bin edge data to a dict
                        bin_edges_dict.update({
                            rep:est.bin_edges_[0]
                        })

                    # Add details of bin to the var dict
                    var_dict.update({
                        var_key:{**var_dict[var_key],
                                 **{'training_data':discretised_training_dict},
                                 **{'testing_data':discretised_testing_dict},
                                 **{'bin_edges':bin_edges_dict}}
                    })
                
            # Data that doesn't require discretisation
            else:
                print(var_key)
                # Add the preprocessed data as the raw data
                var_dict.update({
                    var_key:{**var_dict[var_key],
                             **{'training_data':var_dict[var_key]['training_data_preprocessed']},
                             **{'testing_data':var_dict[var_key]['testing_data_preprocessed']}
                            }
                })
                
        # Update the model dict to have the data in it
        model_dict['variables'].update(var_dict)
        
        return(model_dict)
    
    def plot_discretiser(self,model_dict,exclusion_list):
        '''
        Function for plotting histograms of the data before it was discretised (preprocessed) and the associated bins, for each repetition
        inputs:
            - model dictionary, with all variable data and metadata
        returns:
            - matplotlib.pyplot.figure
        '''
        # Get just the dictionary of variables
        var_dict = model_dict['variables']
        for key in exclusion_list:
            del var_dict[key] 
        
        # Setup figure
        fig, axes = plt.subplots(nrows=len(var_dict.keys()),ncols=2,figsize=(10,20))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 10}

        matplotlib.rc('font', **font)
        
        def single_hist(var_key,ax,one_var_dict,dset):
            '''
            Function for creating a single histogram
            '''
            
            # For each rep, create a histogram
            for rep,training_data in var_dict[var_key][dset].items():
                ax.hist(training_data,histtype='step')

            # get the ylims to use later for text
            ylims = ax.get_ylim()

            # For each bin, across the different reps, find mean and stdev in bins and add to plot
            for edge in np.arange(0,one_var_dict['discretisation']['n_bins']+1,1):
                
                edges_1rep = [one_var_dict['bin_edges'][rep][edge] for rep in one_var_dict['bin_edges'].keys()]
                edge_mean = np.mean(edges_1rep)
                edge_stdev = np.std(edges_1rep)
                ax.plot([edge_mean,edge_mean],ylims,c='k',ls='--')
                ax.errorbar([edge_mean,edge_mean],ylims,xerr=np.array([edge_stdev*2,edge_stdev*2],),capsize=2,fmt='none')
            
            # Add bin labels to plot (assume one of the bootstrapped examples represents all the configuration)
            for edge1,edge2,bin_name in \
                    zip(one_var_dict['bin_edges'][0][:-1],one_var_dict['bin_edges'][0][1:],one_var_dict['discretisation']['bin_names']):
                ax.text(np.mean([edge1,edge2]),ylims[1]*0.95,bin_name,ha='center', va='center') 

            # Label histogram axes
            ax.set_xlabel(one_var_dict['label'])
            ax.set_ylabel('Count')
                        
        # Create training histograms
        for var_key,ax in zip(var_dict.keys(),axes):
            ax = ax[0]
            one_var_dict = var_dict[var_key]
            single_hist(var_key,ax,one_var_dict,'training_data_preprocessed')
            
        # Create testing histograms
        for var_key,ax in zip(var_dict.keys(),axes):
            ax = ax[1]
            one_var_dict = var_dict[var_key]
            single_hist(var_key,ax,one_var_dict,'testing_data_preprocessed')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        
        return(fig)
        

    def save_dataset(self,model_dict,file_label):
        '''
        Function for saving the testing and training data for each bootstrap iteration as a dataframe
        input:
            - Model dictionary, which has an array for each variable for each the testing and training data
            - file_label, which is whatever details you want to include in the file label
        returns:
            - nothing, csvs has been saved
        
        '''        
        
        # For each bootstrap repitition, save a dataframe
        for rep in np.arange(0,model_dict['bootstrap_reps'],1):
            # Create a pandas dataframe to save as csv which can be read by pysmile
            df_training = pd.DataFrame(dict(zip(list(model_dict['variables'].keys()),
                              [model_dict['variables'][x]['training_data'][0] for x in model_dict['variables'].keys()]
                                 )
                             )
                        )
            # Create a pandas dataframe to save as csv which can be read by pysmile
            df_testing = pd.DataFrame(dict(zip(list(model_dict['variables'].keys()),
                              [model_dict['variables'][x]['testing_data'][0] for x in model_dict['variables'].keys()]
                                 )
                             )
                        )
    
            # Save as csv
            df_training.to_csv('training_data_{}_{}.csv'.format(file_label,rep),index=False)
            df_testing.to_csv('testing_data_{}_{}.csv'.format(file_label,rep),index=False) 
    
    def create_SM(self,model_dict,file_label):
        '''
            Function for creating bayesian submodel, one for each rep of bootstrapping.
            Right now, this funtion only works for manually setting the arcs.
            Args:
                - model dictionary, which contains all the data and the metadata
                - file_label, the location of the dataset_files
            Returns:


        '''
        # Create an empty dictionary to put the pysmile objects into per rep
        model_objects_dict = {}
        
        # Loop over each repetition of bootstrapping
        for rep in np.arange(0,model_dict['bootstrap_reps'],1):
            
            # Structure Learning using pysmile
            ds_training = pysmile.learning.DataSet()
            ds_training.read_file('training_data_{}_{}.csv'.format(file_label,rep))
            
            # Also create a testing dataset for later use
            ds_testing = pysmile.learning.DataSet()
            ds_testing.read_file('testing_data_{}_{}.csv'.format(file_label,rep))

#             # Use Bayesian Search algorithm to learn the network from the data
#             bay_search = pysmile.learning.BayesianSearch()
#             net = bay_search.learn(ds_training)

#             # Get all the node ids
#             nodes = net.get_all_nodes()

#             # Create a dictionary of all node and node ids
#             node_ids_dict = {}

#             for node, node_id in zip(net.get_all_node_ids(), nodes):
#                 node_ids_dict.update({
#                     node:node_id
#                 })

#             # Loop over every combination of the nodes and, if there is an arc, remove it
#             for node1,node2 in itertools.product(nodes,nodes):
#                 try:
#                     net.delete_arc(node1,node2)
#                 except:
#                     continue

            # Initialise new network
            net = pysmile.Network()

            # Loop over each variable and create a node
            for var_key,var_dict in model_dict['variables'].items():
                net.add_node(pysmile.NodeType.CPT, var_key)
                # Add node outcomes based on bins of discretisation
                if 'discretisation' in var_dict.keys():
                    for outcome in var_dict['discretisation']['bin_names']:
                        net.add_outcome(var_key,outcome)
                else:
                    for outcome in np.unique(var_dict['bins']):
                        net.add_outcome(var_key,outcome)
                # Delete the preexisting outcomes (state0 and state1)
                net.delete_outcome(var_key,'State0')
                net.delete_outcome(var_key,'State1')

            # Create a dictionary of all node and node ids
            node_ids_dict = {} 
            for node, node_id in zip(net.get_all_node_ids(), net.get_all_nodes()):
                node_ids_dict.update({
                    node:node_id
                })
                
            # Add arcs according to the dict provided
            for var_key,var_dict in model_dict['variables'].items():
                for child in var_dict['child_nodes']:
                    net.add_arc(node_ids_dict[var_key],node_ids_dict[child])

            # I don't know what matching does exactly
            matching = ds_training.match_network(net)

            # Get the expectation maximisation (EM) function for parameter estimation
            em = pysmile.learning.EM()

            # Learn the paramters using EM
            em.learn(data=ds_training, net=net, matching=matching)

            # Save the network
            net.write_file("SM_{}_{}.xdsl".format(file_label,rep))
            
            # Add network and dataset to a dict
            model_objects_dict.update({
                rep:{
                    'net':net,
                    'training_file':'training_data_{}_{}.csv'.format(file_label,rep),
                    'testing_file':'testing_data_{}_{}.csv'.format(file_label,rep),
                    'ds_training':ds_training,
                    'ds_testing':ds_testing
                }
            })
            
        # Update the model dictionary to have node ids in it (assume all reps have same ids)
        [var_dict.update({'id':node_ids_dict[var_key]}) for var_key,var_dict in model_dict['variables'].items()]
            
        # Add all model objects to the dict 
        model_dict.update({
            'model':model_objects_dict
        })

        return(model_dict)
    
    def get_conditional_prob_table(self,model_dict,node_handle):
        '''
        Function for displaying the conditional probability tables
        input:
            - Model dictionary with the networks for each rep in it
            - the name of the node that you want (node_handle) (str)
            
        returns:
            - pandas.DataFrame of the probabilities of the outcomes
        '''
        
        # Loop over each repetition of bootstrapping
        for rep in np.arange(0,model_dict['bootstrap_reps'],1):

            # Get the network for this rep
            net = model_dict['model'][rep]['net']
            
            # Get the node id
            node_id = net.get_node(node_handle)

            # For a given node, get the parent ids
            node_parent_ids = net.get_parents(node_id)

            # Create an empty dictionary to put all the outcomes for each parent node into
            parent_outcomes_dict = {}

            # Loop over each of the parent nodes, and put all their outcomes into a dict
            for node_parent_id in node_parent_ids:
                parent_outcomes = np.arange(0,net.get_outcome_count(node_parent_id),1)

                parent_outcomes_dict.update({
                    node_parent_id:[net.get_outcome_id(node_parent_id, x) for x in parent_outcomes]
                })

            # Find every combination of the parent nodes
            parent_combinations = \
                list(itertools.product(*(parent_outcomes_dict[Name] for Name in sorted(parent_outcomes_dict))))

            # For a given node, get all the possible outcomes
            node_outcomes = net.get_outcome_ids(node_id)

            ### Now that I have all the parent combinations, and all the outcomes from this node, find
            ### all the possible combinations for between these two

            # Create a dictdionary to put all possible parent combinations into for each node outcome
            all_outcomes_dict = {}

            # Loop over each parent node outcome combination and add this node outcomes to dict
            for parent_combination in parent_combinations:
                all_outcomes_dict.update({
                    parent_combination:node_outcomes
                })

            # Create a dataframe of all the possible outcomes
            df_all_outcomes = pd.DataFrame(all_outcomes_dict)

            # Melt the dataframe so that all the columns (where each is for a node outcome) are represented
            # by a single column, showing the current node outcome
            df_all_outcomes = df_all_outcomes.melt(var_name=[net.get_node_id(x) for x in node_parent_ids],
                                                   value_name=net.get_node_id(node_id))

            # Add conditional probabilities
            df_all_outcomes['prob'] = net.get_node_definition(node_id)

            # Get a list of the parent node names
            parent_node_names = [net.get_node_id(x) for x in parent_outcomes_dict.keys()]

            # Make each column add to one
            df_all_outcomes = df_all_outcomes.pivot(index=node_handle, columns=parent_node_names)['prob']
        
        return(df_all_outcomes)
    
    def add_evidence_to_dict(self,model_dict,evidence_dict):
        '''
        Function for adding evidence to the model dictionary
        input:
            - model dictionary, with the pysmile networks and data, but no evidence
            - dictionary with evidence in it (list in corresponding order to bins for that variable)
              each key in the dictionary must match the key in the model_dict
          returns:
            - Model dict with the evidence added
        '''
        # Add evidence to dict
        [model_dict['variables'][var_key].update({'evidence':var_evidence})
             for var_key,var_evidence in evidence_dict.items()]
        
        return(model_dict)
    
    def update_evidence(self,model_dict):
        '''
        Function for taking the evidence in the model dictionary and adding it to the network, and updating beliefs based on that
        input:
            - model_dictionary, which has the networks in it and the evidence to be applied to the network
        returns:
            - model_dictionary with the beliefs for each of the networks updated
        '''
        
        # Create a place to put the resulting probabilities into
        for var_key in model_dict['variables'].keys():
            model_dict['variables'][var_key]['resulting_probs'] = {}

        # Loop over each repetition of bootstrapping
        for rep in np.arange(0,model_dict['bootstrap_reps'],1):
            
            # Get the network for this rep
            net = model_dict['model'][rep]['net']
            
            # Clear all existing evidence
            net.clear_all_evidence()

            # For each variable, go through variable attributes dict and add evidence where is is some
            for var_key,var_params in model_dict['variables'].items():

                # Get node id
                node_id = net.get_node(var_key)

                # Set node evidence, if there is any
                if 'evidence' in model_dict['variables'][var_key].keys():

                    # Get the evidence in the same order as the they are in the network
                    evidence_dict = dict(zip(model_dict['variables'][var_key]['discretisation']['bin_names'],model_dict['variables'][var_key]['evidence']))
                    evidence_list = [evidence_dict[x] for x in net.get_outcome_ids(node_id)]

                    # Apply evidence to network
                    net.set_virtual_evidence(node_id,evidence_list)

            # With the new evidence, update your belief
            net.update_beliefs()

            # Get all the beliefs based on the evidence provided:
            for var_key in model_dict['variables'].keys():
                # Get the node id for the node result handle
                node_id = net.get_node(var_key)

                # Get the probabilities
                prob = net.get_node_value(var_key)

                # Assign to dict with each of the outcomes of the node
                prob_dict = {}
                for bin_name,prob in zip(model_dict['variables'][var_key]['discretisation']['bin_names'],prob):
                    prob_dict.update({
                        bin_name:prob
                    })
                
                # Update the network in the dictionary for this rep
                model_dict['variables'][var_key]['resulting_probs'].update({rep:prob_dict})
                
            # Update the network in the dictionary for this rep
            model_dict['model'][rep].update({'net':net})
            
        return(model_dict)
    
    def create_validator(self,pysmile_dict,node_name,rep):
        '''
        Creates a validator function for this rep. Used in other functions here
        inputs:
            - dictionary of pysmile objects
            - node_name: name of the node to be validated
        outputs:
            - validator function
        '''
        
        net = pysmile_dict['net']
        ds_testing = pysmile_dict['ds_testing']
        
        # Setup  the validator
        matching = ds_testing.match_network(net)
        validator = pysmile.learning.Validator(net, ds_testing, matching)

        # get the id of the node
        classNodeHandle = net.get_node(node_name)

        # Add total water level node to the validator
        validator.add_class_node(classNodeHandle)

        # Get the expectation maximisation (EM) function for parameter estimation
        em = pysmile.learning.EM()

        # Using expectation-maximisation, determine the accuracy using k_fold (5) cross validation
        validator.k_fold(em, 5)
        
        return(validator)
        
    def get_accuracies(self,model_dict,node_name):
        '''
        Function used for evaluating the performance of the BN. Gets the 'accuracy' for predicting a node outcome
        inputs:
            - model dictionary, with evidence set and beliefs updated and ds_testing in it
        Returns:
            - dictionary of the accuracies for predicting the outcomes of the data


        '''
        
        # Create an empty dictionary to store the accuracies
        acc_dict = {}
        
        # Loop over each rep
        for rep,pysmile_dict in model_dict['model'].items():
            
            # Get network
            net = pysmile_dict['net']
           
            # get the id of the node
            classNodeHandle = net.get_node(node_name)
        
            # Loop over each of the outcomes for this node
            for outcome in net.get_outcome_ids(classNodeHandle):
                outcome_id = net.get_outcome_ids(classNodeHandle).index(outcome)
                
                # Get the validator function
                validator = self.create_validator(pysmile_dict,node_name,rep)
                
                # Get accuracy and add to dictionary
                acc = validator.get_accuracy(classNodeHandle, outcome_id)
                acc_dict.update({
                    (rep,outcome):acc
                })
                
        return(acc_dict)
        
    def confusion_matrix(self,model_dict,node_name,rep):
        '''
        Function used for evaluating the performance of the BN across all variables using a confusion matrix for a single rep.
        Args:
            - Model dictionary, which has all the model vars in it
            - node_name: str of the name of the variable that you want to analyse with the confusion matrix
            - rep: the bootstrap iteration that you want to analyse
        Returns:
            - figure and axes matplotlib.pyplot objects


        '''
        
        # Get the network for this iteration
        net = model_dict['model'][rep]['net']
        
        # get the id of the node
        classNodeHandle = net.get_node(node_name)
        
        # Create the validator
        validator = self.create_validator(model_dict['model'][rep],node_name,rep)
        
        # Get the confusion matrix 
        confusion_matrix = validator.get_confusion_matrix(classNodeHandle)
        
        # Setup the figure
        fig, ax1 = plt.subplots(figsize=(10,10))
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 20}

        matplotlib.rc('font', **font)

        # Plot the correlation matrix of the pandas dataframe of all variables except time
        plot_output = ax1.matshow(confusion_matrix,cmap='Blues')

        ### some code that can be used for making Confusion matrix into a fraction
    #     # Create an empty dictionary to add the confusion maxtrix variables to
    #     confusion_maxtrix_frac = {}

    #     # Loop over each column of the confusion maxtrix, get it as a fraction, and add back to the data
    #     for col in np.arange(0,len(net_SM1.get_outcome_ids(2)),1):
    #         test2 = np.array(test)[col,:]
    #         test2 = test2/np.sum(test2)

    #         confusion_maxtrix_frac.update({
    #             col:test2
    #         })

        # Format axes
        ax1.set_xticks(np.arange(0,len(net.get_outcome_ids(classNodeHandle)),1))
        ax1.set_yticks(np.arange(0,len(net.get_outcome_ids(classNodeHandle)),1))
        ax1.set_xticklabels(net.get_outcome_ids(classNodeHandle))
        ax1.set_yticklabels(net.get_outcome_ids(classNodeHandle))
        plt.xticks(rotation=90)
        ax1.set_ylabel('Predicted values')
        ax1.set_xlabel('Actual values')
        ax1.xaxis.set_label_position('top')
        ax1.set_title('Confusion matrix'.format(node_name))

        # Add legend
        cb = fig.colorbar(plot_output)
        cb.ax.set_ylabel('Data points')

        return(fig, ax1)
    
    def create_BN_graph(self):
        '''
            Function that is just to set up the bayesian network graph
                kwargs:
                    - None
                returns:
                    - The graph with nothing in it
        '''
        
        # Create graph using graphviz
        graph = graphviz.Digraph(format='svg',engine="dot")#,graph_attr={'nodesep': '100','ranksep': '50'})
        
        # Return graph
        return(graph)

    def create_nodes(self,graph,model_dict,rep):
        '''
            Function that adds nodes to the graphviz object, with evidence/beliefs graphs
                kwargs:
                    - graph: graphviz.dot.Digraph object with nothing in it
                    - model_dict: the dictionary of model parameters, which has had evidence set already
                    - rep: the number of the bootstrap result that you want to plot
                returns:
                    - The graph object with the nodes added (in no particular spot)        
        '''
        # Create a node for each variable
        for var_key in model_dict['variables'].keys():
            
            # For this rep, get the resulting probabilities
            df_probs =  pd.DataFrame.from_dict(model_dict['variables'][var_key]['resulting_probs'][rep],orient='index',columns=['prob'])
            
            # Setup the figure to go in node
            fig, ax = plt.subplots(figsize=(7,7))
            font = {'family' : 'DejaVu Sans',
                    'size'   : 20}
            matplotlib.rc('font', **font)

            # Try to plot the evidence (if there has been any set); if not, get the resulting distributions
            try:
                ax.barh(y=np.arange(0,len(model_dict['variables'][var_key]['evidence']),1),
                        width=np.array(model_dict['variables'][var_key]['evidence']),
                        tick_label=model_dict['variables'][var_key]['discretisation']['bin_names'],
                        facecolor ='g',
                        linewidth=2)
            except:
                ax.barh(y=np.arange(0,len(df_probs),1),
                        width=np.array(df_probs.prob),
                        tick_label=df_probs.index,
                        facecolor='k',
                        edgecolor=None)

            # Format axes
            ax.set_xlabel('Probability')
            ax.set_title(var_key)
            
            # Delete any existing graphs
            for fl in glob.glob('{}_key*_graph.jpeg'.format(var_key)):
                #Do what you want with the file
                os.remove(fl)
            
            # Save the figure (need random key because otherwise it will not pull in new graph each time this is run, it will pull from
            # temp memory, and so the graphs won't update)
            random = np.random.randint(1000)
            fig.savefig('{}_key{}_graph.jpeg'.format(var_key,random))
            plt.close()
            
            # Add details of bin to the var dict
            model_dict['variables'][var_key].update({'figure':'{}_key{}_graph.jpeg'.format(var_key,random)})
        
        # Loop over each node (name and ID) and add node to graph
        for var_key,var_dict in model_dict['variables'].items():
            node_id = var_dict['id']
            graph.attr(size='18,15',pad='0.5',fonsize='20')
            graph.node(str(node_id),shape='rectangle', label=var_dict['label'])#"",
#                        image=model_dict['variables'][var_key]['figure'],fontsize='20')
            
        # Return the graph with the nodes created
        return(graph,model_dict)
    
    def create_arcs(self,graph,model_dict):
        '''
            Function that create arcs and adds them to the existing graph using the dataset and dataframe
            kwargs:
                - graph: A graphviz.dot.Digraph, which has nodes in it
                - model_dict: Python dictionary, that has all the model metadata in it
            returns:
                - The graphviz.dot.Digraph object with arcs added.
        '''

        # Loop over each variable and get node id
        for var_key,var_dict in model_dict['variables'].items():
            node_id = var_dict['id']
            # For each var, loop over all child nodes and get ids. Plot the arcs based on this
            for child in var_dict['child_nodes']:
                child_id = model_dict['variables'][child]['id']
                graph.edge(str(node_id),str(child_id))

        # Return the graph with edges
        return(graph)
    
    
    def univariant_sensitivity(self,model_dict,input_node,input_outcome,output_node):
        '''
        used for varying the prior distribution of a single variable, updating the network, and calculating the posterior distribution for another node
        inputs:
            -
        returns:
            -
        '''
        
        # Figure setup
        fig = plt.figure(figsize=(10,10))
        plt.subplots_adjust(hspace=0.2)
        ax1 = plt.subplot2grid((2,1),(0,0))
        ax2 = plt.subplot2grid((2,1),(1,0))
        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 15}
        matplotlib.rc('font', **font)

        # Create colourmap for the figure
        cm = pylab.get_cmap('coolwarm')
        
        # Create an empty dict to put posterior dfs into for each rep
        posterior_dfs_dict = {}
        
        # Setup legend
        legend_lines = [] #ax2
        
        # Repeat for each iteration
        for rep in model_dict['model'].keys():
        
            # Get the network for this rep
            net = model_dict['model'][rep]['net']

            # Find the outcomes and the outcome indecies for input and output nodes
            input_outcomes = net.get_outcome_ids(input_node)
            input_outcome_index = input_outcomes.index(input_outcome)
            output_outcomes = net.get_outcome_ids(output_node)

            # Create an empty dictionary of posterior probabilities
            posterior_prob_dicts = {}

            # Define characteristics of the prior distribution
            a = 0 # skewness
            mean = input_outcome_index  # mean should be the index outcome that I want to test; the largest part of the PDF
            sigma_list = [0.4,0.8,1,2,5,100]

            # Loop over each of the sigmas for a different PDF
            for sigma in sigma_list:

                # Create a list where each value represents a single node outcome in the posterior distribution 
                x_list = [x for x in range(len(output_outcomes))]

                # Calculate pdf (see wiki for forumla)
                low_phi = [(np.exp(1)**(-0.5*((x-mean)/sigma)**2.0))/(sigma*np.sqrt(sigma*np.pi)) for x in x_list]
                cap_phi = [0.5*(1+math.erf(a*x/np.sqrt(2))) for x in x_list] 
                f_x = [2*low*cap for low,cap in zip(low_phi,cap_phi)] # pdf

                # normalise
                f_x = f_x/np.sum(f_x)

                # create an RGBA tuple from colormap for this line
                color = cm(1.*sigma_list.index(sigma)/len(sigma_list)) 

#                 # Plot the prior distribution
#                 ax1.plot(x_list,f_x,c=color)

                # Smoothing the prior
                bspl = splrep(x_list,f_x, s=0)
                
                # add points for each bin value
                ax1.scatter(x_list,f_x,c=[color]*len(x_list))

                #values for the x axis
                x_smooth = np.linspace(min(x_list), max(x_list), 1000)

                #get y values from interpolated curve
                bspl_y = splev(x_smooth, bspl)

                #plot interpolated curve of the posterior distribution
                ax1.plot(x_smooth, bspl_y, c=color)
                
                # Make sure there is no existing evidence in the network
                net.clear_all_evidence()

                # Set evidence (prior distribution)
                net.set_virtual_evidence(net.get_node(input_node),f_x)

                # Calculate posterior probs
                net.update_beliefs()

                # Get the posterior probs
                posterior_probs = net.get_node_value(output_node)

                # Add to dict for later
                posterior_prob_dicts.update({
                    sigma:posterior_probs
                })

                # Smoothing the results ############PROBS DONT ACTUALLY WANT TO SMOOTH IT
                list_x = np.arange(0,len(output_outcomes),1)
                bspl = splrep(list_x,posterior_probs, s=0)
                
                # add points for each bin value
                ax2.scatter(list_x,posterior_probs,c=[color]*len(list_x))

                #values for the x axis
                x_smooth = np.linspace(min(list_x), max(list_x), 1000)

                #get y values from interpolated curve
                bspl_y = splev(x_smooth, bspl)

                #plot interpolated curve of the posterior distribution
                ax2.plot(x_smooth, bspl_y, c=color)

                # Add color to legend
                if rep==list(model_dict['model'].keys())[0]:
                    legend_lines.append(matplotlib.lines.Line2D([0],[0],c=color))

            # Create dataframe of posteriors to return
            df_posterior = pd.DataFrame.from_dict(posterior_prob_dicts)
            
            # Add posterior distributions dict for each rep to a dict
            posterior_dfs_dict.update({
                rep:df_posterior
            })

        ##############################################################

        input_label = model_dict['variables'][input_node]['label']
        output_label = model_dict['variables'][output_node]['label']
        legend_labels = ['$\sigma$: {}'.format(x) for x in sigma_list]

        
        ax1.set_title("Sensitivty analysis")
        ax1.set_xlabel(input_label)
        ax1.set_ylabel('Prior probabilities')
        ax1.set_xticks(np.arange(0,len(input_outcomes),1))
        ax1.set_xticklabels(input_outcomes)
        ax1.set_xlim(np.min(x_list),np.max(x_list))

        ax2.set_ylim(0.0,1.0)
        ax2.set_ylabel('Posterior probability')
        ax2.set_xticks(np.arange(0,len(output_outcomes),1))
        ax2.set_xticklabels(output_outcomes)
        ax2.set_xlabel("{}".format(output_label))
        ax2.legend(handles=legend_lines,labels=legend_labels,bbox_to_anchor=(1.3, 1.0),
                 title="Prior distribution")

        matplotlib.rc('font', **font)

        plt.show()

        return(posterior_dfs_dict,fig)
        