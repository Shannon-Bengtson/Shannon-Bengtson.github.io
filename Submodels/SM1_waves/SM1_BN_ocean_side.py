# # Preprocess the data and save as a new csv that can be read by pysmile
# Create the network without considering time
df_Pto_ocean = df_Pto_ocean[all_vars_ocean]

# Drop the variable that I created called that removed tide
df_Pto_ocean = df_Pto_ocean.drop('TWL_point_218_less_Tide',axis=1)

# Discretise the data using kmeans clustering
# Set up discretising function
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')

# Turn dataframe into array for the discretiser
pto_ocean_array = np.array(df_Pto_ocean[list(df_Pto_ocean)])

# Use the discretiser on the array and gen output
est.fit(pto_ocean_array)
pto_ocean_disc_array = est.transform(pto_ocean_array)

# Turn the resulting discretised array back to a dataframe
df_pto_ocean_disc = pd.DataFrame(pto_ocean_disc_array).astype(str)

# Relabel the columns of the new dataframe
df_pto_ocean_disc.columns = list(df_Pto_ocean)
    
# Save back as csv
df_pto_ocean_disc.to_csv('../Data_files/Pto_218_oceanside_preprocessed.csv',index=False)

# create a network from the old network (so that I don't have to make nodes again..)
net_ocean_manual = net_ocean

# Get all the node ids
all_nodes_ocean = net_ocean_manual.get_all_nodes()

# Loop over every combination of the nodes and, if there is an arc, remove it
for node1,node2 in itertools.product(all_nodes_ocean,all_nodes_ocean):
    try:
        net_ocean_manual.delete_arc(node1,node2)
    except:
        continue

# Add the arcs that I think should exist
net_ocean_manual.add_arc(0,4) # Tide -> TWL_point_218
net_ocean_manual.add_arc(1,4) # MSL -> TWL_point_218
net_ocean_manual.add_arc(2,4) # Hs_point_218 -> TWL_point_218
net_ocean_manual.add_arc(3,4) # Tm_point_218 -> TWL_point_218

# Plot the graph
graph_ocean_manual = BN_visualisation(ds_ocean,df_Pto_ocean,net_ocean_manual).graph
print('Ocean side:')
graph_ocean_manual

## For ocean side
# Setup  the validator
matching_ocean = ds_ocean.match_network(net_ocean)
validator_ocean = pysmile.learning.Validator(net_ocean, ds_ocean, matching_ocean)

# Validate the results on the total water level node 
classNodeHandle_ocean = net_ocean.get_node("TWL_point_218")

# Add total water level node to the validator
validator_ocean.add_class_node(classNodeHandle_ocean)

# Get the expectation maximisation (EM) function for parameter estimation
em = pysmile.learning.EM()

# Using expectation-maximisation, determine the accuracy using k_fold (5) cross validation
validator_ocean.k_fold(em, 5)
acc_ocean = validator_ocean.get_accuracy(classNodeHandle_ocean, 0)

print("Accuracy for predicting TWL, ocean side: ",acc_ocean)

# Learn the paramters using EM
em.learn(data=ds_ocean, net=net_ocean, matching=matching_ocean)

# Save the network
net_ocean.write_file("../BN_model_files/Pto_218_oceanside_learned_network.xdsl")