# # Preprocess the data and save as a new csv that can be read by pysmile
# Create the network without considering time
df_Pto_lagoon = df_Pto_lagoon[all_vars_lagoon]

# Drop the variable that I created called that removed tide
df_Pto_lagoon = df_Pto_lagoon.drop('TWL_point_110_less_Tide',axis=1)

# Discretise the data using kmeans clustering
# Set up discretising function
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

# Turn dataframe into array for the discretiser
pto_lagoon_array = np.array(df_Pto_lagoon[list(df_Pto_lagoon)])

# Use the discretiser on the array and gen output
est.fit(pto_lagoon_array)
pto_lagoon_disc_array = est.transform(pto_lagoon_array)

# Turn the resulting discretised array back to a dataframe
df_pto_lagoon_disc = pd.DataFrame(pto_lagoon_disc_array).astype(str)

# Relabel the columns of the new dataframe
df_pto_lagoon_disc.columns = list(df_Pto_lagoon)
    
# Save back as csv
df_pto_lagoon_disc.to_csv('../Data_files/Pto_218_lagoonside_preprocessed.csv',index=False)

# create a network from the old network (so that I don't have to make nodes again..)
net_lagoon_manual = net_lagoon

# Get all the node ids
all_nodes_lagoon = net_lagoon_manual.get_all_nodes()

# Loop over every combination of the nodes and, if there is an arc, remove it
for node1,node2 in itertools.product(all_nodes_lagoon,all_nodes_lagoon):
    try:
        net_lagoon_manual.delete_arc(node1,node2)
    except:
        continue
        
lagoon_node_ids = {}

for node, node_id in zip(net_lagoon_manual.get_all_node_ids(), all_nodes_lagoon):
    lagoon_node_ids.update({
        node:node_id
    })


# Add the arcs that I think should exist
net_lagoon_manual.add_arc(lagoon_node_ids['Tide'],lagoon_node_ids['TWL_point_110']) 
net_lagoon_manual.add_arc(lagoon_node_ids['MSL'],lagoon_node_ids['TWL_point_110'])
net_lagoon_manual.add_arc(lagoon_node_ids['Hs_point_110'],lagoon_node_ids['TWL_point_110'])
net_lagoon_manual.add_arc(lagoon_node_ids['Tm_offshore'],lagoon_node_ids['Hs_point_110'])
net_lagoon_manual.add_arc(lagoon_node_ids['Hs_offshore'],lagoon_node_ids['Hs_point_110'])
net_lagoon_manual.add_arc(lagoon_node_ids['Wind'],lagoon_node_ids['Hs_offshore'])
net_lagoon_manual.add_arc(lagoon_node_ids['Wind'],lagoon_node_ids['Tm_offshore'])
net_lagoon_manual.add_arc(lagoon_node_ids['Dir_offshore'],lagoon_node_ids['Hs_point_110'])
net_lagoon_manual.add_arc(lagoon_node_ids['WindDir'],lagoon_node_ids['Hs_offshore'])
net_lagoon_manual.add_arc(lagoon_node_ids['WindDir'],lagoon_node_ids['Tm_offshore'])
net_lagoon_manual.add_arc(lagoon_node_ids['Wind'],lagoon_node_ids['Dir_offshore'])
net_lagoon_manual.add_arc(lagoon_node_ids['WindDir'],lagoon_node_ids['Dir_offshore'])

# Plot the graph  
graph_lagoon_manual = BN_visualisation(ds_lagoon,df_Pto_lagoon,net_lagoon_manual).graph
print('Lagoon Side:')
graph_lagoon_manual

## For lagoon side
# Setup  the validator
matching_lagoon = ds_lagoon.match_network(net_lagoon)
validator_lagoon = pysmile.learning.Validator(net_lagoon, ds_lagoon, matching_lagoon)

# Validate the results on the total water level node 
classNodeHandle_lagoon = net_lagoon.get_node("TWL_point_110")

# Add total water level node to the validator
validator_lagoon.add_class_node(classNodeHandle_lagoon)

# Get the expectation maximisation (EM) function for parameter estimation
em = pysmile.learning.EM()

# Using expectation-maximisation, determine the accuracy using k_fold (5) cross validation
validator_lagoon.k_fold(em, 5)
acc_lagoon = validator_lagoon.get_accuracy(classNodeHandle_lagoon, 0)

print("Accuracy for predicting TWL, lagoon side: ",acc_lagoon)

# Learn the paramters using EM
em.learn(data=ds_lagoon, net=net_lagoon, matching=matching_lagoon)

# Save the network
net_lagoon.write_file("../BN_model_files/Pto_218_lagoonside_learned_network.xdsl")

print('MSL id:',net_lagoon.get_node('MSL'))
print('TWL_point_110 id:',net_lagoon.get_node('TWL_point_110'))

net_lagoon.clear_all_evidence()
net_lagoon.set_virtual_evidence(2,[0,1,0,0])
print(net_lagoon.get_virtual_evidence(2))
net_lagoon.update_beliefs()

print(net_lagoon.get_node_value(8))

print('Tide id:',net_lagoon.get_node('Tide'))
print('TWL_point_110 id:',net_lagoon.get_node('TWL_point_110'))

net_lagoon.clear_all_evidence()
net_lagoon.set_virtual_evidence(5,[0,0,1,0])
# print(net_lagoon.get_virtual_evidence(2))
net_lagoon.update_beliefs()

print(net_lagoon.get_node_value(8))
