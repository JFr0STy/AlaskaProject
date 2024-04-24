#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jeremiah Brown 
"""
# %%
import pandas as pd
import geopandas as gpd
import maup
import time
import networkx as nx
from pyproj import CRS
import pickle
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.proposals import recom
from gerrychain.constraints import contiguous
from gerrychain.updaters import Tally, cut_edges
from functools import partial 
import matplotlib.pyplot as plt
import networkx as nx
from shapely.ops import nearest_points
maup.progress.enabled = True


#population

start_time = time.time()
population_df = gpd.read_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/AlaskaBlockPL94-171-2020table/ak_pl2020_p2_b.shp")

# Pickle the GeoDataFrame
with open('population.pkl', 'wb') as pickle_file:
    pickle.dump(population_df, pickle_file)

# Load the GeoDataFrame from the pickle file
with open('population.pkl', 'rb') as pickle_file:
    loaded_population_df = pickle.load(pickle_file)
end_time = time.time()
print("The time to import ak_pl2020_p2_b.shp is:",
      (end_time-start_time)/60, "mins")

#2020 election
start_time = time.time()
election_df = gpd.read_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/VEST2020AlaskaPrecinctAndElectionResults/ak_vest_20.shp")
end_time = time.time()
print("The time to import ak_vest_20.shp is:",
      (end_time-start_time)/60, "mins")

#congressional district data
start_time = time.time()
district_df = gpd.read_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/2022AlaskaStateSenateApprovedInterimPlan/ak_sldu_adopted_2022.shp")
end_time = time.time()
print(district_df.geometry)
print("The time to import ak_sldu_adopted_2022.shp is:",
      (end_time-start_time)/60, "mins")

print(population_df.columns)
print(election_df.columns)
print(district_df.columns)
print(election_df)



election_df = election_df.to_crs(election_df.estimate_utm_crs())
population_df = population_df.to_crs(population_df.estimate_utm_crs())
district_df = district_df.to_crs(district_df.estimate_utm_crs())
print("Starting repair")
print("finished repair")



print(population_df.crs,"    ", election_df.crs, "     ", district_df.crs)

print("hasattr")
print(hasattr(population_df, 'geometry'))
print(hasattr(election_df, 'geometry'))

      
print(population_df.crs,"    ", election_df.crs, "     ", district_df.crs)

print("start assign")
blocks_to_precincts_assignment = maup.assign(population_df.geometry, election_df.geometry)

pop_column_names = ['P0020001']

for name in pop_column_names:
    election_df[name] = population_df[name].groupby(blocks_to_precincts_assignment).sum()

print(population_df['P0020001'].sum())
print(election_df['P0020001'].sum())

print("start second assign")
precincts_to_districts_assignment = maup.assign(election_df.geometry, district_df.geometry)
print("finish second assign")
election_df["SD"] = precincts_to_districts_assignment

print("COLS",election_df.columns)

district_col_name = "SEN_DIST"

print(set(election_df["SD"]))
for precinct_index in range(len(election_df)):
    election_df.at[precinct_index, "SD"] = district_df.at[election_df.at[precinct_index, "SD"], district_col_name]
print(set(district_df[district_col_name]))
print(set(election_df["SD"]))

print(list(election_df.columns))

rename_dict = {'P0020001': 'TOTPOP', 'G20PREDBID': 'G20PRED', 'G20PRERTRU': 'G20PRER'}

election_df.rename(columns=rename_dict, inplace = True)
election_df.drop(columns=['G20PRELJOR', 'G20PREGJAN', 'G20PRECBLA', 'G20PREIPIE',
'G20PREOFUE', 'G20USSRSUL', 'G20USSDGRO', 'G20USSOHOW', 'G20HALRYOU',
'G20HALDGAL'], inplace=True)

election_df.plot()

print(election_df.loc[election_df["SD"] == 1, "TOTPOP"].sum())
print(election_df.loc[election_df["SD"] == 2, "TOTPOP"].sum())
pop_vals = [election_df.loc[election_df["SD"] == n, "TOTPOP"].sum() for n in range(1, 18)]
print(pop_vals)

election_df.to_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/AK.shp")
shp_file = gpd.read_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/AK.shp")
shp_file.to_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/AK.geojson", driver="GeoJSON")
# %%
# Prepares Data for GerryChain
graph = Graph.from_file("/Users/frostyfizzle/CS-Redistricting/ProjectCode/AK.shp")




print("Before operation results")
#Veomett code 
components = list(nx.connected_components(graph))
for component in components:
    print("length of this component is ", len(component))
    if len(component) < 5:
        for node in component:
            print(node)


# Converts the GerryChain Graph to a NetworkX graph for manipulation
nx_graph = nx.Graph(graph)

# Find the nearest node in terms of geographic distance between polygons
def find_nearest_node(current_node, node_list, nx_graph):
    current_polygon = nx_graph.nodes[current_node]['geometry']
    closest_node, min_dist = None, float('inf')
    for node in node_list:
        if node != current_node and 'geometry' in nx_graph.nodes[node]:
            other_polygon = nx_graph.nodes[node]['geometry']
            # Use nearest_points to find the closest points and calculate the distance between them
            point1, point2 = nearest_points(current_polygon, other_polygon)
            dist = point1.distance(point2)  # Returns the distance in units of the coordinate system
            if dist < min_dist:
                closest_node, min_dist = node, dist
    return closest_node

# Connect each isolated node to the nearest node in the graph
for component in components:
    if len(component) < 5:
        for node in component:
            # Exclude nodes from the same small component
            potential_targets = [n for n in nx_graph.nodes() if n not in component]
            nearest_node = find_nearest_node(node, potential_targets, nx_graph)
            if nearest_node is not None:
                nx_graph.add_edge(node, nearest_node)

print("After operation results") 
updated_graph = Graph(nx_graph)
components = list(nx.connected_components(updated_graph))
for component in components:
    print("length of this component is ", len(component))
    if len(component) < 5:
        for node in component:
            print(node)

# Removing isolated nodes from the graph
for component in components:
    if len(component) < 5:
        updated_graph.remove_nodes_from(component)


print("After removing any isolated nodes")
components = list(nx.connected_components(updated_graph))
for component in components:
    print("length of this component is ", len(component))
    if len(component) < 5:
        for node in component:
            print(node)

pos = nx.spring_layout(updated_graph)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(updated_graph, pos, node_size=50)

# edges
nx.draw_networkx_edges(updated_graph, pos, width=1)

plt.title("Graph Representation with Connected Nodes")
plt.show()


#Checking for isolated nodes
isolated_nodes = [node for node in updated_graph.nodes if updated_graph.degree(node) == 0]
print("Isolated nodes:", isolated_nodes)

#Printing 5 nodes data
for node, data in list(updated_graph.nodes(data=True))[:5]:
    print(node, data)

updaters = {"population": Tally("TOTPOP"), "cut_edges": cut_edges}
initial_partition = Partition(updated_graph, assignment="DISTRICT", updaters=updaters)
total_population = sum(updated_graph.nodes[node]['TOTPOP'] for node in updated_graph.nodes())
number_of_districts = len(set(data['DISTRICT'] for node, data in updated_graph.nodes(data=True)))
population_target = total_population / number_of_districts
epsilon = 0.05  # Allow districts to vary by up to 5%
'''
configured_recom = partial(
    recom,
    pop_col="TOTPOP",
    pop_target=population_target,
    epsilon=epsilon
)




# Create the Markov Chain 
chain = MarkovChain(
    proposal=configured_recom,
    constraints=[contiguous],
    accept=lambda partition: True,
    initial_state=initial_partition,
    total_steps=10000
)


try:
    for step, partition in enumerate(chain):
        print(f"Step {step}: Population = {sum(partition['population'].values())}, Cut Edges = {len(partition['cut_edges'])}")
except IndexError as e:
    print("Failed at step:", step)
    print("Current partition details:", partition)
'''
#Debugging
def diagnostic_recom(partition):
    print("Attempting recom with the following partition state:")
    print(f"Number of districts: {len(set(partition.assignment.values()))}")
    print(f"Population stats: {sum(partition['population'].values())}")
    try:
        # Attempt to execute the recom function
        return recom(partition, pop_col='TOTPOP', pop_target=population_target, epsilon=0.10)
    except Exception as e:
        print("Recom failed with error:", str(e))
        # Log current graph edges and nodes
        print("Edges available:", list(partition.graph.edges()))
        print("Nodes detail:")
        for node in partition.graph.nodes(data=True):
            print(node)
        raise

# Initialize the Markov Chain with diagnostic recom
chain = MarkovChain(
    proposal=diagnostic_recom,
    constraints=[contiguous],
    accept=lambda partition: True,
    initial_state=initial_partition,
    total_steps=1  # Start with one step for testing
)

# Execute one step to see output
try:
    next(chain)
except Exception as e:
    print("Chain failed with error:", str(e))
end_time = time.time()
print("Total time: ",
      (end_time-start_time)/60, "mins")

# %%
