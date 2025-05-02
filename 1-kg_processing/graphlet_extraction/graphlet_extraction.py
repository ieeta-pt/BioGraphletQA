import os
os.environ["OMP_WAIT_POLICY"] = "active"
import pandas as pd
import graph_tool.all as gt
import networkx as nx
import json
from tqdm import tqdm
from collections import defaultdict
import random

random.seed(42)



PATH_TO_CONSTANTS = "../../"
with open(PATH_TO_CONSTANTS+'constants.json') as f:
    CONSTANTS = json.load(f)



# First we get the NX graph
nodes_file = PATH_TO_CONSTANTS+CONSTANTS['cleaned_nodes_csv']  # Replace with your nodes CSV file path
edges_file = PATH_TO_CONSTANTS+CONSTANTS['edges_csv']  # Replace with your edges CSV file path
# Load nodes and edges into DataFrames
nodes_df = pd.read_csv(nodes_file)
edges_df = pd.read_csv(edges_file)
edges_df = edges_df[edges_df['type'] != 'has_code'] #drop the has_code stuff
# Create a directed graph
KG = nx.DiGraph()
# Add nodes with attributes
for _, row in (nodes_df.iterrows()):
    node_id = row['id']  # Replace with the appropriate column name for node ID
    attributes = row.to_dict()
    KG.add_node(node_id, **attributes)    
# Add edges with attributes
for _, row in (edges_df.iterrows()):
    source = row['source_id']  # Replace with the appropriate column name for source node
    target = row['target_id']  # Replace with the appropriate column name for target node
    attributes = row.to_dict()
    KG.add_edge(source, target, **attributes)
nodes_to_remove = [n for n, d in KG.nodes(data=True) if pd.isna(d.get('name'))]

print(f"NetworkX KG: {len(KG.nodes)} nodes, {len(KG.edges)} edges")


#now we load the GT graph as undirected

# Create a undirected graph
G = gt.Graph(directed=False)

# Create a mapping from node IDs to graph vertices
node_map = {}
node_id_prop = G.new_vertex_property("string")
node_name_prop = G.new_vertex_property("string")

for _, row in nodes_df.iterrows():  # Assuming 'id' and 'name' are columns containing node identifiers and names
    node_id, node_name = row['id'], row['name']
    if pd.notna(node_name):  # Only add nodes whose names are not NaN
        v = G.add_vertex()
        node_map[node_id] = v
        node_id_prop[v] = node_id
        node_name_prop[v] = node_name

G.vp['id'] = node_id_prop  # Store node IDs as a property
G.vp['name'] = node_name_prop  # Store node names as a property

# Add edges
type_prop = G.new_edge_property("string")

for _, row in edges_df.iterrows():
    src, tgt, edge_type = row['source_id'], row['target_id'], row['type']
    if src in node_map and tgt in node_map:
        e = G.add_edge(node_map[src], node_map[tgt])
        type_prop[e] = edge_type

G.ep['type'] = type_prop  # Store edge types as a property

gt.remove_parallel_edges(G) # remove edges that go two ways

print(f"GT KG: {G}")

nodes_to_remove = [v for v in G.vertices() if v.out_degree()+ v.in_degree() < 3 or v.out_degree()+ v.in_degree() > 100]

# Remove the selected nodes
G.set_vertex_filter(None)  # Ensure no previous filters are applied
G.remove_vertex(nodes_to_remove, fast=True)  # 'fast=True' for efficiency

print(f"Graph now has {G}")





# step one is to calculate the motifs of all, for each k
GRAPHLET_SIZES = [3,4,5]
TARGET = 10_000

print("starting initial graphlet size fetch")
all_motifs = {}
all_counts = {}
for i in tqdm(GRAPHLET_SIZES):
    motifs, counts = gt.motifs(G, p=1.0, k = i)
    all_motifs[i] = motifs
    all_counts[i] = counts
print("step 1 done: Graphlet fetch")

#step 2 is to get the target ratios:
count_ratio = {}
for k,v in all_counts.items():
    count_ratio[k] = [min(1.0,TARGET/i) for i in v]
    
print("step 2 done: Ratio calculation")

#step 3 is to do to scale:

final_motifs = defaultdict(lambda: list())
final_counts = defaultdict(lambda: list())
final_maps = defaultdict(lambda: list())
for k,v in tqdm(count_ratio.items()):
    for graphlet, ratio in zip(all_motifs[k], count_ratio[k]):
        motifs, counts, maps = gt.motifs(G, p=ratio, k = k, motif_list=[graphlet], return_maps=True)
        final_motifs[k].extend(motifs)
        final_counts[k].extend(counts)
        final_maps[k].extend(maps)

print("step 3 done: Graphlet Extraction")


# prepare mappings for graphlet_ids
graplets_path = PATH_TO_CONSTANTS+CONSTANTS['graphlets']
with open(graplets_path, "r") as file:
    templated_graphlets = json.load(file)
graphlet_numbered = {}
for size, graphs in templated_graphlets.items():
    for graph_data in graphs:
        _graphlet = gt.Graph(graph_data["edges"], directed=False)
        graphlet_numbered[graph_data['id']] = _graphlet


#motif_maps and metdata
motif_to_id_map = {}
metadata = {}
for n_nodes, motif_list in final_motifs.items():
    for i, motif in enumerate(motif_list):
        for g_id,_graphlet in graphlet_numbered.items():
            if gt.isomorphism(_graphlet, motif ):
                motif_to_id_map[motif] = g_id
                metadata[g_id] = {"counts":all_counts[n_nodes][i], "ratio":count_ratio[n_nodes][i], "final_counts":final_counts[n_nodes][i] }
                


# now we need to figure out how to store

outs = []
graphlet_id = -1
for n_nodes, graphlet_data in final_maps.items():
    for graphlet_index, graphlet_list in enumerate(graphlet_data):
        #find the correct g_id        
        graphlet_id = motif_to_id_map[final_motifs[n_nodes][graphlet_index]]
        for vertex_property_map in graphlet_list: 
            graphlet_vertex_list = []
            node_names =[]
            node_text = []
            
            id_v_map = {}
            v_name_map = {}
            
            for v in sorted(final_motifs[n_nodes][graphlet_index].vertices()):
                value = vertex_property_map[v]  
                v_name = G.vertex_properties['name'][value]
                v_id = G.vertex_properties['id'][value]
                
                
                v_name_map[v] = v_name
                id_v_map[v_id] = int(v)
                
                node_names.append(v_name)
                graphlet_vertex_list.append(v_id)
                node_text.append(f" - {(int(v))}: {v_name} \n")
                
            sub_graph = KG.subgraph(graphlet_vertex_list)
            edge_labels = [data.get("type") for u, v, data in sub_graph.edges(data=True)]
            edges = [(id_v_map[i[0]], id_v_map[i[1]]) for i in list(sub_graph.edges)]
            
            
            outs.append({"nodes":graphlet_vertex_list , 
                         "edges": edges, 
                         "graphlet_id": graphlet_id,
                         "edge_labels": edge_labels,
                         "node_names": node_names, 
                         "graphlet_text": node_text,
                         "num_nodes": n_nodes,
                         "graphlet_index": graphlet_index
                         })

random.shuffle(outs)
data_with_ids = [{"id": i, **item, } for i, item in enumerate(outs)]

# Save to a JSONL file
with open(PATH_TO_CONSTANTS+CONSTANTS['templates'], "w") as file:
    for item in data_with_ids:
        file.write(json.dumps(item) + "\n")
        
with open(PATH_TO_CONSTANTS+CONSTANTS['templates_metadata'], 'w') as fp:
    json.dump(metadata, fp)
#Prep metada 

