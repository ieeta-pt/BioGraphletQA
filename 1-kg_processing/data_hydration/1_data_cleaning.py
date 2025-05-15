import pandas as pd
import os
import json

def print_function(df, _type):
    print(f"Length of the set of names: {len(df.name.unique())}, Total names: {(df.name.count())}, Names Missing: {(df['name'].isna().sum())}, Total {len(df)}")
    print(f"For {_type}")
    _df=df[df['type'] == _type]
    print(f"Length of the set of names: {len(_df.name.unique())}, Total names: {(_df.name.count())}, Names Missing: {(_df['name'].isna().sum())}, Total {len(_df)}")


def from_text(df, name, _type):
    df['name'] = df.apply(
        lambda row: row['name']  # Keep existing name if it exists
        if row['name'] != None or row['type'] != _type 
        else (
            row['properties_dict'].get(name, None)
        ),
        axis=1
    )
    print_function(df, _type)
    return df

def from_dict(df,file_path,code_name, _type):    
    with open(file_path, "r") as file:
        small_dict = json.load(file)

    df['name'] = df.apply(
        lambda row: row['name']  # Keep existing name if it exists
        if row['name'] != None or row['type'] != _type 
        else (
            small_dict.get(
            row['properties_dict'].get(code_name, '').split(';')[0]  # Handle missing UNIPROTKB safely
            if row['properties_dict'].get(code_name) else None, # Check for None
            None  # Default if not in uniprot_dict
        )
        ),
        axis=1
    )
    print_function(df, _type)
    return df

def from_csv(df,file_path,code_name, id_field, name_field, _type, tsv=False, id_as_string=False):
    if tsv:
        small_df = pd.read_csv(file_path, sep='\t')
    else:
        small_df = pd.read_csv(file_path)
    
    if id_as_string:
        small_df[id_field] =  small_df[id_field].astype(str)
    small_dict = dict(zip(small_df[id_field], small_df[name_field]))

    df['name'] = df.apply(
        lambda row: row['name']  # Keep existing name if it exists
        if row['name'] != None or row['type'] != _type 
        else (
            small_dict.get(
            row['properties_dict'].get(code_name, '').split(';')[0]  # Handle missing UNIPROTKB safely
            if row['properties_dict'].get(code_name) else None, # Check for None
            None  # Default if not in uniprot_dict
        )
        ),
        axis=1
    )
    
    print_function(df, _type)
    
    return df

def extract_codes(df, compound_name, out_name):
    _set = set(df['properties_dict'].apply(lambda x: x.get(compound_name) if isinstance(x, dict) else None).dropna())
    with open(out_name, 'w') as file:
        for value in _set:
            file.write(f"{value}\n")
            



PATH_TO_CONSTANTS = "../../"

with open(PATH_TO_CONSTANTS +"constants.json") as f:
    CONSTANTS = json.load(f)



file_path = PATH_TO_CONSTANTS+CONSTANTS['nodes_csv']
kg_nodes_df = pd.read_csv(file_path)

kg_nodes_df['properties_dict'] = kg_nodes_df['properties'].apply(json.loads)

kg_nodes_df['name'] = None

# kg_nodes_df.type.unique()
# compound', 'code', 'protein', 'molecule', 'activity', 'effect', 'gene', 'disease', 'phenotype', 'pathway', 'indication', 'side_effect'


for _type in kg_nodes_df.type.unique():
    all_keys = set()
    for properties in kg_nodes_df[kg_nodes_df.type ==_type].properties_dict:
        all_keys.update(properties.keys())
    print(f"{_type}:{all_keys}")
        


# Define configurations for each dataset
datasets = {
    'compound':{
        'WIKIPEDIA':{ 'function': 'from_text', 'args': {}},
        'SIDER':{ 'function': 'from_csv', 'file_path': 'hydrate_names/CIDER_drug_names.tsv', 'args': {'id_field': 'Code', 'name_field': 'Name', 'tsv': True}},
        'DRUGBANK':{ 'function': 'from_csv', 'file_path': 'hydrate_names/names_drugs_drugbank.tsv', 'args': {'id_field': 'code', 'name_field': 'name', 'tsv': True}},
        'PUBCHEM SUBSTANCE': {'function': 'from_csv', 'file_path': 'hydrate_names/pubchem_substance.csv', 'args': {'id_field': 'sid', 'name_field': 'subssynonym', 'id_as_string': True}},
        'PUBCHEM COMPOUND' :{'function': 'from_csv', 'file_path': 'hydrate_names/pubchem_compound.csv', 'args': {'id_field': 'cid', 'name_field': 'cmpdname', 'id_as_string': True}},
        'NPASS':{ 'function': 'from_csv', 'file_path': 'hydrate_names/npass.tsv', 'args': {'id_field': 'np_id', 'name_field': 'pref_name', 'tsv': True}},
        'PHARMGKB':{ 'function': 'from_csv', 'file_path': 'hydrate_names/pharmakgb_chemicals.tsv', 'args': {'id_field': 'PharmGKB Accession Id', 'name_field': 'Name', 'tsv': True}},
        'CHEBI':{ 'function': 'from_csv', 'file_path': 'hydrate_names/chebi.tsv', 'args': {'id_field': 'ID', 'name_field': 'NAME', 'tsv': True}},

    },
    'protein':{
        'HUGO GENE NOMENCLATURE COMMITTEE':{'function': 'from_csv', 'file_path': 'hydrate_names/hgnc.tsv', 'args': {'id_field': 'hgnc_id', 'name_field': 'name', 'tsv': True}},
        # 'UNIPROTKB':{ 'function': 'from_csv', 'file_path': 'hydrate_names/uniprot.tsv', 'args': {'id_field': 'From', 'name_field': 'Protein names', 'tsv': True}},
        # 'UNIPROTKB':{ 'function': 'from_csv', 'file_path': 'hydrate_names/uniprot.tsv', 'args': {'id_field': 'From', 'name_field': 'Protein names', 'tsv': True}},
        'UNIPROTKB':{ 'function': 'from_dict', 'file_path': 'hydrate_names/uniprot_new_cleaned.json'},

        'NPASS':{ 'function': 'from_csv', 'file_path': 'hydrate_names/npass2.tsv', 'args': {'id_field': 'target_id', 'name_field': 'target_name', 'tsv': True}}
    },
    'molecule':{
        'DRUGBANK':{ 'function': 'from_dict', 'file_path': 'hydrate_names/drugbank_targets.json'}
    },
    
    'activity':{
        'NAME_DRUGBANK':{ 'function': 'from_text', 'args': {}},
    },
    'gene':{
        'NCBI GENE':{'function': 'from_dict', 'file_path': 'hydrate_names/ncbi_gene_dict.json'}
    },
    'disease': {
        'OMIM':{ 'function': 'from_csv', 'file_path': 'hydrate_names/mimTitles.tsv', 'args': {'id_field': 'MIM Number', 'name_field': 'Preferred Title; symbol', 'tsv': True, 'id_as_string': True}},
        'PHARMGKB':{ 'function': 'from_csv', 'file_path': 'hydrate_names/pharmakgb_phenotypes.tsv', 'args': {'id_field': 'PharmGKB Accession Id', 'name_field': 'Name', 'tsv': True}},
        'MESH':{ 'function': 'from_dict', 'file_path': 'hydrate_names/mesh.json', 'args': {}},
        'SNOMEDCT':{ 'function': 'from_dict', 'file_path': 'hydrate_names/snomed.json', 'args': {}},
        'UMLS':{ 'function': 'from_dict', 'file_path': 'hydrate_names/umls.json', 'args': {}},
        'ORPHANET':{ 'function': 'from_dict', 'file_path': 'hydrate_names/orphanet_comp.json', 'args': {}},
    },
    'phenotype': { 
        'HPO':{ 'function': 'from_dict', 'file_path': 'hydrate_names/hpo_comp.json', 'args': {}}
        },

     'pathway':{
        'REACTOME':{ 'function': 'from_csv', 'file_path': 'hydrate_names/reactome.tsv', 'args': {'id_field': 'id', 'name_field': 'name', 'tsv': True}}
        },
    
    'effect':{
        'NAME_DRUGBANK':{ 'function': 'from_text', 'args': {}}
        },
        
    'side_effect':{
        'NAME':{ 'function': 'from_text', 'args': {}},
    },
    
    'indication':{
        'NAME':{ 'function': 'from_text', 'args': {}},
    }
}


selected_sources = {
    'compound': ['WIKIPEDIA', 'PUBCHEM COMPOUND', 
                #  'CHEMBL',
                'CHEBI','DRUGBANK', 'NPASS', 'SIDER', 'PHARMGKB'],
    'protein': ['UNIPROTKB', 'NPASS'],
    'molecule':['DRUGBANK'],
    'activity':['NAME_DRUGBANK'],
    'gene':['NCBI GENE'],
    'disease': ['OMIM', 'SNOMEDCT', 'MESH', 'UMLS', 'ORPHANET', 'PHARMGKB'],
    'phenotype':['HPO'],
    'pathway':['REACTOME'],
    'effect':['NAME_DRUGBANK'],
    'side_effect':['NAME'],
    'indication':['NAME']
}

for _type, sources in selected_sources.items():
    for source in sources:
        dataset = datasets[_type][source]
        print(f"Processing dataset: {source}")

        if dataset['function'] == 'from_text':
            kg_nodes_df = from_text(kg_nodes_df, source, _type)
        elif dataset['function'] == 'from_csv':
            kg_nodes_df = from_csv(
                kg_nodes_df,
                dataset['file_path'],
                source,
                dataset['args']['id_field'],
                dataset['args']['name_field'],
                _type,
                **{k: v for k, v in dataset['args'].items() if k not in ['id_field', 'name_field']}
            )
        elif dataset['function'] == 'from_dict':
            kg_nodes_df = from_dict(kg_nodes_df, dataset['file_path'], source, _type)

kg_nodes_df.to_csv(PATH_TO_CONSTANTS+CONSTANTS['cleaned_nodes_csv'])

