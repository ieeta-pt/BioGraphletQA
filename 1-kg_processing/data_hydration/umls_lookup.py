import requests
from tqdm import tqdm
import json

# Step 1: Read API Key from the File
def read_api_key(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

# Step 2: Get TGT
def get_tgt(api_key):
    url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    response = requests.post(url, data={"apikey": api_key})
    response.raise_for_status()
    return response.headers['location']

# Step 3: Get Service Ticket
def get_service_ticket(tgt_url):
    response = requests.post(tgt_url, data={"service": "http://umlsks.nlm.nih.gov"})
    response.raise_for_status()
    return response.text

# Step 4: Search for ID
def search_id(api_key, source, id_to_search):
    tgt_url = get_tgt(api_key)
    service_ticket = get_service_ticket(tgt_url)
    if source == 'umls':
        url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{id_to_search}"
    else:
        url = f"https://uts-ws.nlm.nih.gov/rest/content/current/source/{source}/{id_to_search}"
    response = requests.get(url, params={"ticket": service_ticket})
    response.raise_for_status()
    return response.json()

# Step 5: Create Dictionary of ID to Name Mappings
def create_id_name_mapping(api_key, source, id_list):
    id_name_mapping = {}
    for id_to_search in tqdm(id_list):
        try:
            result = search_id(api_key, source, id_to_search)
            # Extract name from the JSON response
            name = result['result']['name']
            id_name_mapping[id_to_search] = name
            print(f"Mapped ID: {id_to_search} -> Name: {name}")
        except requests.exceptions.RequestException as e:
            print(f"Error with ID {id_to_search}: {e}")
    return id_name_mapping



api_key_file = "umlskey.txt"
api_key = read_api_key(api_key_file)

with open('umls_set.txt', 'r') as file:
    codes = [line.strip() for line in file.readlines()]

# Parameters for the UMLS API query
source = "umls"  # Replace with your desired source
id_list = codes  # Replace with your list of IDs

# Create the mapping
id_name_mapping = create_id_name_mapping(api_key, source, id_list)

# Print the final dictionary
print("Final ID to Name Mapping:", len(id_name_mapping))



with open("hydrate_names/umls.json", "w") as file:
    json.dump(id_name_mapping, file, indent=4)