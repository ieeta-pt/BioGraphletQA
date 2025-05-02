import requests
from tqdm import tqdm
import json

def get_compound_names(chembl_ids):
    """
    Fetches compound names for a list of ChEMBL IDs from the ChEMBL API.

    Parameters:
        chembl_ids (list): A list of ChEMBL IDs.

    Returns:
        list: A list of dictionaries containing ChEMBL IDs and their compound names.
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/"
    results = []

    for chembl_id in tqdm(chembl_ids):
        try:
            # Make the API request
            response = requests.get(f"{base_url}{chembl_id}", headers={"Accept": "application/json"})
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            
            # Extract the compound name
            compound_name = data.get("pref_name", "Unknown")
            
            # Append the result as a dictionary
            results.append({"chembl_id": chembl_id, "compound_name": compound_name})
        
        except requests.exceptions.RequestException as e:
            # Append error info for this ID
            results.append({"chembl_id": chembl_id, "error": str(e)})

    return results

with open('chembl.txt', 'r') as file:
    codes = [line.strip() for line in file.readlines()]

# Example usage
results = get_compound_names(codes)

# Print the results
# for result in results:
    # print(result)


output_file = "chembl.json"
with open(output_file, "w") as file:
    json.dump(results, file)