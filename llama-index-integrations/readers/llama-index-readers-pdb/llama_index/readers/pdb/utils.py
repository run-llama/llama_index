import requests
from typing import List, Dict


def get_pdb_publications_from_rcsb(pdb_id: str) -> List[Dict]:
    base_url = "https://data.rcsb.org/rest/v1/core/"
    pubmed_query = f"{base_url}pubmed/{pdb_id}"
    entry_query = f"{base_url}entry/{pdb_id}"
    pubmed_response = requests.get(pubmed_query)
    entry_response = requests.get(entry_query)
    if pubmed_response.status_code != 200:
        raise Exception(
            f"RCSB API call (pubmed) for {pdb_id} failed with status code: {pubmed_response.status_code}"
        )
    if entry_response.status_code != 200:
        raise Exception(
            f"RCSB API call (entry) for {pdb_id} failed with status code: {entry_response.status_code}"
        )
    return (entry_response.json(), pubmed_response.json())


def parse_rcsb_publication_dict(entry_response: Dict, pubmed_response: Dict):
    parsed_dict = {}
    citations = entry_response["citation"]
    primary_citation = [pub for pub in citations if pub["id"] == "primary"][0]
    parsed_dict[primary_citation["title"]] = {
        "doi": pubmed_response["rcsb_pubmed_doi"],
        "abstract": {"abstract": pubmed_response["rcsb_pubmed_abstract_text"]},
        "primary": True,
    }
    return primary_citation["title"], parsed_dict


def get_pdb_publications_from_ebi(pdb_id: str) -> List[Dict]:
    pdb_id = str.lower(pdb_id)
    base_url = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/publications/"
    response = requests.get(f"{base_url}{pdb_id}")
    if response.status_code != 200:
        raise Exception(
            f"EBI API call for ({pdb_id}) failed with status code: {response.status_code}"
        )
    pub_dicts = response.json()[pdb_id]
    return pub_dicts


def parse_ebi_publication_list(pub_list: List[Dict]):
    parsed_dict = {}
    for i, pub_dict in enumerate(pub_list):
        parsed_dict[pub_dict["title"]] = {
            "doi": pub_dict["doi"],
            "abstract": pub_dict["abstract"],
            "primary": i == 0,
        }
    return pub_list[0]["title"], parsed_dict


def get_pdb_abstract(pdb_id: str) -> Dict:
    try:
        pub_dicts_list = get_pdb_publications_from_ebi(pdb_id)
        pimary_title, pubs_dict = parse_ebi_publication_list(pub_dicts_list)
    except Exception:
        try:
            entry_response, pubmed_response = get_pdb_publications_from_rcsb(pdb_id)
            pimary_title, pubs_dict = parse_rcsb_publication_dict(
                entry_response, pubmed_response
            )
        except Exception:
            raise Exception("Failed to fetch data from both RCSB and EBI API")
    return pimary_title, pubs_dict
