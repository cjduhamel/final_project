"""
Read through S2ORC data and assign each paper id to its corresponding shard filename.
"""

import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import csv
import logging

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data/s2orc/"
OUT_FILE = "id_to_shard.csv"
MAX_PAPERS = 10000 
MAX_TEXT = 2000

shard_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.gz')]

def process_shard(shard_path):
    records = []
    shard_filename = os.path.basename(shard_path)
    try:
        with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line)
                    corpusid = paper.get('corpusid')
                    if corpusid:
                        records.append({'corpusid': corpusid, 'shard_filename': shard_filename})
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.info(f"Error reading {shard_filename}: {e}")
    return records


def index_dataset():
    """
    Index the dataset by creating a mapping from paper IDs to shard filenames.
    """
    # Write the CSV header first
    pd.DataFrame(columns=["corpusid", "shard_filename"]).to_csv(OUT_FILE, index=False)

    # Now process shards in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_shard, path): path for path in shard_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing shards"):
            result = future.result()
            if result:
                df_chunk = pd.DataFrame(result)
                # Append mode, no header
                df_chunk.to_csv(OUT_FILE, mode='a', header=False, index=False)
            else:
                logging.debug(f"No valid records found in shard: {futures[future]}")

def load_index():
    """
    Load the index CSV into a dict

    Returns:
        dict: Mapping from corpusid to shard filename.
    """
    id_to_shard = {}
    with open(OUT_FILE, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Loading index"):
            id_to_shard[int(row['corpusid'])] = row['shard_filename']
    return id_to_shard

def find_paper_in_lines(corpusid, lines):
    """Given lines from a shard, find the paper with the specified corpusid."""
    for line in lines:
        try:
            paper = json.loads(line)
            if paper.get('corpusid') == corpusid:
                return paper
        except json.JSONDecodeError:
            continue
    return None

paper_cache = {}

def get_paper(corpusid, id_to_shard):
    """
    Retrieve the paper JSON for a given corpusid using the index.

    Inputs:
        corpusid (int): The corpusid of the paper to retrieve.
        id_to_shard (dict): Mapping from corpusid to shard filename.
    Returns:
        dict or None: The paper JSON if found, else None.
    """

    #check cache first
    if corpusid in paper_cache:
        return paper_cache[corpusid]

    shard_filename = id_to_shard.get(corpusid)
    if not shard_filename:
        logging.debug(f"CorpusID {corpusid} not found in index.")
        return None

    #get number of available multiprocessing cores
    cores = os.cpu_count() or 1


    shard_path = os.path.join(DATA_DIR, shard_filename)
    try:
        with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

            #split lines into chunks for parallel processing
            num_lines = len(lines)
            chunk_size = (num_lines // cores) + 1
            line_chunks = [lines[i:i + chunk_size] for i in range(0, num_lines, chunk_size)]

            #use multiprocessing and starmap to find the paper in parallel
            with Pool(processes=cores) as pool:
                results = pool.starmap(find_paper_in_lines, [(corpusid, chunk) for chunk in line_chunks])


            #check results from all chunks
            paper = None
            for result in results:
                if result:
                    paper = result
                    break

            if paper:
                logging.debug(f"Found paper with CorpusID {corpusid} in shard {shard_filename}.")
                paper_cache[corpusid] = paper
                return paper
            else:
                logging.debug(f"Paper with CorpusID {corpusid} not found in shard {shard_filename}.")
    except Exception as e:
        logging.info(f"Error reading {shard_filename}: {e}")
    return None

def empty_cache():
    """Empty the paper cache."""
    paper_cache.clear()

def get_referenced_id(paper, ref_ids):
    """
    Given a paper from the s2orc dataset and the ref_ids field from a sample, return the corresponding paper IDs.
    Args:
        paper (dict): The paper data from the s2orc dataset.
        ref_ids (list): List of reference IDs from a sample.
    Returns:
        dict: Key-value pairs whith the BIBREF as key and the corresponding paper ID as value.
    """

    if ref_ids is None:
        return {}
    ref_id_map = {}
    raw_refs = paper.get('body', {}).get('annotations', {}).get('bib_ref', [])
    paper_refs = json.loads(raw_refs) if isinstance(raw_refs, str) else raw_refs
    if not paper_refs:
        logging.debug("No paper references found.")
        return ref_id_map

    logging.debug("Paper references found:", paper_refs)
    logging.debug("Paper references type:", type(paper_refs))

    for ref_sections in ref_ids: #ref_ids is a list of lists
        for ref_id in ref_sections: #get each ref_id
            formatted_ref_id = ref_id.replace("BIBREF", "b") #convert to s2orc format
            for ref in paper_refs: #search for the ref_id in the paper's references
                #get the current ref_id from the paper's references
                attributes = ref.get('attributes', {})
                if not attributes:
                    continue
                curr_ref_id = attributes.get('ref_id')
                if curr_ref_id == formatted_ref_id:
                    ref_id_map[ref_id] = attributes.get('matched_paper_id')
                    break
            if ref_id not in ref_id_map:
                logging.debug(f"Reference ID {ref_id} not found in paper references.")
                    

    return ref_id_map

def load_citeworth():
    """
    Load the CiteWorth dataset
    """
    splits = {'train': 'train.jsonl', 'validation': 'dev.jsonl', 'test': 'test.jsonl'}
    df_train = pd.read_json("hf://datasets/copenlu/citeworth/" + splits["train"], lines=True)
    return df_train

def load_processed_ids(log_file="processed_papers.log"):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def build_complete_dataset():
    """
    Build a complete dataset combining the CiteWorth data and S2ORC data.

    Loads the Citeworth dataset, retrieves all corresponding s2orc papers, along with any referenced papers. 
    Also collects different parts of the paper from the citeworth dataset and combines them into a single entry.

    Method:
        For each entry in the CiteWorth dataset:
            - If the paper_id is not already in the complete dataset:
                - Retrieve the main paper from S2ORC using the paper_id.
                - For each reference in the citeworth entry, retrieve the corresponding paper from S2ORC.
                - Combine the main paper and referenced papers into a single entry.
            - If the paper_id is already in the complete dataset:
                - Add the new paper section to the existing entry (paper text is an array of sections, ordered by their section_index)

    Returns:
        dict: Complete dataset with all papers and their references (to be saved as jsonl)
    """

    id_to_shard = load_index() #get the id to shard mapping
    citeworth_df = load_citeworth() #load the citeworth dataset
    
    #group by paper_id to process all sections of the same paper together
    grouped = citeworth_df.groupby('paper_id')

    processed_ids = load_processed_ids() #load already processed paper ids

    paper_count = 0
    pbar = tqdm(total=MAX_PAPERS, desc="Building complete dataset", unit="papers")
    for paper_id, group in grouped:
        if paper_count >= MAX_PAPERS:
            logging.info(f"Reached maximum paper limit of {MAX_PAPERS}. Stopping dataset build.")
            break

        dataset_chunk = []
        #skip already processed paper ids
        if paper_id in processed_ids:
            logging.debug(f"Skipping already processed paper_id: {paper_id}")
            continue

        if paper_id not in id_to_shard:
            logging.debug(f"Paper ID {paper_id} not found in S2ORC index.")
            continue

        if isinstance(paper_id, str) and paper_id.isdigit() or isinstance(paper_id, int):
            paper_id = int(paper_id)
        else:
            logging.info(f"Invalid paper_id format: {paper_id}")
            continue
        
        #get the paper from s2orc
        paper = get_paper(paper_id, id_to_shard)
        if not paper:
            logging.info(f"Paper ID {paper_id} not found in S2ORC dataset.")
            continue
        
        #get the referenced paper s2orc ids
        all_ref_ids = {}
        for _, row in group.iterrows():
            for sample in row['samples']:
                
                ref_ids = sample.get('ref_ids', [])
                ref_id_map = get_referenced_id(paper, ref_ids)
                
                all_ref_ids.update(ref_id_map)

        #get the referenced papers from s2orc
        referenced_papers = {}
        for ref_id, ref_paper_id in all_ref_ids.items():
            if ref_paper_id:
                if ref_paper_id not in id_to_shard:
                    logging.debug(f"Referenced paper ID {ref_paper_id} not found in S2ORC index.")
                    continue
                ref_paper = get_paper(ref_paper_id, id_to_shard)
                if ref_paper:
                    referenced_papers[ref_id] = ref_paper
                else:
                    logging.debug(f"Referenced paper ID {ref_paper_id} not found in S2ORC dataset.")
                

        #for each section in the paper, iterate through all the sentences (samples)
        for _, row in group.iterrows():
            
            #for each sample, create rows for each possible referenced paper, with label 1 if it is referenced, 0 otherwise
            for sample in row['samples']:
                sentence = sample.get('text', "")
                
                #loop through all referenced papers
                for ref_id, ref_paper in referenced_papers.items():
                    item = {} # new item for each referenced paper

                    #get all ref ids in the sample (it is a list of lists in case of multiple seperate substrings of references)
                    ids_to_check = []
                    get_ref_ids = sample.get('ref_ids', [])
                    if not get_ref_ids:
                        continue
                    for ref_id_section in get_ref_ids:
                        ids_to_check.extend(ref_id_section)

                    #set label based on whether the current ref_id is in the sample's ref_ids
                    label = 1 if ref_id in ids_to_check else 0

                    item['original_paper_id'] = paper_id
                    item['sentence'] = sentence

                    ref_paper_id = ref_paper.get('corpusid', "")
                    item['ref_paper_id'] = ref_paper_id

                    ref_title = ref_paper.get('title', "")
                    item['ref_paper_title'] = ref_title

                    ref_authors = ref_paper.get('authors', [])
                    if isinstance(ref_authors, list):
                        item['ref_paper_authors'] = ", ".join(ref_authors)

                    #get the first 500 characters of the reference paper's abstract as context
                    text = ref_paper.get('body', {}).get('text', "")

                    if not text:
                        text = ref_paper.get('content', {}).get('text', "")

                    item['ref_paper_text'] = text[:MAX_TEXT] if text else ""
                    item['label'] = label
                    dataset_chunk.append(item)

        #save the current chunk to file
        if dataset_chunk:
            paper_count += 1
            pbar.update(1)

        save_chunk(dataset_chunk, paper_id)
        

    empty_cache() #clear the cache to free memory


def save_chunk(dataset_chunk, paper_id, out_file="complete_dataset.jsonl", log_file="processed_papers.log"):
    """
    Save a chunk of the complete dataset to a JSONL file.
    Create the JSONL file if it does not exist, else append to it.
    Append the paper_id to a log file after saving the chunk.

    Args:
        dataset (list): The complete dataset to save.
        paper_id (int): The paper ID for logging purposes.
        out_file (str): The output file path.
    """
    mode = 'a' if os.path.exists(out_file) else 'w'
    with open(out_file, mode, encoding='utf-8') as f:
        for item in dataset_chunk:
            f.write(json.dumps(item) + '\n')

    #save the last processed paper_id
    mode = 'a' if os.path.exists(log_file) else 'w'
    with open(log_file, mode, encoding='utf-8') as log_f:
        log_f.write(f"{paper_id}\n")

def create_dataset():
    """
    Create and save the complete dataset.
    """

    build_complete_dataset()
    logging.info(f"Complete dataset saved to complete_dataset.jsonl")


if __name__ == "__main__":
    create_dataset()
