"""
Read through S2ORC data and assign each paper id to its corresponding shard filename.
"""

import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

print(f"Detected logical cores: {os.cpu_count()}")
data_dir = "data/s2orc/"
out_file = "id_to_shard.csv"
shard_filenames = [f for f in os.listdir(data_dir) if f.endswith('.gz')]

import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

data_dir = "data/s2orc/"
out_file = "id_to_shard.csv"
shard_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.gz')]

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
        print(f"Error reading {shard_filename}: {e}")
    return records


# Write the CSV header first
pd.DataFrame(columns=["corpusid", "shard_filename"]).to_csv(out_file, index=False)

# Now process shards in parallel
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_shard, path): path for path in shard_paths}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing shards"):
        result = future.result()
        if result:
            df_chunk = pd.DataFrame(result)
            # Append mode, no header
            df_chunk.to_csv(out_file, mode='a', header=False, index=False)
        else:
            print(f"No valid records found in shard: {futures[future]}")



