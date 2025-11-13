import json
import os
import re
import requests
import wget
from tqdm import tqdm
from dotenv import load_dotenv

#get key from .env
load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
DATASET_NAME = "s2orc"
LOCAL_PATH = "data/s2orc/"
os.makedirs(LOCAL_PATH, exist_ok=True)

# get latest release's ID
response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
RELEASE_ID = response["release_id"]
print(f"Latest release ID: {RELEASE_ID}")

# get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
# download via wget. this can take a while...
response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": API_KEY}).json()
for url in tqdm(response["files"]):
    match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
    assert match.group(1) == RELEASE_ID
    SHARD_ID = match.group(2)

    response = requests.get(url, stream=True)
    with open(os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz"), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

print("Downloaded all shards.")