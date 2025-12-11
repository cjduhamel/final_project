import json
import os
import re
import requests
import wget
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

#get key from .env
load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
DATASET_NAME = "s2orc"
LOCAL_PATH = "data/s2orc/"
os.makedirs(LOCAL_PATH, exist_ok=True)

CSV_PATH = "s2orc_downloads.csv"
reg_exp = r"https://ai2-s2ag\.s3\.amazonaws\.com/staging/([^/]+)/s2orc[^/]*/([^?]+)\.gz"


def get_urls(csv_path=CSV_PATH):
    # get latest release's ID
    response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
    RELEASE_ID = response["release_id"]
    print(f"Latest release ID: {RELEASE_ID}")

    # get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
    # download via wget. this can take a while...
    response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": API_KEY}).json()

    # store urls in a csv file, it will be used by other scripts to download files in parallel
    columns = ["shard_id", "url", "file_name", "downloaded"]
    df = pd.DataFrame(columns=columns)

    for url in response["files"]:
        match = re.search(reg_exp, url)
        if not match:
            print(f"Invalid URL format: {url}")
            continue
        assert match.group(1) == RELEASE_ID
        SHARD_ID = match.group(2)

        df = pd.concat([df, pd.DataFrame([{
            "shard_id": SHARD_ID,
            "url": url,
            "file_name": "",
            "downloaded": False
        }])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df

def refresh_urls():
    # get latest release's ID
    response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
    RELEASE_ID = response["release_id"]
    print(f"Latest release ID: {RELEASE_ID}")

    # get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
    # download via wget. this can take a while...
    response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": API_KEY}).json()

    #load existing csv
    df_existing = pd.read_csv(CSV_PATH)

    #replace urls in existing csv
    for url in response["files"]:
        match = re.search(reg_exp, url)
        if not match:
            print(f"Invalid URL format: {url}")
            continue
        assert match.group(1) == RELEASE_ID
        SHARD_ID = match.group(2)

        #update url in existing df
        df_existing.loc[df_existing["shard_id"] == SHARD_ID, "url"] = url

    #save updated df
    df_existing.to_csv(CSV_PATH, index=False)
    return df_existing




def download_shard(df, row):
    url = row["url"]
    downloaded = row.get("downloaded", False)

    if downloaded:
        print(f"Skipping {url}, already marked as downloaded.")
        return

    # Extract SHARD_ID and create destination filename
    match = re.search(reg_exp, url)
    if not match:
        print(f"Invalid URL format: {url}")
        return

    shard_id = match.group(2)
    file_path = os.path.join(LOCAL_PATH, f"{shard_id}.gz")

    try:
        print(f"Downloading {shard_id}...")
        response = requests.get(url, stream=True)
        if response.status_code == 403 or "ExpiredToken" in response.text:
            raise Exception("Expired token")
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {shard_id}")

       
        # Update the CSV to mark as downloaded
        df.loc[df["shard_id"] == shard_id, "downloaded"] = True
        df.loc[df["shard_id"] == shard_id, "file_name"] = f"{shard_id}.gz"
        df.to_csv(CSV_PATH, index=False)
        print(f"Updated CSV for {shard_id}")

    except Exception as e:
        print(f"Failed to download {shard_id}: {e}")
        raise e

if __name__ == "__main__":

    #get urls and save to CSV if not exists
    if not os.path.exists(CSV_PATH):
        get_urls()

    # Download all files listed in the CSV in parallel
    url_df = pd.read_csv(CSV_PATH)

    # Filter to URLs not marked as downloaded
    pending_rows = url_df[url_df["downloaded"] != True].to_dict(orient="records")
    print(f"Pending downloads: {len(pending_rows)} out of {len(url_df)}")

    # delete any partially downloaded files
    for row in pending_rows:
        url = row["url"]
        match = re.search(reg_exp, url)
        if match:
            shard_id = match.group(2)
            file_path = os.path.join(LOCAL_PATH, f"{shard_id}.gz")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted partial file: {file_path}")

    #not parallel for debugging
    for i in tqdm(range(len(pending_rows)), desc="Downloading shards", unit="shard"):
        retry_count = 0
        max_retries = 20
        while retry_count < max_retries:
            try:
                row = pending_rows[i]
                download_shard(url_df, row)
                retry_count = 0
                break
            except Exception as e:
                if "expired" in str(e).lower() or "403" in str(e).lower():
                    print("Token expired, refreshing URLs...")
                    url_df = refresh_urls()
                    #keep only what was previously in pending_rows
                    url_df = url_df[url_df["shard_id"].isin([r["shard_id"] for r in pending_rows])]
                    pending_rows = url_df.to_dict(orient="records")
                    
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("Max retries reached, skipping this shard.")
                    continue
                else:
                    print(f"Download failed for {pending_rows[i]['shard_id']}, skipping. Error: {e}")
                    break
        


