# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Optional
import requests
import yaml
from utils import read_config as reader
from extract_store_frames import process_all_videos, describe_all_videos

# Add the parent directory of the current script to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

host = None
port = None
generate_frames = None
path = None
image_output_dir = None
meta_output_dir = None
N = None

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(asctime)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S"
    )

def update_localtime():
    # align /etc/timezone and /etc/localtime in container
    # Read the timezone from /etc/timezone
    timezone_cmd = ["cat", "/etc/timezone"]
    timezone_result = subprocess.run(timezone_cmd, capture_output=True, text=True, check=True)

    if timezone_result.returncode == 0:
        timezone = timezone_result.stdout.strip()

        # Create symbolic link to /etc/localtime
        ln_cmd = ["ln", "-snf", f"/usr/share/zoneinfo/{timezone}", "/etc/localtime"]
        ln_result = subprocess.run(ln_cmd, capture_output=True, check=True)

        if ln_result.returncode == 0:
            logging.info("Timezone set successfully.")
        else:
            logging.info("Error setting timezone.")
    else:
        logging.info("Error reading timezone file.")

def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x

def read_file(path):
    content = None
    with open(path, "r") as file:
        content = file.read()
    return content

def get_data(api_url:str, query:Optional[dict] = None):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(e)
        return None

def post_data(api_url: str, body:dict):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(e)
        return None
    
def post_file(api_url: str, images, metadatas):
    try:
        response = requests.post(api_url, files=images, data=metadatas)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(e)
        return None

def check_health_vectordb():
    response = get_data(vector_health_url, {})
    return response

def init_vectordb(db: str):
    results = post_data(vector_init_url, {"selected_db": db})
    return results

def store_into_vectordb(metadata_file_path, db, insert_url, video_name: Optional[str] = None):
    GMetadata = read_json(metadata_file_path)
    global_counter = 0

    total_videos = len(GMetadata.keys())
    
    if video_name is not None:
        GMetadata = {video_name: GMetadata[video_name]}
        
    for _, (video, data) in enumerate(GMetadata.items()):

        image_name_list = []
        metadata_list = []
        ids = []
        
        image_files = [] # send image files to remote db
        
        # process frames
        frame_metadata = read_json(data["extracted_frame_metadata_file"])
        for frame_id, frame_details in frame_metadata.items():
            global_counter += 1
            meta_data = {
                "timestamp": frame_details["timestamp"],
                "frame_path": frame_details["frame_path"],
                "video": video,
                "embedding_path": data["embedding_path"],
                "date": frame_details["date"],
                "year": frame_details["year"],
                "month": frame_details["month"],
                "day": frame_details["day"],
                "time": frame_details["time"],
                "hours": frame_details["hours"],
                "minutes": frame_details["minutes"],
                "seconds": frame_details["seconds"],
            }
            if db == "vdms":
                meta_data.update({"date_time": frame_details["date_time"]})  #{"_date":frame_details["date_time"]},
            elif db == "chroma":
                pass
                
            image_path = frame_details["frame_path"]
            image_name_list.append(image_path)

            metadata_list.append(meta_data)
            ids.append(str(global_counter))
        # generate clip embeddings
        for image_path in image_name_list:
            with open(image_path, "rb") as image_file:
                image_files.append(("images", (os.path.basename(image_path), image_file.read(), "image/jpeg")))
        
        metadata_dict = {"metadatas": metadata_list}
        metadatas = {"metadatas": json.dumps(metadata_dict)}
        try:
            results = post_file(insert_url, image_files, metadatas)
        except Exception as e:
            raise e
        
        logging.info(f"✅ {_+1}/{total_videos} video {video} added to {db}, image len {len(image_name_list)}, metadata len {len(metadata_list)}")
    
def generate_image_embeddings(db):
    if generate_frames:
        logging.info ("Processing all videos, Generated frames will be stored at")
        logging.info (f"input video folder = {path}")
        logging.info (f"frames output folder = {image_output_dir}")
        logging.info (f"metadata files output folder = {meta_output_dir}")
        process_all_videos(path, image_output_dir, meta_output_dir, N, db)
    
    global_metadata_file_path = meta_output_dir + "metadata.json"
    logging.info(f"global metadata file available at {global_metadata_file_path}")
    store_into_vectordb(global_metadata_file_path, db, image_insert_url)
    
def retrieval_testing():
    Q = "man holding red basket"
    logging.info (f"Testing Query {Q}")
    results = get_data(vector_query_url, {"prompt": Q})
        
if __name__ == "__main__":
    
    update_localtime() # only for docker container
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process configuration file for generating and storing embeddings.")
    # Add argument for configuration file
    parser.add_argument("--config", type=str, help="Path to configuration file (e.g., config.yaml)")

    # for NESS demo
    parser.add_argument("--db", type=str, help="chroma/vdms (for NESS demo)" )
    parser.add_argument("--generate", type=str, default="True", help="generate & ingest embedding or not (for NESS demo)")
    # Parse command-line arguments
    args = parser.parse_args()

    # Read configuration file
    logging.info("Reading config file")  
    config = reader(args.config)
    msg = "Config file read successfully: \n" + yaml.dump(config, default_flow_style=False, sort_keys=False)
    logging.info(msg)

    generate_frames = config["generate_frames"]
    embed_frames = config["embed_frames"]
    path = config["videos"]
    desc_output_dir = config["description"]
    image_output_dir = config["image_output_dir"]
    meta_output_dir = config["meta_output_dir"]
    N = config["number_of_frames_per_second"]
    
    host = config["vector_db"]["host"]
    port = config["vector_db"]["port"]
    # selected_db = config["vector_db"]["choice_of_db"] # comment out for NESS
    selected_db = args.db
    url_pre = "http://" + host + ":" + str(port)
    vector_query_url = url_pre + config["vector_db"]["vector_query_url"]
    vector_init_url = url_pre + config["vector_db"]["vector_init_url"]
    vector_health_url = url_pre + config["vector_db"]["vector_health_url"]
    image_insert_url = url_pre + config["vector_db"]["image_insert_url"]
    text_insert_url = url_pre + config["vector_db"]["text_insert_url"]
    
    # Creating DB
    logging.info(f"Connect to {selected_db} at {host}:{port}, generate: {args.generate}")
    check_health_vectordb() # health check for vectorDB    
    init_vectordb(selected_db)
    
    if args.generate == "True":
        logging.info(f"Configuring DB with image embedding. It may take few minutes to extract and ingest all required frames.")
        generate_image_embeddings(selected_db)
    retrieval_testing()    
    logging.info("✅ Done!")
