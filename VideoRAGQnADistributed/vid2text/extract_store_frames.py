# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import logging
import os

import cv2
import requests
from tzlocal import get_localzone


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(asctime)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S"
    )

def extract_frames(video_path, image_output_dir, meta_output_dir, N, date_time, local_timezone, selected_db):
    """
    Extracts frames from a video file and saves them as images along with metadata.

    Args:
        video_path (str): The path to the video file.
        image_output_dir (str): The directory to store the extracted frames as images.
        meta_output_dir (str): The directory to store the metadata file.
        N (int): The number of frames to extract per second.
        date_time (datetime): The current date and time.
        local_timezone (tzinfo): The local timezone of the machine.
        selected_db (str): The selected database.

    Returns:
        tuple: A tuple containing the frames per second (fps), total number of frames, and the path to the metadata file.
    """
    video = video_path.split("/")[-1]
    # Create a directory to store frames and metadata
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(meta_output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if int(cv2.__version__.split(".")[0]) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    mod = int(fps // N)
    if mod == 0: 
        mod = 1
    
    logging.info(f"total frames {total_frames}, fps {fps}, frames/sec {N}, frames for extraction {mod}")
    
    # Variables to track frame count and desired frames
    frame_count = 0
    
    # Metadata dictionary to store timestamp and image paths
    metadata = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % mod == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds
            frame_path = os.path.join(image_output_dir, f"{video}_{frame_count}.jpg")
            time = date_time.strftime("%H:%M:%S")
            date = date_time.strftime("%Y-%m-%d")
            hours, minutes, seconds = map(float, time.split(":"))
            year, month, day = map(int, date.split("-"))
            
            cv2.imwrite(frame_path, frame)  # Save the frame as an image


            metadata[frame_count] = {
                "timestamp": timestamp,
                "frame_path": frame_path,
                "date": date,
                "year": year,
                "month": month,
                "day": day,
                "time": time,
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
            }
            if selected_db == "vdms":
                # Localize the current time to the local timezone of the machine
                # Tahani might not need this
                current_time_local = date_time.replace(tzinfo=datetime.timezone.utc).astimezone(local_timezone)

                # Convert the localized time to ISO 8601 format with timezone offset
                iso_date_time = current_time_local.isoformat()
                metadata[frame_count]["date_time"] = {"_date": str(iso_date_time)}

    # Save metadata to a JSON file
    metadata_file = os.path.join(meta_output_dir, f"{video}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Release the video capture and close all windows
    cap.release()
    return fps, total_frames, metadata_file

def process_all_videos(path, image_output_dir, meta_output_dir, N, selected_db):
    """
    Process all videos in the specified path, extract frames, and store metadata.
    
    Args:
        path (str): The directory path containing the videos to process.
        image_output_dir (str): The directory to store the extracted frames.
        meta_output_dir (str): The directory to store the metadata.
        N (int): The number of frames to extract per second.
        selected_db: The selected database for storage.
    
    Returns:
        None
    """
    videos = [file for file in os.listdir(path) if file.endswith(".mp4")]

    # logging.info(f"Total {len(videos)} videos will be processed")
    metadata = {}

    for i, each_video in enumerate(videos):
        video_path = os.path.join(path, each_video)
        date_time = datetime.datetime.now() 
        logging.info(f"date_time : {date_time}")
        # Get the local timezone of the machine
        local_timezone = get_localzone()
        fps, total_frames, metadata_file = extract_frames(video_path, image_output_dir, meta_output_dir, N, date_time, local_timezone, selected_db)
        metadata[each_video] = {
            "fps": fps, 
            "total_frames": total_frames, 
            "extracted_frame_metadata_file": metadata_file,
            "embedding_path": f"embeddings/{each_video}.pt",
            "video_path": f"{path}/{each_video}",
            }
        logging.info(f"✅  {i+1}/{len(videos)}")

    metadata_file = os.path.join(meta_output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def process_single_videos(video_path, image_output_dir, meta_output_dir, N, selected_db):
    """
    Process single video, extract frames, and store metadata.
    
    Args:
        video_path (str): The path to the video file to process.
        image_output_dir (str): The directory to store the extracted frames.
        meta_output_dir (str): The directory to store the metadata.
        N (int): The number of frames to extract per second.
        selected_db: The selected database for storage.
    
    Returns:
        None
    """
    # logging.info(f"Total {len(videos)} videos will be processed")
    metadata = {}
    exist_metadata = {}
    
    date_time = datetime.datetime.now() 
    logging.info(f"date_time : {date_time}")
    # Get the local timezone of the machine
    local_timezone = get_localzone()
    fps, total_frames, metadata_file = extract_frames(video_path, image_output_dir, meta_output_dir, N, date_time, local_timezone, selected_db)
    video_name = video_path.split("/")[-1]
    logging.info(f"✅  {video_name}: frames extracted.") 
    metadata[video_name] = {
        "fps": fps, 
        "total_frames": total_frames, 
        "extracted_frame_metadata_file": metadata_file,
        "embedding_path": f"embeddings/{video_name}.pt",
        "video_path": f"{video_path}",
        }
    metadata_file = os.path.join(meta_output_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        exist_metadata = {}
    else:
        with open(metadata_file, "r") as f:
            exist_metadata = json.load(f)
    logging.debug(f"existance metadata: {exist_metadata.keys()}")
    
    for video, data in metadata.items():
        exist_metadata[video] = data
    logging.debug(f"new metadata: {exist_metadata.keys()}")
    
    with open(metadata_file, "w") as f:
        json.dump(exist_metadata, f, indent=4)
        f.truncate()  # delete the rest

    logging.info(f"✅  {video_name}: metadata saved successfully.")


def describe_all_videos(url, video_dir, desc_dir):
    if not os.path.exists(desc_dir):
        os.makedirs(desc_dir)
    videos = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    for each_video in videos:
        video_path = os.path.join(video_dir, each_video)
        logging.info(f"Describing {video_path}")
        
        logging.info(f"url: {url}")
        # Save video description
        try:
            response = requests.post(url+f"?path={video_path}")
            response.raise_for_status()  # Raise an exception for HTTP errors
            llm_message = response.json()["llm_message"]
            # Add your code here to handle the llm_message
        except requests.exceptions.HTTPError as err:
            print("HTTP error occurred:", err)
        except requests.exceptions.RequestException as err:
            print("Request failed:", err)
        except KeyError as err:
            print("KeyError: 'llm_message' key not found in the JSON response")

        logging.info(f"LLM response: {llm_message}")
        
        desc_location = os.path.join(desc_dir, each_video+".txt")
        print(f"desc_location: {desc_location}")
        with open(desc_location, "w") as buffer:
            buffer.write(llm_message)
            