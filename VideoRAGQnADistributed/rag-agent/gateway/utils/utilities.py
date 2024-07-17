# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests
import yaml


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_top_doc(videos: list) -> str:
    hit_score = {}
    if videos == None:
        return None
    for video in videos:
        try:
            video_name = video["metadata"]["video"]
            if video_name not in hit_score.keys():
                hit_score[video_name] = 0
            hit_score[video_name] += 1
        except KeyError as r:
            print(f"no video name {r}")

    x = dict(sorted(hit_score.items(), key=lambda item: -item[1])) # sorted dict of video name and score
    top_name = list(x.keys())[0]
    print(f"top docs = {x}")
    
    return top_name

def get_data(api_url:str, query:dict):
    try:
        response = requests.get(api_url, query)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        raise e

def post_data(api_url: str, body:dict):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        raise e