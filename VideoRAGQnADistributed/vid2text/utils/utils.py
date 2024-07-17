import os
import requests
import yaml

from fastapi import Request


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def set_proxy(addr:str):
    # for DNS: "http://child-prc.intel.com:913"
    # for Huggingface downloading: "http://proxy-igk.intel.com:912"
    os.environ['http_proxy'] = addr
    os.environ['https_proxy'] = addr
    os.environ['HTTP_PROXY'] = addr
    os.environ['HTTPS_PROXY'] = addr

def get_chat_handler(request: Request):
    """
    Helper to grab dependencies that live in the app.state
    """
    # See application for the key name in `app`.
    return request.app.state.chat_handler

def post_data(api_url: str, body:dict):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise e
