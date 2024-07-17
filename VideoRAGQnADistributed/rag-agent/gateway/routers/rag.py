# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils import get_top_doc, get_data, post_data, read_config
from utils import get_formatted_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     [%(asctime)s] %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
    )

config = read_config("./config.yaml")
model_server_url = "http://" + config['model_server']['host'] + ':' + str(config['model_server']['port']) + config['model_server']['url'] 
vector_query_url = "http://" + config['vector_db']['host'] + ':' + str(config['vector_db']['port']) + config['vector_db']['vector_query_url'] 
prepare_db_url = "http://" + config['vid2text']['host'] + ':' + str(config['vid2text']['port']) + config['vid2text']['prepare_db_url'] 
describe_url = "http://" + config['vid2text']['host'] + ':' + str(config['vid2text']['port']) + config['vid2text']['describe_url'] 

router = APIRouter(prefix="/agent", tags=["RAG agent"])


    
class initConfig(BaseModel):
    LLM: str
    vectorDB: str

class promptConfig(BaseModel):
    prompt: str

"""
curl -X 'POST' \
  'http://172.16.186.168:7000/agent/init' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "string",
  "vectordb": "vdms"
}'
"""
@router.post("/init", summary="init")
@router.post("/init/", include_in_schema=False)
async def init(conf: initConfig):
    logging.info("Received conf: %s", conf)
    try:
        # post selected db to vid2text for data preparation
        
        logging.info(f"selected_db:{conf.vectorDB.lower()}")
        results = post_data(prepare_db_url, {"selected_db": conf.vectorDB})
        
        # TODO: post selected db to model server for setup
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    return JSONResponse({"message:": "successfully init"})

@router.post("/prompt", summary="get prompt from query using RAG")
@router.post("/prompt/", include_in_schema=False)
async def rag(prompt: promptConfig):
    try:
        # retrieve video from vector store and get description
        videos = get_data(vector_query_url, {"prompt": prompt.prompt})
        top_video = get_top_doc(videos)
        scene_des = get_data(describe_url, {"name": top_video})["description"]
        logging.info(f"scene description of video {top_video}:{scene_des}")        
        # prompt formatter
        formatted_prompt = get_formatted_prompt(scene=scene_des, prompt=prompt.prompt)
        logging.info(f"formatted_prompt:{formatted_prompt}")
    except Exception as e:
        logging.error(f"{e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"{e}") from e

    return JSONResponse({"prompt": f"{formatted_prompt}",
                         "video": f"{top_video}",
                         })
