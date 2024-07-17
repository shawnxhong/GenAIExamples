import concurrent.futures
import datetime
import json
import os
import logging
import subprocess
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from video_llama import ChatHandler
from utils import read_config, get_chat_handler, post_data
from extract_store_frames import process_single_videos
from generate_store_embeddings import store_into_vectordb

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     [%(asctime)s] %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
    )

router = APIRouter(prefix="/vid2text", tags=["vid2text"])

# Read configuration file
logging.info("Reading config file")  
config = read_config("config.yaml")

desc_dir= config["description"]
upload_dir = config["videos"] #args.videos_folder #
image_output_dir = config["image_output_dir"]
meta_output_dir = config["meta_output_dir"]
N = config["number_of_frames_per_second"]
# read selection from user
vectordb = "vdms"


host = config["vector_db"]["host"]
port = config["vector_db"]["port"]
url_pre = "http://" + host + ":" + str(port)
image_insert_url = url_pre + config["vector_db"]["image_insert_url"]
vector_init_url = url_pre + config["vector_db"]["vector_init_url"]

class DB(BaseModel):
    selected_db: str = Field(..., description="selected_db for vectorstore")

    
@router.post("/add_video", summary="add_video")
@router.post("/add_video/", include_in_schema=False)
async def video_infer(video: UploadFile = File(...),
                      chat_handler: ChatHandler = Depends(get_chat_handler)):
  
    # if file.content_type != "video/mp4": # checked by frontend
    #     return JSONResponse(status_code=400, content={"message": "Invalid file type. Only mp4 videos are allowed."})
    logging.info(f"selected db: {vectordb}")
    try:
        # Save Video file
        video_location = os.path.join(upload_dir, video.filename)
        logging.info(f"video_location: {video_location}")
        with open(video_location, "wb") as buffer:
            buffer.write(await video.read())
        
        start_time = datetime.datetime.now()
        
        # TODO: do this asynchronously, dont hang return
        # Save video description
        def save_description():
            des1 = datetime.datetime.now()
            logging.info(f"start generate description for {video_location}")
            chat_handler.upload(up_video=video_location, audio_flag=True)
            llm_message = chat_handler.ask_answer(user_message="describe the video in detail")
            
            if not os.path.exists(desc_dir):
                os.makedirs(desc_dir)
            desc_location = os.path.join(desc_dir, video.filename+".txt")
            logging.info(f"desc_location: {desc_location}")
            with open(desc_location, "w") as buffer:
                buffer.write(llm_message)
            des2 = datetime.datetime.now()
            logging.info(f"generate description time: {(des2 - des1).total_seconds()} seconds")
            
            return llm_message, desc_location
        # Video ingest
        def ingest_video():
            ingest1 = datetime.datetime.now()
            logging.info(f"start ingest video {video_location}")
            process_single_videos(video_location, image_output_dir, meta_output_dir, N, vectordb)
            global_metadata_file_path = meta_output_dir + "metadata.json"
            store_into_vectordb(metadata_file_path=global_metadata_file_path, 
                                db=vectordb, 
                                insert_url=image_insert_url,
                                video_name=video.filename)
            ingest2 = datetime.datetime.now()
            logging.info(f"ingest video time: {(ingest2 - ingest1).total_seconds()} seconds")
        
        # Run tasks concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            description_future = executor.submit(save_description)
            ingest_future = executor.submit(ingest_video)

            llm_message, desc_location = description_future.result()
            ingest_future.result()
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time.total_seconds()} seconds")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send video: {e}") from e

    return {"llm_message": llm_message,
                 "video_url": video.filename,
                 "desc_path": desc_location
                 }

@router.post("/prepare_db", summary="prepare all videos")
@router.post("/prepare_db/", include_in_schema=False)
async def video_infer(db: DB):

    logging.info(f"selected_db: {db.selected_db}")
    global vectordb
    vectordb = db.selected_db
    try:
        results = post_data(vector_init_url, {"selected_db": vectordb})
        logging.info(results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize vector db: {e}") from e
        
    # script_path = 'generate_store_embeddings.py'
    # command = ['python', script_path, '--config', 'config.yaml', '--db', vectordb, '--generate', 'False']
    # logging.info(f"command: {command}")
    # subprocess.run(command)
    
    return {"Initialize database": "success"}

@router.get("/describe_video", summary="describe video")
@router.get("/describe_video/", include_in_schema=False)
async def video_infer(name: str):
    '''
        name: name of video to be described
    '''
    desc_location = os.path.join(desc_dir, name+".txt")
    logging.info(f"desc_location: {desc_location}")
    desc = ""
    try:
        with open(desc_location, "r") as file:
            desc = file.read()
            logging.info(desc)  # Print the description for debugging purposes
            return {"description": desc}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{desc_location}' not found.")
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {e}")
   
# @router.get("/download", summary="download video")
# @router.get("/download/", include_in_schema=False)
# async def download(filename: str):
#     file_path = os.path.join(upload_dir, filename)
#     if os.path.isfile(file_path):
#         try:
#             return FileResponse(file_path, media_type="video/mp4")
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Failed to send video: {e}") from e
#     raise HTTPException(status_code=404, detail=f"File '{file_path}' not found.")
