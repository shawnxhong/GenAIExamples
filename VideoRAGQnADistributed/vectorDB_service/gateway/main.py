import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import utilities
import uvicorn
from routers import (
    health,
    crud,
    video_llama
)
from core.db_handler import DB_Handler

CONFIG_PATH = "/gateway/conf/conf.yaml"
HANDLER_PICKLE_PATH = ""

configs = utilities.read_config(CONFIG_PATH)
db_handler = DB_Handler(configs)

# Load the db & retriever object when start
HANDLER_PICKLE_PATH = configs['handler_pickle_path']
if os.path.exists(HANDLER_PICKLE_PATH):
    db_handler.load_from_pkl_file(HANDLER_PICKLE_PATH)

app = FastAPI(
    title="VectorDB retriever API",
    description="",
    redoc_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(crud.router)
app.include_router(video_llama.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)