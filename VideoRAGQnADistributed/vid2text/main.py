from contextlib import asynccontextmanager
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers import (
    health,
    video_infer
)
from video_llama import ChatHandler

# align /etc/timezone and /etc/localtime
timezone_cmd = ["cat", "/etc/timezone"]
timezone_result = subprocess.run(timezone_cmd, capture_output=True, text=True, check=True)

if timezone_result.returncode == 0:
    timezone = timezone_result.stdout.strip()

    # Create symbolic link to /etc/localtime
    ln_cmd = ["ln", "-snf", f"/usr/share/zoneinfo/{timezone}", "/etc/localtime"]
    ln_result = subprocess.run(ln_cmd, capture_output=True, check=True)

    if ln_result.returncode == 0:
        print("Timezone set successfully.")
    else:
        print("Error setting timezone.")
else:
    print("Error reading timezone file.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chat_handler = ChatHandler()
    yield

app = FastAPI(
    title="Vid2Text API",
    description="",
    redoc_url="/",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(video_infer.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)