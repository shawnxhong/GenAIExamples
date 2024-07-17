. /root/vid2text/bin/activate

# file server
python -m http.server 7999 --bind 0.0.0.0 --directory /vid2text/videos &

# process all video: fill the 2 database with prepared video

# Check the value of the GENERATE environment variable
if [ "$GENERATE" = "False" ]; then
  python generate_store_embeddings.py --config config.yaml --db chroma --generate False
  python generate_store_embeddings.py --config config.yaml --db vdms --generate False
else
  python generate_store_embeddings.py --config config.yaml --db chroma
  python generate_store_embeddings.py --config config.yaml --db vdms
fi

# service on demand on 7777: save video & desc & ingest, get decription, init database for selected db
python main.py

# the rag-agent(on behalf of ui) will ask for the video and description, after return, video to play on ui, decription->prompt to sent to LLM
