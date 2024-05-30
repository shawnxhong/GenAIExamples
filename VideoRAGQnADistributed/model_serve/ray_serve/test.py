
import requests

ip = "your_ip"

prompt = "introduce the solar system"
sample_input = {"text": prompt, "config": {}, "stream": True}

# Non-streaming response
# outputs = requests.post("http://"+ip+":8866/generate", json=sample_input, stream=False)
# print(outputs.text, flush=True)


# Streaming response
sample_input["stream"] = True
outputs = requests.post("http://"+ip+":8866/generate", json=sample_input, stream=True)

outputs.raise_for_status()
for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
    print(output, end="", flush=True)