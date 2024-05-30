from starlette.requests import Request
from starlette.responses import StreamingResponse
from typing import Dict, Any
import torch
import ray
from ray import serve
import yaml
import time
import asyncio
from functools import partial
from queue import Empty
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TextIteratorStreamer
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

with open("./conf/conf.yaml", 'r') as f:
    configs = yaml.safe_load(f)

# dashboard addr
# http://<ip>:8265/#/overview 
ray.init(_node_ip_address="0.0.0.0", 
         dashboard_host="0.0.0.0",
         dashboard_port=8265,
         log_to_driver=True,
         num_cpus=configs["ncpus"],
         _temp_dir=dir_path,
         )

serve.start(detached=True, 
            http_options={"host": configs["main_service"]["host"], 
                          "port": configs["main_service"]["port"]})

@serve.deployment(ray_actor_options={"num_cpus": configs["ncpus"]}, 
                  route_prefix=configs["route_prefix"])
class Llama2Deployment:
    def __init__(self, model_path:str):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                       use_fast=False, 
                                                       use_auth_token="")

        self.hf_config = AutoConfig.from_pretrained(model_path,
                                                    torchscript=True,
                                                    use_auth_token="",
                                                    trust_remote_code=False)

        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          config=self.hf_config,
                                                          torch_dtype=torch.float32,
                                                          low_cpu_mem_usage=False,
                                                          use_auth_token="")
        self.model = self.model.eval().to(self.device)
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.padding_side = "left"

        # Use async loop in streaming
        self.loop = asyncio.get_running_loop()
    
    def tokenize(self, prompt: str):
        """Tokenize the input and move to CPU."""

        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return input_tokens.input_ids.to(device=self.device)
    
    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Take a prompt and generate a response."""

        input_ids = self.tokenize(prompt)
        gen_tokens = self.model.generate(input_ids, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    async def consume_streamer_async(self, streamer):
        """Consume the streamer asynchronously."""

        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)

    def streaming_generate(self, prompt: str, streamer, **config: Dict[str, Any]):
        """Generate a streamed response given an input."""

        input_ids = self.tokenize(prompt)
        self.model.generate(input_ids, streamer=streamer, **config)

    async def __call__(self, http_request: Request):
        """Handle HTTP requests."""

        # Load fields from the request
        json_request: str = await http_request.json()
        text = json_request["text"]
        # Config used in generation
        config = json_request.get("config", {})
        streaming_response = json_request["stream"]

        # Prepare prompts
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)

        # Process config
        config.setdefault("max_new_tokens", 256)

        # Non-streaming case
        if not streaming_response:
            return self.generate(prompts, **config)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
        # Convert the streamer into a generator
        self.loop.run_in_executor(
            None, partial(self.streaming_generate, prompts, streamer, **config)
        )
        return StreamingResponse(
            self.consume_streamer_async(streamer),
            status_code=200,
            media_type="text/plain",
        )

entrypoint = Llama2Deployment.bind(configs["model_path"])