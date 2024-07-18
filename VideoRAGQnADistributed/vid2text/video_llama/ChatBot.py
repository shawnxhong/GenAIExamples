import argparse
from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

from singleton_decorator import singleton
import os


def set_proxy(addr:str):
    # for DNS: "http://child-prc.intel.com:913"
    # for Huggingface downloading: "http://proxy-igk.intel.com:912"
    os.environ['http_proxy'] = addr
    os.environ['https_proxy'] = addr
    os.environ['HTTP_PROXY'] = addr
    os.environ['HTTPS_PROXY'] = addr

@singleton
class ChatHandler:

    def __init__(self):
        set_proxy("http://proxy-igk.intel.com:912")
        args = self.parse_args()
        self.num_beams = 1
        self.temperature = 1.0
        print('set num_beams: {}'.format(self.num_beams))
        print('set temperature: {}'.format(self.temperature))

        self.chat_state = conv_llava_llama_2.copy()
        self.img_list = list()

        print('Initializing Chat')
        cfg = Config(args)
        model_config = cfg.model_cfg
        # model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.chat = Chat(model, vis_processor, device='cpu')
        set_proxy("http://child-prc.intel.com:913")
        print('Initialization Finished')
    
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--model_type", type=str, default='LLama2', help="The type of LLM")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args()
        return args

    def upload(self, up_img=False, up_video=False, audio_flag=False):
        if up_img and not up_video:
            self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            self.chat.upload_img(up_img, self.chat_state, self.img_list)
        elif not up_img and up_video:
            self.chat_state.system =  ""
            if audio_flag:
                self.chat.upload_video(up_video, self.chat_state, self.img_list)
            else:
                self.chat.upload_video_without_audio(up_video, self.chat_state, self.img_list)

    def ask_answer(self, user_message):
        self.chat.ask(user_message, self.chat_state)
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       num_beams=self.num_beams,
                                       temperature=self.temperature,
                                       max_new_tokens=300,
                                       max_length=2000)[0]

        return llm_message

    def reset(self):
        if self.chat_state is not None:
            self.chat_state.messages = list()
        if self.img_list is not None:
            self.img_list = list()
