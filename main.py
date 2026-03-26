import os

import torch
from dotenv import load_dotenv

load_dotenv()
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    cache_path = os.environ.get("HUGGINGFACE_HUB_CACHE")
    model_name = "Qwen/Qwen3.5-4B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    save_path = cache_path.replace("_Cache", "") + f"/{model_name.split('/')[1]}"

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    download_model()