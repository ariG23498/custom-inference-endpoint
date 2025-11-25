import io
import torch
import requests

url = "http://localhost:8000/predict"

payload = {
    "prompt": ["hello", "hello"],
}
resp = requests.post(url, json=payload)

prompt_embeds = torch.load(io.BytesIO(resp.content))
print(f"Shape of prompt embeds = {prompt_embeds.shape}")
