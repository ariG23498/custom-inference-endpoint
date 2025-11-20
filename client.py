import torch
import requests

url = "http://localhost:8000/predict"

payload = {
    "prompt": ["hello", "hello"],
}

resp = requests.post(url, json=payload).json()

pe = torch.tensor(resp["prompt_embeds"], dtype=torch.float32)
ti = torch.tensor(resp["text_ids"], dtype=torch.int32)

print(pe.shape)  # e.g. torch.Size([2, 512, 15360])
print(ti.shape)  # e.g. torch.Size([2, 512, 4])

torch.testing.assert_close(
    pe[0],
    pe[1],
    rtol=1e-4,
    atol=1e-4,
)
