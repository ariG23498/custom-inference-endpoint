import os
import io
from transformers import Mistral3ForConditionalGeneration, AutoProcessor
import torch
from text_encoding import encode_prompt
import requests
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    text_enc_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        text_enc_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    tok_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    tokenizer = AutoProcessor.from_pretrained(tok_id)

    prompt = ["hello aritra", "how are you?"]
    local_prompt_embeds, _ = encode_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
    )
    local_prompt_embeds = local_prompt_embeds.detach().cpu()

    url = f"{os.environ['ENDPOINT']}/predict"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    }
    payload = {
        "prompt": prompt,
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()

    remote_prompt_embeds = torch.load(io.BytesIO(resp.content))

    torch.testing.assert_close(
        local_prompt_embeds,
        remote_prompt_embeds,
        rtol=1e-3,
        atol=1e-3,
    )
