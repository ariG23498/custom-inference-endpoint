import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import Mistral3ForConditionalGeneration, AutoProcessor
import torch
from text_encoding import encode_prompt
import requests

if __name__ == "__main__":
    text_enc_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        text_enc_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    tok_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    tokenizer = AutoProcessor.from_pretrained(tok_id)

    prompt = ["hello aritra", "how are you?"]
    local_prompt_embeds, local_text_ids = encode_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
    )
    local_prompt_embeds = local_prompt_embeds.detach().cpu()
    local_text_ids = local_text_ids.detach().cpu()

    url = "https://kcqrahk0dlxhl2zs.us-east-1.aws.endpoints.huggingface.cloud/predict"
    payload = {
        "prompt": prompt,
    }
    resp = requests.post(url, json=payload).json()
    remote_prompt_embeds = torch.tensor(resp["prompt_embeds"], dtype=torch.bfloat16)
    remote_text_ids = torch.tensor(resp["text_ids"], dtype=torch.int64)

    torch.testing.assert_close(
        local_prompt_embeds,
        remote_prompt_embeds,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(local_text_ids, remote_text_ids)
