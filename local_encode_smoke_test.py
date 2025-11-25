import torch
from mistral_text_encoding_core import encode_prompt
from transformers import Mistral3ForConditionalGeneration, AutoProcessor

text_enc_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    text_enc_id, dtype=torch.bfloat16, device_map="cuda"
).eval()

tok_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
tokenizer = AutoProcessor.from_pretrained(tok_id)

prompt = ["hello aritra", "how are you?"]
prompt_embeds, _ = encode_prompt(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    prompt=prompt,
)

print(f"Shape of prompt embeds = {prompt_embeds.shape}")
