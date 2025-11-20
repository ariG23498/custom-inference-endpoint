from transformers import Mistral3ForConditionalGeneration, AutoProcessor
from typing import Union, List, Optional
import torch


def format_text_input(prompts: List[str], system_message: str = None):
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


def get_mistral_3_small_prompt_embeds(
    text_encoder: Mistral3ForConditionalGeneration,
    tokenizer: AutoProcessor,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 512,
    system_message: str = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation.""",
    hidden_states_layers: List[int] = (10, 20, 30),
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Format input messages
    messages_batch = format_text_input(prompts=prompt, system_message=system_message)

    # Process all messages at once
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )

    # Move to device
    input_ids = inputs["input_ids"].to(text_encoder.device)
    attention_mask = inputs["attention_mask"].to(text_encoder.device)

    # Forward pass through the model
    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Only use outputs from intermediate layers and stack them
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=text_encoder.dtype, device=text_encoder.device)

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


def prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: Optional[torch.Tensor] = None,
):
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)

    return torch.stack(out_ids)


def encode_prompt(
    text_encoder: Mistral3ForConditionalGeneration,
    tokenizer: AutoProcessor,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
):
    if prompt is None:
        prompt = ""

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds = get_mistral_3_small_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    text_ids = prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(text_encoder.device)
    return prompt_embeds, text_ids
