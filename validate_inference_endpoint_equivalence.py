import os
import io
import time
from transformers import Mistral3ForConditionalGeneration, AutoProcessor
import torch
from mistral_text_encoding_core import encode_prompt
import requests
from dotenv import load_dotenv

load_dotenv()


def post_with_retries(
    url: str,
    json: dict,
    headers: dict,
    max_retries: int = 20,
    initial_delay: float = 5.0,  # seconds
    backoff_factor: float = 1.5,  # gentler growth
    max_wait_minutes: float = 10.0,  # total wall-clock budget
    timeout: float = 60.0,  # per-request timeout
) -> requests.Response:
    """
    POST with retry logic:
    - retry on HTTP 503
    - retry on network errors
    - exponential-ish backoff between attempts
    - stop after max_wait_minutes of total waiting
    """
    delay = initial_delay
    deadline = time.time() + max_wait_minutes * 60.0

    for attempt in range(1, max_retries + 1):
        # If we've already used up our total wait budget, bail out
        if time.time() > deadline:
            raise TimeoutError(
                f"Exceeded max_wait_minutes={max_wait_minutes} before getting a successful response."
            )

        try:
            resp = requests.post(url, json=json, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            # Network-level error
            if attempt == max_retries:
                raise
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Exceeded max_wait_minutes={max_wait_minutes} after request error."
                )
            sleep_for = min(delay, remaining)
            print(
                f"[attempt {attempt}] request error: {e}, retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)
            delay *= backoff_factor
            continue

        if resp.status_code == 503:
            # Likely cold start / scaled-to-zero
            if attempt == max_retries:
                resp.raise_for_status()
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Exceeded max_wait_minutes={max_wait_minutes} with repeated 503s."
                )
            sleep_for = min(delay, remaining)
            print(f"[attempt {attempt}] got 503, retrying in {sleep_for:.1f}s...")
            time.sleep(sleep_for)
            delay *= backoff_factor
            continue

        # Non-503: either OK or a hard error
        resp.raise_for_status()
        return resp

    # Shouldn’t really get here, but just in case:
    raise RuntimeError("Exhausted retries without a successful response.")


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
    payload = {"prompt": prompt}

    resp = post_with_retries(
        url=url,
        json=payload,
        headers=headers,
        max_wait_minutes=10.0,  # tweak this: e.g. 5, 10, 15...
        initial_delay=5.0,
        backoff_factor=2,
    )

    remote_prompt_embeds = torch.load(io.BytesIO(resp.content))

    torch.testing.assert_close(
        local_prompt_embeds,
        remote_prompt_embeds,
        rtol=1e-3,
        atol=1e-3,
    )
    print("✅ local and remote prompt_embeds match")
