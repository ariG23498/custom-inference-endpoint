import os
import time
import logging
from typing import List, Union, Optional, Dict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Mistral3ForConditionalGeneration, AutoProcessor

from text_encoding import encode_prompt

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("mistral-text-encoding-api")

# ------------------------------------------------------
# Global caches
# ------------------------------------------------------
text_encoder: Optional[Mistral3ForConditionalGeneration] = None
tokenizer: Optional[AutoProcessor] = None

# ------------------------------------------------------
# Config
# ------------------------------------------------------
TEXT_ENCODER_ID = os.getenv(
    "TEXT_ENCODER_ID", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
)
TOKENIZER_ID = os.getenv(
    "TOKENIZER_ID", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
DTYPE = torch.bfloat16


# ------------------------------------------------------
# Lifespan (startup + shutdown)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_encoder, tokenizer

    logger.info("🔄 Loading models...")

    t0 = time.time()
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        TEXT_ENCODER_ID,
        dtype=DTYPE,
        device_map=DEVICE_MAP,
    ).eval()
    logger.info(
        "✅ Loaded Mistral text encoder (%.2fs) dtype=%s device=%s",
        time.time() - t0,
        text_encoder.dtype,
        DEVICE_MAP,
    )

    t1 = time.time()
    tokenizer = AutoProcessor.from_pretrained(TOKENIZER_ID)
    logger.info("✅ Loaded tokenizer in %.2fs", time.time() - t1)

    torch.set_grad_enabled(False)

    yield

    logger.info("🧹 Cleaning up model resources...")
    del text_encoder
    del tokenizer


# ------------------------------------------------------
# FastAPI app
# ------------------------------------------------------
app = FastAPI(
    title="Mistral Text Encoding API",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------
class PredictRequest(BaseModel):
    prompt: Union[str, List[str]]


class PredictResponse(BaseModel):
    # (B_out, L, D)
    prompt_embeds: List[List[List[float]]]
    # (B_out, L, 4)
    text_ids: List[List[List[int]]]
    # for introspection/debug
    shapes: Dict[str, List[int]]
    time_ms: float


# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Mistral Text Encoder API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global text_encoder, tokenizer

    if text_encoder is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()

    prompt = req.prompt
    num_prompts = 1 if isinstance(prompt, str) else len(prompt)

    logger.info(
        "📝 /predict | prompts=%d ",
        num_prompts,
    )

    prompt_embeds, text_ids = encode_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    duration = (time.time() - t0) * 1000.0

    logger.info(
        "✨ Encoded in %.2f ms | prompt_embeds.shape=%s | text_ids.shape=%s",
        duration,
        tuple(prompt_embeds.shape),
        tuple(text_ids.shape),
    )

    # Keep original shapes; just make them JSON-serializable
    pe_cpu = prompt_embeds.detach().cpu().to(torch.float32)
    ti_cpu = text_ids.detach().cpu().to(torch.int32)

    prompt_embeds_out = pe_cpu.tolist()  # List[B_out][L][D]
    text_ids_out = ti_cpu.tolist()  # List[B_out][L][4]

    shapes = {
        "prompt_embeds": list(pe_cpu.shape),  # [B_out, L, D]
        "text_ids": list(ti_cpu.shape),  # [B_out, L, 4]
    }

    return PredictResponse(
        prompt_embeds=prompt_embeds_out,
        text_ids=text_ids_out,
        shapes=shapes,
        time_ms=duration,
    )
