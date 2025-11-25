import os
import io
import time
import logging
from typing import List, Union, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
# Config
# ------------------------------------------------------
TEXT_ENCODER_ID = os.getenv("TEXT_ENCODER_ID", "/repository")
TOKENIZER_ID = os.getenv(
    "TOKENIZER_ID", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
DTYPE = torch.bfloat16


# ------------------------------------------------------
# Model Manager
# ------------------------------------------------------
class ModelManager:
    """Manages loading/unloading and access to the Mistral text encoder + tokenizer."""

    def __init__(
        self,
        text_encoder_id: str = TEXT_ENCODER_ID,
        tokenizer_id: str = TOKENIZER_ID,
        device_map: str = DEVICE_MAP,
        dtype: torch.dtype = DTYPE,
    ):
        self.text_encoder_id = text_encoder_id
        self.tokenizer_id = tokenizer_id
        self.device_map = device_map
        self.dtype = dtype

        self.text_encoder: Optional[Mistral3ForConditionalGeneration] = None
        self.tokenizer: Optional[AutoProcessor] = None

    async def load(self):
        """Load the text encoder and tokenizer into memory."""
        logger.info("🔄 Loading models...")

        t0 = time.time()
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.text_encoder_id,
            dtype=self.dtype,
            device_map=self.device_map,
        ).eval()
        logger.info(
            "✅ Loaded Mistral text encoder (%.2fs) dtype=%s device=%s",
            time.time() - t0,
            self.text_encoder.dtype,
            self.device_map,
        )

        t1 = time.time()
        self.tokenizer = AutoProcessor.from_pretrained(self.tokenizer_id)
        logger.info("✅ Loaded tokenizer in %.2fs", time.time() - t1)

        torch.set_grad_enabled(False)

    async def unload(self):
        """Cleanup method to properly unload models."""
        logger.info("🧹 Cleaning up model resources...")
        try:
            if self.text_encoder is not None:
                del self.text_encoder
                self.text_encoder = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Model components unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model components: {str(e)}")

    def get_models(self):
        """Return loaded models or raise if not available."""
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self.text_encoder, self.tokenizer


model_manager = ModelManager()


# ------------------------------------------------------
# Lifespan (startup + shutdown)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    try:
        yield
    finally:
        await model_manager.unload()


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


# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Mistral Text Encoder API is running."}


@app.get("/health")
def health():
    """Simple health endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.text_encoder is not None
        and model_manager.tokenizer is not None,
        "device_map": model_manager.device_map,
        "dtype": str(model_manager.dtype),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        text_encoder, tokenizer = model_manager.get_models()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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

    # Save tensor to bytes
    buffer = io.BytesIO()
    torch.save(prompt_embeds.cpu(), buffer)
    buffer.seek(0)

    # Clear GPU cache
    del prompt_embeds, text_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
    )
