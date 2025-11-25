# Base image with PyTorch + CUDA + cuDNN
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# -------------------------------------------------------------------
# System deps (minimal)
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Env variables
# -------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/models \
    HF_HOME=/models \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# -------------------------------------------------------------------
# Install Python deps
# NOTE: torch/torchvision already come from the base image.
# -------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "transformers==4.57.1" \
        "accelerate==1.11.0" \
        fastapi \
        "uvicorn[standard]"

# -------------------------------------------------------------------
# Copy application files
# -------------------------------------------------------------------
COPY . /app

# Create cache dir for HF models
RUN mkdir -p /models

# -------------------------------------------------------------------
# Non-root user
# -------------------------------------------------------------------
RUN useradd -m -u 1000 user && \
    chown -R user:user /app /models

USER user

# -------------------------------------------------------------------
# Expose port
# -------------------------------------------------------------------
EXPOSE 7860

# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------
# If your FastAPI file is `api.py` and app instance is `app`
CMD ["uvicorn", "text_encoding_fastapi_service:app", "--host", "0.0.0.0", "--port", "7860"]
