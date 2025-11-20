# 🚀 Deploying a Custom LLM Service (FastAPI + Docker + Inference Endpoints)

Deploying an LLM is easy with [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/main/en/index), but exposing **custom logic** (custom hidden-state extraction, custom embeddings, custom metadata) is where things get interesting.
This post walks through **building, testing, containerizing, and deploying** a custom text-encoding service using:

* **Mistral-3-Small** for encoding
* **FastAPI** for serving
* **Docker** for packaging
* **Hugging Face Inference Endpoints** for production

The goal: provide a REST endpoint that returns **custom prompt embeddings** and a **custom 4-dim text-ID tensor**, computed exactly the way *you* want.

## 1. 🧩 Why Custom LLM Functions Matter

LLM deployments typically expose generation or chat APIs. But real production systems often require **intermediate model artifacts**:

* specific hidden layers
* concatenated embeddings
* custom coordinate encodings
* pre-processing rules
* application-defined text IDs

Your `text_encoding.py` implements exactly this:
you extract hidden states from layers `(10, 20, 30)`, stack/reshape them, and attach custom `text_ids`.

To expose this as an API, we wrap your custom function inside a FastAPI service.

## 2. ⚙️ Building the FastAPI Application

### Key Points

* The model and tokenizer load **once**, inside `FastAPI`’s **lifespan** context.
* Gradients are disabled globally.
* `/predict` returns JSON-compatible tensors.
* Incoming prompts may be a string or list of strings.

### Main Server: `api.py`

Your server:

* Loads the Mistral encoder + processor at startup
* Calls **`encode_prompt()`**, your custom embedding function
* Detaches and moves tensors to CPU for JSON serialization
* Returns shapes + timing
* Runs on CUDA with `bfloat16`

This gives the endpoint:

```
POST /predict
{
    "prompt": ["hello world"]
}
```

Returns:

```
{
  "prompt_embeds": [...],
  "text_ids": [...],
  "shapes": { ... },
  "time_ms": ...
}
```


## 3. ⚡ FastAPI Startup Optimization

You correctly use:

```python
@asynccontextmanager
async def lifespan(app):
```

This ensures:

1. Models load exactly once
2. On shutdown, resources are freed
3. The service is ready before the first request arrives

This is a production-grade serving pattern.

## 4. 🧪 Local Testing with the Client

Your `client.py` POSTs prompts to the local server and validates output equivalence:

```python
torch.testing.assert_close(pe[0], pe[1], rtol=1e-4, atol=1e-4)
```

This proves determinism of your custom encoder.

Run locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
python client.py
```

You now have a **validated local API**.


## 5. 🐳 Containerizing the Service

Your `Dockerfile`:

* Uses `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`
* Installs `transformers`, `accelerate`, `fastapi`, `uvicorn`
* Creates `/models` for HF cache
* Copies your app code
* Runs service as non-root user
* Exposes port **7860**

Build:

```bash
docker build -t mistral-text-encoder .
```

Run locally:

```bash
docker run --gpus all -p 7860:7860 mistral-text-encoder
```

Test:

```bash
curl http://localhost:7860/
```

## 6. 📦 Push Container to a Registry

### Tag image

```bash
docker tag mistral-text-encoder \
    registry.huggingface.co/<username>/<repo>:latest
```

### Login

```bash
huggingface-cli login
docker login registry.huggingface.co
```

### Push

```bash
docker push registry.huggingface.co/<username>/<repo>:latest
```

Your custom container is now available for deployment.


## 7. ☁️ Deploy to Hugging Face Inference Endpoints

You would need to build a [Inference Endpoint](https://huggingface.co/docs/inference-endpoints/main/en/quick_start) and then customize with [your own container](https://huggingface.co/docs/inference-endpoints/main/en/engines/custom_container).

Deploy using the docker image that you have just created, and you are GOOD TO GO!

Your remote URL becomes:

```
https://<endpoint-name>/predict
```

Now you can replace local URL in `driver.py`:

```python
url = "<REMOTE_URL>/predict"
```

And verify equivalence between **local embeddings** and **remote embeddings**:

```python
torch.testing.assert_close(local_prompt_embeds, remote_prompt_embeds)
torch.testing.assert_close(local_text_ids, remote_text_ids)
```

This proves your endpoint is deterministic and production-ready.
