# 🚀 Deploying Custom Mistral Text Encodings with FastAPI, Docker, and 🤗 Inference Endpoints

Most model deployments expose *generation*. But production systems often need something else: **custom model internals** — hidden states, stacked embeddings, layer-specific representations, or custom token coordinate IDs.

This post walks through a minimal, reproducible pipeline to:

1. Build a **custom text-encoding function** using Mistral-3-Small
2. Serve it with **FastAPI**
3. Package it as a **Docker container**
4. Deploy it on **🤗 Inference Endpoints**
5. Verify **local vs. remote determinism**

This gives you a **production-grade** REST endpoint that returns *your* exact embeddings.

## 1. Core Idea: Keep Custom Logic Isolated

All custom encoding logic, formatting prompts, extracting intermediate hidden states, reshaping embeddings, generating token IDs, lives in a single file:

```
mistral_text_encoding_core.py
```

This keeps experimentation clean:

* You test the logic in isolation
* You import it inside FastAPI without duplicating code
* You can update the embedding logic without touching the API or Dockerfiles

To run it locally:

```bash
python local_encode_smoke_test.py
```

You should see a shape like:

```
Shape of prompt embeds = torch.Size([2, 512, 15360])
```

If this works, everything downstream will work.

## 2. Serving the Encoder with FastAPI

The service is built in:

```
text_encoding_fastapi_service.py
```

The FastAPI server:

* Loads Mistral once at startup
* Exposes a `/predict` endpoint
* Runs the custom `encode_prompt()` from `mistral_text_encoding_core.py`
* Serializes embeddings to bytes using `torch.save`
* Returns them as `application/octet-stream`

To start the API locally:

> [!NOTE]
> When you want to run the FastAPI service locally make sure to make the following change in the `text_encoding_fastapi_service.py`

```diff
- TEXT_ENCODER_ID = os.getenv("TEXT_ENCODER_ID", "/repository")
+ TEXT_ENCODER_ID = os.getenv("TEXT_ENCODER_ID", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
```

```bash
uvicorn text_encoding_fastapi_service:app --host 0.0.0.0 --port 8000
```

Test it with:

```
python fastapi_smoke_test.py
```

If shapes match your local smoke test, you are ready for deployment.

## 3. Containerizing the Service

The Dockerfile bundles:

* PyTorch + CUDA (from the base image)
* FastAPI + Uvicorn
* Transformers + Accelerate
* Your encoding logic and API

Build the image:

```bash
docker build -t mistral-endpoint . --platform linux/amd64
```

Run locally with GPU:

```bash
docker run --gpus all -p 7860:7860 mistral-endpoint
```

Check health:

```bash
curl http://localhost:7860/health
```

If the model loads and dtype/device look correct, we can proceed.

## 4. Publishing the Container

Push the image:

```bash
docker tag mistral-endpoint youruser/mistral-endpoint:v1
docker push youruser/mistral-endpoint:v1
```

You now have a container accessible from Inference Endpoints.

## 7. ☁️ Deploy to Hugging Face Inference Endpoints

The first step is to create an endpoint.
Head over to [inference endpoints](https://endpoints.huggingface.co/) and click on the "+ New" button.

| ![endpoint creation button](./assets/new-endpoint.png) |
| :--: |
| Click on the "new" button to create an endpoint |

You will then have a pop up screen for the models to choose from. Choose the `mistralai/Mistral-Small-3.2-24B-Instruct-2506` model from the Hub. This will load the model to the `/repository`
folder that we will use for model loading instead of downloading the model over internet (saving loading time and startup of the endpoint).

| ![choose the model to deploy from](./assets/deploy-from.png) |
| :--: |
| Choose any model you like here |

We will now select the "Configure" button to go to the next step.

| ![configure button](./assets/configure.png) |
| :--: |
| Configure the deployment |

You will be taken to the next page where you can select the Hardware (we select one A100), Authentication (we select authenticated) and other configurations.
Here we scroll down to the "Inference Engine" section and select "Custom".

| ![custom inference engine](./assets/inference-engine.png) |
| :--: |
| Custom inference engine |

As you can see, we fill the "Container URL" with our custom docker URL, and expose the container port.

* **Container URL:** `docker.io/youruser/mistral-text-encoder:v1`
* **Container Port:** `7860`

After this you should be good to hit the "Create Endpoint" button. Once the deployment succeeds your remote URL becomes:

```
https://<endpoint-name>/predict
```

Now you can replace local URL in [`driver.py`](./driver.py):

```python
url = "https://<name>.endpoints.huggingface.cloud/predict"
```

## 6. Verifying Local vs Remote Embeddings

Use the verification script:

```
validate_inference_endpoint_equivalence.py
```

Create a `.env`:

```
HF_TOKEN=hf_your_token
ENDPOINT=https://<name>.endpoints.huggingface.cloud
```

Run:

```bash
python validate_inference_endpoint_equivalence.py
```

Expected output:

```
✅ local and remote prompt_embeds match
```

This confirms deterministic behavior between:

* your local PyTorch environment
* your containerized FastAPI service
* the running Inference Endpoint

This end-to-end consistency is critical for downstream pipelines.

# ✔️ Important Caveats

These are subtle details that matter in real deployments:

### 1. The model selected in the Endpoint UI *is not a no-op*

* Inference Endpoints download that model into **`/repository`** *inside your container*
* This makes `from_pretrained("/repository")` extremely fast
* If you choose a different model in the UI → your embeddings will change
* Keep UI selection & container code in sync

### 2. Cold starts return **503**, not errors

Inference Endpoints can sleep when idle.
You will see:

```
503 Service Unavailable
```

until the container starts.

The driver uses retry logic with exponential backoff — keep this.

### 3. Authentication is required

Calling the endpoint requires:

```
Authorization: Bearer <HF_TOKEN>
```

Even free Hugging Face accounts work, but:

* Tokens must have `inference` scope
* Private endpoints require proper token permissions
