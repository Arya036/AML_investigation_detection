"""
Modal deployment wrapper for SENTINEL AI (PS6) — FastAPI backend.

What this does
--------------
Deploys your EXISTING FastAPI app (api/main.py -> `app`) to Modal as a
serverless HTTPS endpoint. No credit card required; it runs on Modal's
$30/month free Starter credits.

- Scales to zero when idle  -> stays free (you only spend credits while it
  actually serves a request).
- Cold-starts on the next request after idle (first hit takes ~30-60s to load
  Torch + the embedding model; subsequent hits are fast).
- Your data (fundflow.db, the Chroma vector store, and the cached MiniLM
  embedding model) lives on a Modal VOLUME, so it PERSISTS across deploys and
  cold starts. No baking gigabytes into the image, no re-downloading the model.

You do NOT rewrite your backend. This file just imports and serves it.

Deploy in 6 commands (full beginner runbook is in DEPLOY_MODAL.md):
    pip install modal
    modal setup
    modal secret create sentinel-secrets OPENAI_API_KEY=sk-REPLACE_ME
    modal volume create sentinel-data
    modal volume put sentinel-data ./fundflow.db /fundflow.db
    modal volume put sentinel-data ./rag/vector_store /rag/vector_store
    modal deploy modal_app.py

Run this file FROM YOUR REPO ROOT (the folder that contains api/, config.py,
requirements_ps6.txt, rag/, etc.), because it copies that folder into the image.
"""

import modal

# ---------------------------------------------------------------------------
# 1. Container image: Python 3.11 + your backend deps + your source code
# ---------------------------------------------------------------------------
# NOTE: data files (fundflow.db, rag/vector_store) are deliberately NOT copied
# into the image -- they are served from the Modal Volume instead (see below).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl", "git")
    .pip_install_from_requirements("requirements_ps6.txt")
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=[
            "frontend",
            "landing",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "*.pyc",
            "fundflow.db",        # -> lives on the Volume
            "rag/vector_store",   # -> lives on the Volume
            "*.zip",
            "docs",
            ".env",               # never ship your local secrets
        ],
    )
)

app = modal.App("sentinel-ai")

# Persistent storage for the DB, the Chroma vector store, and the model cache.
# create_if_missing=True means `modal deploy` won't fail if you forgot to make
# it first -- but you still need to UPLOAD your data into it (see runbook).
data_volume = modal.Volume.from_name("sentinel-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# 2. The web endpoint: serves your existing FastAPI app, unchanged
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("sentinel-secrets")],
    cpu=1.0,
    memory=3072,           # 3 GB: enough for Torch + MiniLM + Chroma + XGBoost
    timeout=600,           # allow long agent investigations (~90s+)
    scaledown_window=300,  # stay warm 5 min after last request (fewer cold starts)
    # min_containers=1,    # <-- ENABLE ONLY on demo day. Always-on 3GB/1cpu
    #                            runs ~24/7 and will exceed the $30/mo free
    #                            credits. Leave OFF to stay free (scale-to-zero).
)
@modal.concurrent(max_inputs=50)  # if your Modal version errors here, delete this line
@modal.asgi_app()
def fastapi_app():
    import os

    # Point the backend at the Volume-backed data + a persistent model cache.
    # These override the Windows paths from your local .env.
    os.environ["PS6_DB_PATH"] = "/data/fundflow.db"
    os.environ["CHROMA_DB_PATH"] = "/data/rag/vector_store"
    # Cache the downloaded embedding model on the Volume so it downloads ONCE.
    os.environ["HF_HOME"] = "/data/.cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/data/.cache/hf"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/data/.cache/st"
    # OPENAI_API_KEY is injected automatically from the `sentinel-secrets` Secret.

    # Import your existing FastAPI instance (api/main.py -> `app`).
    from api.main import app as web_app
    return web_app


# ---------------------------------------------------------------------------
# 3. Retraining function: separate from web container (longer timeout, more RAM)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("sentinel-secrets")],
    cpu=2.0,
    memory=4096,           # 4 GB for training
    timeout=1800,          # 30 minutes
)
def retrain_model(generate_drift: bool = False, amount_shift: float = 1.5, night_boost: float = 0.30):
    """Run the retraining pipeline as a separate Modal function.

    Usage:
        modal run modal_app.py::retrain_model
        modal run modal_app.py::retrain_model --generate-drift True
    """
    import os
    os.environ["PS6_DB_PATH"] = "/data/fundflow.db"
    os.environ["MLOPS_DATA_DIR"] = "/data/mlops"

    from mlops.retrain import retrain
    result = retrain(
        drifted=generate_drift,
        shift_config={
            "amount_shift_factor": amount_shift,
            "night_hour_boost": night_boost,
        } if generate_drift else None,
    )

    # Commit the volume so the web container can see the new model
    data_volume.commit()
    print("Volume committed. Call POST /mlops/reload on the web container to load the new model.")

    return result
