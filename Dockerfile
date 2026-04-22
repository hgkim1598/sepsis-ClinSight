FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    LOCAL_MODEL_PATH=/app/models \
    PATIENTS_DIR=/app/demo/patients \
    PYTHONPATH=/app:/app/pipeline

WORKDIR /app

# System deps for scientific / TF / HDF5 wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
      libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (pinned in dependency order) ---------------------------
# Pinned versions derive from docs/environment_analysis.md — the values
# actually baked into models/ artifacts (sklearn 1.8.0 in stacking LRs,
# XGBoost 3.2.0 in .json, Keras 3.12.1 in aki_gru_final.h5). Later
# `pip install -r pipeline/requirements.txt` leaves these satisfied
# because those entries (torch / numpy / xgboost) are unpinned there.

# 1) numpy first — shared ABI floor for sklearn / torch / TF.
#    1.26.4 has cp312 wheels and satisfies TF 2.18 (>=1.26,<2.2) + torch 2.2 (<2).
RUN pip install "numpy==1.26.4"

# 2) CPU-only torch.
#    NOTE: user-requested 2.0.1+cpu has no cp312 wheel — bumped to the first
#    CPU release that ships cp312 (2.2.2). Matches dev machine closely enough
#    since .pt files are plain OrderedDict state_dicts (no torch version
#    embedded in the file).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2+cpu"

# 3) TensorFlow 2.18 — first stable TF line that is Keras-3-native.
RUN pip install "tensorflow==2.18.0"

# 4) Keras 3.12.1 — matches keras_version embedded in aki_gru_final.h5.
#    Installed AFTER tensorflow so it overrides TF's bundled keras.
RUN pip install "keras==3.12.1"

# 5) scikit-learn 1.8.0 — matches _sklearn_version on stacking LR / ARDS calibrator.
RUN pip install "scikit-learn==1.8.0"

# 6) XGBoost 3.2.0 — matches version tag in mortality_xgb.json / aki_xgb_final.pkl.
#    sic_xgb.json was saved with 3.1.3 but XGBoost is forward-compat on JSON format.
RUN pip install "xgboost==3.2.0"

# 7) Rest of the pipeline requirements (fastapi / uvicorn / pandas / huggingface_hub / ...).
#    torch / numpy / xgboost lines in this file are unpinned and stay satisfied
#    by the explicit installs above, so pip won't re-resolve them.
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install -r /app/pipeline/requirements.txt

# --- App source ----------------------------------------------------------
COPY pipeline/ /app/pipeline/
COPY demo/patients/ /app/demo/patients/

# Models are downloaded at startup from the private HF repo into this dir.
RUN mkdir -p /app/models

EXPOSE 7860

CMD ["uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "7860"]
