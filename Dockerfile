FROM python:3.11-slim

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

# --- Python deps ---------------------------------------------------------
# 0) scikit-learn pinned — saved joblib artifacts (stacking LR / calibrator)
#    were trained on this version; unpinning risks pickle-compat warnings.
RUN pip install "scikit-learn==1.8.0"

# 1) CPU-only torch first (HF free Spaces have no GPU; CPU wheel is much smaller).
#    Installing this before requirements.txt means the bare `torch` line in
#    pipeline/requirements.txt is already satisfied and won't be re-resolved.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.0.1+cpu

# 2) TensorFlow pinned to 2.13.0 — kept out of pipeline/requirements.txt to avoid
#    breaking local dev, and pinned here for Keras .h5 compatibility with
#    aki_gru_final.h5.
RUN pip install tensorflow==2.13.0

# 3) Rest of the pipeline requirements
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install -r /app/pipeline/requirements.txt

# --- App source ----------------------------------------------------------
COPY pipeline/ /app/pipeline/
COPY demo/patients/ /app/demo/patients/

# Models are downloaded at startup from the private HF repo into this dir.
RUN mkdir -p /app/models

EXPOSE 7860

CMD ["uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "7860"]
