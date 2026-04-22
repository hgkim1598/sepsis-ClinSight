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

# --- Python deps (pinned to match the team's training .venv) -------------
# Versions mirror the pip-freeze of the environment the models/ artifacts
# were produced in, so inference runs in the same ABI. The explicit pins
# below are duplicated in pipeline/requirements.txt — the resolver treats
# them as already-satisfied when the bulk install runs at the end.

# 1) numpy — shared ABI floor for sklearn / torch / TF.
RUN pip install "numpy==2.4.4"

# 2) CPU-only torch from the official PyTorch index (HF free Spaces have no
#    GPU; the CPU wheel is much smaller than the default CUDA build).
#    Local-version tag `+cpu` still satisfies the bare `torch==2.11.0` line
#    downstream thanks to PEP 440 local-version handling.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu"

# 3) TensorFlow — Keras-3-native release line matching the team venv.
RUN pip install "tensorflow==2.21.0"

# 4) Keras — matches keras_version embedded in aki_gru_final.h5.
#    Installed AFTER tensorflow so it overrides TF's bundled keras.
RUN pip install "keras==3.14.0"

# 5) scikit-learn — matches _sklearn_version on stacking LR / ARDS calibrator.
RUN pip install "scikit-learn==1.8.0"

# 6) Remaining deps (xgboost / shap / scipy / pandas / pyarrow / joblib /
#    fastapi / uvicorn / huggingface_hub). Steps 1–5 above are already
#    satisfied, so pip won't re-resolve those lines.
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install -r /app/pipeline/requirements.txt

# --- App source ----------------------------------------------------------
COPY pipeline/ /app/pipeline/
COPY demo/patients/ /app/demo/patients/

# Models are downloaded at startup from the private HF repo into this dir.
RUN mkdir -p /app/models

EXPOSE 7860

CMD ["uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "7860"]
