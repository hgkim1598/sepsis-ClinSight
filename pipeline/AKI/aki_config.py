import os

# ── 로컬 모델 경로 / HF 리포 ──────────────────────────────────
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', '/app/models')
HF_REPO_ID       = os.getenv('HF_REPO_ID', 'hgkim1598/sepsis-clinsight-models')
THRESHOLD        = 0.5
SEQ_LEN          = 12

# ── 피처 정의 ─────────────────────────────────────────────────
# 13개 raw 피처 (순서 고정)
RAW_COLS = [
    'heart_rate', 'mbp', 'resp_rate', 'temperature', 'spo2',
    'creatinine', 'bun', 'bicarbonate', 'sodium', 'potassium',
    'glucose', 'urine_output', 'glucose_vital'
]

N_RAW     = len(RAW_COLS)   # 13
N_GRU_CH  = N_RAW * 4      # raw + delta + mean + std = 52
N_XGB_CH  = N_RAW * 5      # raw + delta + mean + std + mask = 65
N_XGB_FLAT = SEQ_LEN * N_XGB_CH  # 780