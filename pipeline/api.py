import os
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from hf_model_loader import download_models
from mortality.predict import predict_mortality
from ARDS.ards_predict import predict_ards
from SIC.sic_predict import predict_sic
from AKI.aki_predict import predict_aki


# demo/patients 디렉토리: 환경변수로 오버라이드 가능, 기본값은 레포 루트의 demo/patients
_DEFAULT_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "patients"
PATIENTS_DIR = Path(os.getenv("PATIENTS_DIR", str(_DEFAULT_DEMO_DIR)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # HF 리포에서 모델 파일 다운로드 (이미 있으면 스킵)
    download_models()
    yield


app = FastAPI(
    title="Sepsis ICU Mortality Prediction API",
    description="패혈증 ICU 환자 사망률/ARDS/SIC/AKI 예측 파이프라인",
    version="2.0.0",
    lifespan=lifespan,
)


def _load_patient(patient_id: str):
    pdir = PATIENTS_DIR / patient_id
    if not pdir.is_dir():
        raise HTTPException(status_code=404, detail=f"환자 데이터 없음: {patient_id}")

    try:
        with open(pdir / "patient_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        for key in ["intime", "sepsis_onset_time", "window_start_vital", "window_start_lab", "window_end"]:
            if key in meta and isinstance(meta[key], str):
                meta[key] = datetime.fromisoformat(meta[key])

        vital_ts = pd.read_parquet(pdir / "vital_ts.parquet")
        lab_df   = pd.read_parquet(pdir / "lab_df.parquet")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"환자 데이터 로드 실패: {patient_id} ({e})")

    return meta, vital_ts, lab_df


# ── 환자 목록 ─────────────────────────────────────────────────
@app.get("/patients")
def list_patients():
    if not PATIENTS_DIR.is_dir():
        return {"patients": []}
    ids = sorted(
        p.name for p in PATIENTS_DIR.iterdir()
        if p.is_dir() and (p / "patient_meta.json").exists()
    )
    return {"patients": ids}


# ── Raw 데이터 조회 (프론트용) ────────────────────────────────
@app.get("/patients/{patient_id}/data")
def get_patient_data(patient_id: str):
    meta, vital_ts, lab_df = _load_patient(patient_id)
    return {
        "patient_id":   patient_id,
        "patient_meta": {k: str(v) for k, v in meta.items()},
        "vital_ts":     vital_ts.assign(charttime=vital_ts['charttime'].astype(str)).to_dict(orient='records'),
        "lab_df":       lab_df.assign(charttime=lab_df['charttime'].astype(str)).to_dict(orient='records'),
    }


# ── 추론 실행 ─────────────────────────────────────────────────
@app.post("/predict/{patient_id}")
def predict(patient_id: str):
    meta, vital_ts, lab_df = _load_patient(patient_id)

    mortality_result = predict_mortality(vital_ts, lab_df, meta, patient_id=patient_id)
    ards_result      = predict_ards(vital_ts, lab_df, meta)
    sic_result       = predict_sic(vital_ts, lab_df, meta)
    aki_result       = predict_aki(vital_ts, lab_df, meta)

    return {
        **mortality_result,
        **ards_result,
        **sic_result,
        **aki_result,
    }
