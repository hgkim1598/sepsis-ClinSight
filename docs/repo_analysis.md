# ICU-sepsis-ClinSight 레포지토리 분석 보고서

> 분석 대상 경로: `c:/Users/hgkim/pre-project/ICU-sepsis-ClinSight`
> 분석일: 2026-04-21
> 대상 브랜치: `main`

---

## 1. 전체 디렉토리 구조

`.git/`, `.venv/`, `__pycache__/` 제외.

```
ICU-sepsis-ClinSight/
├── .gitignore
├── README.md
├── requirements.txt                     # 루트 통합 requirements (Frontend + Backend)
│
├── aki/                                 # 학습 스크립트 (pipeline 바깥, 실험용)
│   ├── aki_eicu_weighted_ensemble_external.py
│   └── stacking_aki_models_proper.py
│
├── dashboard/                           # Streamlit 프론트엔드
│   ├── .env                             # API_BASE_URL 환경변수 (git에 포함됨: .gitignore의 `.env`는 `ENV/`만 걸러내지 못함 → 실제 커밋됨)
│   ├── .env.example
│   ├── README.md
│   ├── api_client.py                    # FastAPI 호출 + mock fallback
│   ├── app.py                           # Streamlit 앱 엔트리 (≈1920라인)
│   ├── feature_labels.py                # 피처 한글 라벨
│   └── requirements.txt
│
├── data/                                # 실제 데이터는 git에 없음
│   └── README.md
│
├── docs/
│   ├── .gitkeep
│   └── repo_analysis.md                 # (본 보고서)
│
└── pipeline/                            # FastAPI 백엔드 + 모델 추론
    ├── README.md
    ├── requirements.txt
    ├── api.py                           # FastAPI 앱 (엔드포인트 정의)
    │
    ├── AKI/                             # AKI (급성신손상) 모델
    │   ├── README.md
    │   ├── aki_config.py
    │   ├── aki_loader.py
    │   ├── aki_predict.py
    │   └── aki_preprocess.py
    │
    ├── ARDS/                            # ARDS (급성호흡곤란증후군) 모델
    │   ├── README.md
    │   ├── ards_config.py
    │   ├── ards_loader.py
    │   ├── ards_model.py
    │   ├── ards_predict.py
    │   ├── ards_preprocess.py
    │   └── ards_train.py
    │
    ├── SIC/                             # SIC (패혈증 유발 응고병증) 모델
    │   ├── README.md
    │   ├── model.py
    │   ├── predict.py                   # 구버전 predict.py (직접 S3 호출)
    │   ├── sic_config.py
    │   ├── sic_loader.py
    │   ├── sic_model.py
    │   ├── sic_predict.py               # API에서 import 되는 현재 버전
    │   ├── sic_preprocess.py
    │   └── train.py
    │
    └── mortality/                       # Mortality (사망) 모델
        ├── README.md
        ├── __main__.py
        ├── config.py
        ├── history.py                   # 이전 추론 결과 S3 저장/로드
        ├── loader.py
        ├── model.py
        ├── predict.py
        └── preprocess.py
```

---

## 2. 모델 파일 목록

**레포지토리 내에는 `.pt`, `.pth`, `.pkl`, `.joblib` 파일이 단 하나도 없습니다.**

모든 모델 아티팩트는 **AWS S3 버킷 `say2-1team` 하위 `pipeline/final_model/` 프리픽스에서 런타임에 로드**하도록 구성되어 있습니다. (코드 기반으로 추정되는 S3 키 목록은 아래 섹션 3 참고.)

### 각 모듈이 기대하는 모델 파일명 (S3 키 기준)

| 모델 | S3 키 (MODEL_PREFIX=`pipeline/final_model` 기준) | 로컬 경로 (USE_S3=false) | 포맷 |
|---|---|---|---|
| Mortality — BiLSTM | `pipeline/final_model/mortality_bilstm.pt` | `./models/mortality_bilstm.pt` | PyTorch state_dict |
| Mortality — XGBoost | `pipeline/final_model/mortality_xgb.json` | `./models/mortality_xgb.json` | XGBoost JSON |
| Mortality — Stacking LR | `pipeline/final_model/mortality_stacking_lr.pkl` | `./models/mortality_stacking_lr.pkl` | joblib(sklearn) |
| ARDS — 통합 artifact | `pipeline/final_model/ards_XGB.joblib` | `./models/ards/ards_XGB.joblib` | joblib (dict: base_model/calibrator/features/threshold) |
| SIC — BiLSTM | `pipeline/final_model/sic_bilstm.pt` | `./models/sic/sic_bilstm.pt` | PyTorch state_dict |
| SIC — XGBoost | `pipeline/final_model/sic_xgb.json` | `./models/sic/sic_xgb.json` | XGBoost JSON |
| SIC — Stacking LR | `pipeline/final_model/sic_stacking_lr.pkl` | `./models/sic/sic_stacking_lr.pkl` | joblib |
| AKI — GRU | `pipeline/final_model/aki_gru_final.h5` | `./models/aki/aki_gru_final.h5` | Keras/TF HDF5 |
| AKI — XGBoost | `pipeline/final_model/aki_xgb_final.pkl` | `./models/aki/aki_xgb_final.pkl` | joblib |

**별도 유의사항:**
- `pipeline/SIC/predict.py` (구버전) 는 다른 경로 사용: `Final_model/saved_models/sic/fold_{k}/best_model.pt`, `fold_{k}/model.json`, `fold_1/feature_names.json`, `meta/meta_model.pkl`, `scaler.pkl` — 현재 `api.py`가 import하는 쪽은 `sic_predict.py`이므로 비활성.
- 환자 이력 저장: `pipeline/mortality/history.py` 가 S3에 `pipeline/patient_history/{patient_id}/latest.json` 덮어쓰기.

---

## 3. 백엔드 (`pipeline/`) 분석

### 3-1. FastAPI 엔드포인트 (`pipeline/api.py`)

| 메서드 | 경로 | 파라미터 | 반환값 |
|---|---|---|---|
| `GET` | `/patients` | 없음 | `{"patients": ["<patient_id>", ...]}` — S3 `pipeline/patients/` 하위 prefix를 열거 |
| `GET` | `/patients/{patient_id}/data` | path: `patient_id` (str) | `{"patient_id": str, "patient_meta": {...문자열화된 meta...}, "vital_ts": [records], "lab_df": [records]}` |
| `POST` | `/predict/{patient_id}` | path: `patient_id` (str) | 4개 모델 결과 병합 dict. 키: `mortality`, `ards`, `sic`, `aki` |

FastAPI 앱 메타:
- title: `"Sepsis ICU Mortality Prediction API"`
- version: `"1.0.0"`

### 3-2. 환자 데이터 로딩 경로

`/predict/{patient_id}` 호출 시 [pipeline/api.py:32](pipeline/api.py#L32) `_load_patient()` 가 S3에서 3개 객체를 가져옴:

- `s3://say2-1team/pipeline/patients/{patient_id}/patient_meta.json`
- `s3://say2-1team/pipeline/patients/{patient_id}/vital_ts.parquet`
- `s3://say2-1team/pipeline/patients/{patient_id}/lab_df.parquet`

`patient_meta.json`의 datetime 필드들(`intime`, `sepsis_onset_time`, `window_start_vital`, `window_start_lab`, `window_end`)은 ISO 문자열 → `datetime`으로 복원.

### 3-3. 각 모델이 로드되는 파일 / 함수

| 모델 | 호출 진입점 (from `api.py`) | 모델 로더 | 로드 대상 파일 |
|---|---|---|---|
| Mortality | `predict_mortality()` ([pipeline/mortality/predict.py](pipeline/mortality/predict.py)) | [pipeline/mortality/loader.py](pipeline/mortality/loader.py) `_load_models()` | `mortality_bilstm.pt` + `mortality_xgb.json` + `mortality_stacking_lr.pkl` |
| ARDS | `predict_ards()` ([pipeline/ARDS/ards_predict.py](pipeline/ARDS/ards_predict.py)) | [pipeline/ARDS/ards_loader.py](pipeline/ARDS/ards_loader.py) `_load_artifact()` | `ards_XGB.joblib` (dict) |
| SIC | `predict_sic()` ([pipeline/SIC/sic_predict.py](pipeline/SIC/sic_predict.py)) | [pipeline/SIC/sic_loader.py](pipeline/SIC/sic_loader.py) `_load_models()` | `sic_bilstm.pt` + `sic_xgb.json` + `sic_stacking_lr.pkl` |
| AKI | `predict_aki()` ([pipeline/AKI/aki_predict.py](pipeline/AKI/aki_predict.py)) | [pipeline/AKI/aki_loader.py](pipeline/AKI/aki_loader.py) `_load_models()` | `aki_gru_final.h5` + `aki_xgb_final.pkl` |

각 로더는 **모듈 전역 변수로 모델을 캐시**(첫 호출에만 로드, 이후 재사용).

### 3-4. 예측 시 입력 데이터 형태

공통적으로 각 `predict_*` 함수는 `(vital_ts: pd.DataFrame, lab_df: pd.DataFrame, patient_meta: dict)` 세 인수를 받음.

#### vital_ts — 활력징후 시계열 (`charttime` 포함)
모델별로 사용하는 컬럼 서브셋:

| 컬럼 | Mortality | ARDS | SIC | AKI |
|---|:-:|:-:|:-:|:-:|
| charttime | ✓ | ✓ | ✓ | ✓ |
| heart_rate | ✓ | ✓ |   | ✓ |
| mbp | ✓ | ✓ |   | ✓ |
| sbp | ✓ | ✓ |   |   |
| dbp | ✓ |   |   |   |
| resp_rate | ✓ | ✓ |   | ✓ |
| spo2 | ✓ | ✓ |   | ✓ |
| temperature | ✓ | ✓ |   | ✓ |
| gcs | ✓ |   |   |   |
| pao2fio2ratio | ✓ |   |   |   |
| map |   |   | ✓ |   |
| pao2 / fio2 |   |   | ✓ |   |
| glucose_vital |   |   |   | ✓ |

#### lab_df — 혈액검사 시계열 (`charttime` 포함)

| 컬럼 | Mortality | ARDS | SIC | AKI |
|---|:-:|:-:|:-:|:-:|
| lactate | ✓ | ✓ | ✓ |   |
| creatinine | ✓ | ✓ | ✓ | ✓ |
| bun | ✓ | ✓ |   | ✓ |
| sodium / potassium / glucose | ✓ |   |   | ✓ |
| bicarbonate | ✓ | ✓ |   | ✓ |
| albumin | ✓ |   |   |   |
| wbc | ✓ | ✓ | ✓ |   |
| platelet | ✓ | ✓ |   |   |
| hemoglobin | ✓ |   |   |   |
| bilirubin_total | ✓ |   | ✓ |   |
| ph |   | ✓ |   |   |
| rdw / aptt / inr |   |   | ✓ |   |
| urine_output |   |   |   | ✓ |

> ARDS는 lab_df 내부의 `{lactate, ph, bicarbonate}`를 bg(혈액가스)로 **자동 분리**.

#### patient_meta — dict

| 키 | Mortality | ARDS | SIC | AKI |
|---|:-:|:-:|:-:|:-:|
| age | ✓ | ✓ | ✓ |   |
| gender (1=M, 0=F) | ✓ | ✓ (또는 `gender_bin`) | ✓ (→ `sex_male`) |   |
| intime | ✓ |   | ✓ | ✓ |
| sepsis_onset_time | ✓ | ✓ (또는 `onset_time`) | ✓ | ✓ |
| window_start_vital / window_start_lab / window_end | ✓ |   |   |   |
| flag_liver_failure, flag_ckd, flag_coagulopathy, flag_diabetes, flag_immunosuppression, flag_chf, flag_septic_shock_hx |   |   | ✓ (없으면 0) |   |

### 3-5. 예측 결과 반환 형태

`POST /predict/{patient_id}` 의 최상위 응답은 4개 모델 블록을 평탄 병합한 dict:

```json
{
  "mortality": { ... },
  "ards":      { ... },
  "sic":       { ... },
  "aki":       { ... }
}
```

각 블록의 공통 필드 구조 (모델별로 일부 차이):

```jsonc
{
  "<model_key>": {
    "probability":    0.0 ~ 1.0,                   // 모든 모델
    "prediction":     0 | 1,                        // probability >= threshold
    "threshold":      0.21 (mort) / 0.30 (ards) / 0.5 (sic,aki),
    "inference_time": "<ISO-8601 UTC>",
    "clinical_indicators": {                        // 모든 모델 (키 집합은 모델별로 상이)
      "<key>": {
        "value": <float|int|None>,
        "reference": {
          "unit": "...",
          "usual_range": "...",
          "risk_value": true | false | null          // ards/sic/aki 일부에서 계산됨
        }
      }
    },
    "data_quality": {
      "n_vital_slots":      <int>,   // mortality, ards
      "n_lab_measurements": <int>,   // mortality, ards
      "n_slots":            <int>,   // sic, aki
      "is_reliable":        <bool>   // 공통
    },
    "feature_values": [ { "feature", "raw_value", "shap_value", "unit",
                          "is_imputed", "change", "change_direction" }, ... ], // mortality, ards
    "top_features":   [ ... 상위 3개 ],             // 모든 모델
    "shap":           [ { "feature", "shap_value" }, ... ]  // sic, aki (전체 리스트)
  }
}
```

모델별 주요 차이:
- **Mortality**: `feature_values`, `top_features` 모두 반환 + 이전 추론과의 `change`, `change_direction` 포함 (S3 `pipeline/patient_history/` 에서 직전 기록 조회).
- **ARDS**: `feature_values`(43개) + `top_features`. `is_imputed`는 항상 `false`, `change`는 `null`, `change_direction`은 `"unknown"` (이력 기능 없음).
- **SIC**: `shap` (STATIC_COLS 기반 전체 리스트) + `top_features`. `feature_values` 없음.
- **AKI**: `shap` (780차원 feature × stat × time 형식 — `{col}_{raw|delta|mean|std|mask}_t{0..11}`) + `top_features`.

**임상 핵심 지표(`clinical_indicators`) 키 집합:**
- Mortality: `ventilation`, `norepinephrine`, `dopamine`, `dobutamine`, `epinephrine`
- ARDS: `po2`, `fio2_bg`, `pao2fio2ratio`, `peep_feat`
- SIC: `platelet`, `inr`
- AKI: `lactate`, `spo2`, `gcs`, `sbp`, `wbc`, `hemoglobin`

---

## 4. 프론트엔드 (`dashboard/`) 분석

### 4-1. API 호출 코드 위치

모든 HTTP 호출은 [dashboard/api_client.py](dashboard/api_client.py)에 집중되어 있고, [dashboard/app.py](dashboard/app.py)는 여기서 정의된 함수만 사용.

| 함수 | 위치 | HTTP | 엔드포인트 | 호출처 |
|---|---|---|---|---|
| `fetch_patients()` | [api_client.py:229](dashboard/api_client.py#L229) | GET | `{API_BASE_URL}/patients` | [app.py:1809](dashboard/app.py#L1809) — 앱 진입 시 환자 ID 목록 |
| `fetch_patient_data(pid)` | [api_client.py:243](dashboard/api_client.py#L243) | GET | `{API_BASE_URL}/patients/{pid}/data` | [app.py:1876](dashboard/app.py#L1876) — 환자 바 표시용 (age/gender/intime/onset) |
| `fetch_predictions(pid)` | [api_client.py:255](dashboard/api_client.py#L255) | POST | `{API_BASE_URL}/predict/{pid}` | [app.py:1856](dashboard/app.py#L1856) — 4개 모델 예측 결과 일괄 |
| `fetch_dashboard_data(...)` | [api_client.py:537](dashboard/api_client.py#L537) | — | (HTTP 호출 없음, 로컬 병합) | [app.py:1866](dashboard/app.py#L1866) — predictions + mock 병합 |

추가로 `MODEL_PREDICT_ENDPOINT = "/api/v1/predict/{patient_id}/{model_name}"` 가 정의되어 있으나 **현재 미사용 (하위 호환 alias, 주석 처리된 TODO).**

타임아웃: `REQUEST_TIMEOUT_SECONDS = 30` ([api_client.py:24](dashboard/api_client.py#L24))

### 4-2. 호출하는 엔드포인트 목록

| 순서 | 메서드 | 경로 | 호출 시점 |
|---|---|---|---|
| 1 | GET | `/patients` | 앱 시작 시 1회 |
| 2 | POST | `/predict/{patient_id}` | 환자 선택 시 (환자별 `st.session_state["prediction_cache"]`에 캐시 → 재호출 방지) |
| 3 | GET | `/patients/{patient_id}/data` | 환자 선택 시마다 (Patient Bar 용) |

실패 시 `fetch_predictions`는 `None` 반환 → `fetch_dashboard_data`가 **mock 데이터로 fallback** (`MOCK_DASHBOARD_DATA`, [api_client.py:156](dashboard/api_client.py#L156)).

### 4-3. 예측 결과 화면 표시 방식

[dashboard/app.py](dashboard/app.py) 의 주요 렌더 함수:

| 렌더러 | 표시 내용 | 사용되는 응답 필드 |
|---|---|---|
| `render_page_header` ([app.py:1123](dashboard/app.py#L1123)) | 제목 · `meta.source_label` · 마지막 업데이트 · 새로고침/테마 버튼 | `meta.*` |
| `render_patient_bar` ([app.py:1165](dashboard/app.py#L1165)) | 환자명 · 나이 · 성별 · ICU 입실 · SOFA · 패혈증 onset | `/patients/{id}/data` 의 `patient_meta`(age/gender/intime/sepsis_onset_time) |
| `render_summary_cards` ([app.py:1236](dashboard/app.py#L1236)) | 4개 모델 카드 (한글명 + probability 도넛 + High/Moderate/Low 배지 + 데이터 부족 ⚠) | 모델별 `probability`, `has_api_data`, `data_quality.is_reliable` |
| `render_detail_panel` ([app.py:1298](dashboard/app.py#L1298)) | 선택된 모델 상세: **주요 기여 요인(SHAP 막대)** + **핵심 지표 측정값 테이블** + **임상 해석 텍스트** | `top_features`, `clinical_indicators`, `description` |

**위험도 색상 임계값** ([app.py:50](dashboard/app.py#L50) `_risk`):
- ≥ 0.70 → High (빨강)
- ≥ 0.40 → Moderate (노랑)
- else → Low (초록)

**SHAP 바 색상** ([app.py:951](dashboard/app.py#L951) `_shap_bars_html`):
- `shap_value ≥ 0` → 빨강 (`SHAP_POS = #ef4444`)
- `< 0` → 초록 (`SHAP_NEG = #22c55e`)
- `has_api_data=False` 시 회색 톤

**Mock 경로 식별**: 각 모델 결과는 `has_api_data` 플래그로 API/Mock 구분 → Mock일 때 도넛/배지/피처명이 회색으로 렌더되어 "Mock" 배지가 붙음.

**핵심 지표 테이블 2-track** ([app.py:1358](dashboard/app.py#L1358)):
- API 응답에 `clinical_indicators`가 있으면 → `_clinical_indicators_table_html` (risk_value에 따라 빨강/초록/기본색)
- 없으면 → `_feature_table_html` (mock top_feature_values, 전체 회색)

**환자 목록/선택**: 좌측 슬라이드 사이드바 (`render_sidebar_and_controls`) + `st.session_state["patient_id"]` + URL `?patient_id=...` 동기화.

---

## 5. 현재 배포 설정

### 5-1. Streamlit Cloud 배포 관련 파일

**Streamlit Cloud/Dockerfile/Procfile 등 배포 전용 설정 파일은 없음.** 저장소에서 탐색한 결과:
- `**/Dockerfile*` — 없음
- `**/Procfile` — 없음
- `**/.streamlit/**` (config.toml, secrets.toml) — 없음
- `st.secrets` 사용 — 없음

있는 것은 requirements 파일 3개와 `.env` 한 쌍.

| 파일 | 내용 | 용도 |
|---|---|---|
| [requirements.txt](requirements.txt) (루트) | streamlit, requests, plotly, python-dotenv + fastapi, uvicorn, boto3, pandas, pyarrow, numpy, scipy, torch, xgboost, shap, joblib | 루트에서 전체(프론트+백) 설치용 — 통합본 |
| [dashboard/requirements.txt](dashboard/requirements.txt) | streamlit, requests, plotly, python-dotenv | Streamlit 프론트 전용 |
| [pipeline/requirements.txt](pipeline/requirements.txt) | fastapi, uvicorn, boto3, pandas, pyarrow, numpy, scipy, torch, xgboost, shap, joblib | FastAPI 백엔드 전용. **주의: `tensorflow` 누락** — [aki_loader.py:8](pipeline/AKI/aki_loader.py#L8)가 `import tensorflow as tf`로 GRU `.h5`를 로드함. 현재 pipeline/requirements.txt로만 설치하면 AKI 추론이 실패함. |
| [dashboard/.env](dashboard/.env) | `API_BASE_URL=http://127.0.0.1:8000` | **git에 포함된 실제 .env** — `.gitignore`는 `.env`(바깥 경로) 한 줄만 있어 `dashboard/.env`는 추적됨. EC2 주소 주석도 포함(`http://43.203.9.88:8000`). |
| [dashboard/.env.example](dashboard/.env.example) | `API_BASE_URL=http://127.0.0.1:8000` | 템플릿 |

**Streamlit Cloud 배포 관점에서 부족한 것들:**
- `.streamlit/config.toml` (포트/테마/서버 설정)
- `.streamlit/secrets.toml` 혹은 `st.secrets` 접근 코드 — 현재는 dotenv(`.env`)만 읽음
- Streamlit Cloud에서 Python 버전 고정용 `runtime.txt` / `python_version` 필드
- pip로 설치할 `packages.txt`(apt 의존성)

### 5-2. 하드코딩된 환경변수 / API 주소

#### (a) API_BASE_URL — 프론트 → 백엔드

| 위치 | 값 |
|---|---|
| [dashboard/api_client.py:16](dashboard/api_client.py#L16) | `os.getenv("API_BASE_URL", "http://127.0.0.1:8000")` — **기본값 하드코딩** |
| [dashboard/.env:1](dashboard/.env#L1) | `API_BASE_URL=http://127.0.0.1:8000` |
| [dashboard/.env](dashboard/.env) 주석 | EC2 주소 주석: `http://43.203.9.88:8000` |
| [dashboard/.env.example:1](dashboard/.env.example#L1) | `API_BASE_URL=http://127.0.0.1:8000` |
| [dashboard/README.md:58, 62, 77, 80](dashboard/README.md) | 127.0.0.1:8000 / localhost:8000 / localhost:8501 참조 |
| [README.md:88](README.md#L88) | `http://localhost:8501` 접속 안내 |

#### (b) S3 버킷명 / prefix — 백엔드

| 위치 | 변수 | 하드코딩된 기본값 |
|---|---|---|
| [pipeline/api.py:24](pipeline/api.py#L24) | `S3_BUCKET` (환경변수 미사용, 리터럴) | `'say2-1team'` |
| [pipeline/api.py:25](pipeline/api.py#L25) | `PATIENT_PREFIX` (리터럴) | `'pipeline/patients'` |
| [pipeline/mortality/config.py:4-9](pipeline/mortality/config.py#L4) | `S3_BUCKET` / `MODEL_PREFIX` / `HISTORY_PREFIX` / `USE_S3` / `LOCAL_MODEL_PATH` | `'say2-1team'` / `'pipeline/final_model'` / `'pipeline/patient_history'` / `true` / `'./models'` |
| [pipeline/ARDS/ards_config.py:9-12](pipeline/ARDS/ards_config.py#L9) | 동일 | `'say2-1team'` / `'pipeline/final_model'` / `true` / `'./models/ards'` |
| [pipeline/SIC/sic_config.py:4-7](pipeline/SIC/sic_config.py#L4) | 동일 | `'say2-1team'` / `'pipeline/final_model'` / `true` / `'./models/sic'` |
| [pipeline/AKI/aki_config.py:4-7](pipeline/AKI/aki_config.py#L4) | 동일 | `'say2-1team'` / `'pipeline/final_model'` / `true` / `'./models/aki'` |
| [pipeline/SIC/predict.py:35-36](pipeline/SIC/predict.py#L35) | `S3_BUCKET` / `MODEL_PREFIX_SIC` (구버전, 현재 api.py가 참조 안 함) | `'say2-1team'` / `'Final_model/saved_models/sic'` |
| [pipeline/mortality/README.md:102-103](pipeline/mortality/README.md#L102) | 환경변수 export 예시 | `say2-1team` / `Final_model/saved_models` |
| [pipeline/SIC/README.md:124-125](pipeline/SIC/README.md#L124) | 동일 | `say2-1team` / `Final_model/saved_models/sic` |

#### (c) Threshold / 상수

- Mortality threshold: `0.21` ([pipeline/mortality/config.py:9](pipeline/mortality/config.py#L9))
- ARDS threshold: `0.30` (artifact 내부에서 로드, default `0.30` in [ards_predict.py:109](pipeline/ARDS/ards_predict.py#L109))
- SIC threshold: `0.5` ([pipeline/SIC/sic_config.py:8](pipeline/SIC/sic_config.py#L8))
- AKI threshold: `0.5` ([pipeline/AKI/aki_config.py:8](pipeline/AKI/aki_config.py#L8))
- SEQ_LEN: Mortality 48 / SIC 48 / AKI 12

### 5-3. AWS 자격증명

`boto3.client('s3')` 는 자격증명을 코드에 포함하지 않으며 **표준 boto3 체인에 의존** (env var `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` 또는 `~/.aws/credentials` 또는 IAM role). Streamlit Cloud 배포 시 별도 secrets 주입 경로가 필요함.

---

## 6. 참고 — 주요 경로 요약

- FastAPI 엔트리: [pipeline/api.py](pipeline/api.py) (`uvicorn pipeline.api:app --port 8000`)
- Streamlit 엔트리: [dashboard/app.py](dashboard/app.py) (`streamlit run dashboard/app.py`)
- API 클라이언트: [dashboard/api_client.py](dashboard/api_client.py)
- 환자 ID 소스: S3 `s3://say2-1team/pipeline/patients/` prefix 목록
- 모델 아티팩트 소스: S3 `s3://say2-1team/pipeline/final_model/`
- 환자 히스토리 저장소 (Mortality 전용): S3 `s3://say2-1team/pipeline/patient_history/{patient_id}/latest.json`
