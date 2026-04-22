---
title: sepsis-clinsight-api
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<div align="center">

# 🏥 ICU ClinSight

**패혈증 ICU 환자의 사망률 / ARDS / SIC / AKI 위험도를 예측하는 임상 의사결정 지원 대시보드**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Docker-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/cloud)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](#license)

### 🚀 [**Live Demo — DEMO_001 환자 대시보드**](https://sepsis-clinsight-jox4mrvrheepe8aojo2djm.streamlit.app/?patient_id=DEMO_001)

</div>

---

## 📖 Overview

ICU에 입원한 패혈증 환자의 **사망률·ARDS·AKI·SIC** 위험도를 실시간 추론으로 제공하는 임상 의사결정 지원(CDSS) 대시보드다.
시계열 활력징후와 랩 수치를 입력받아 4개 모델이 동시에 예측하고, SHAP 기반 주요 피처와 핵심 임상 지표를 함께 반환한다.

> 💡 **Project backstory**
> 원래 팀 프로젝트로 설계·학습한 모델들이었지만, EC2 인스턴스 운영 비용 때문에 팀플 종료 직후 백엔드가 내려갔다.
> 고생해서 만든 모델이 그대로 사장되는 게 아까워 **개인 포트폴리오 버전으로 아키텍처를 재설계**했다 —
> EC2를 **Hugging Face Spaces (Docker + FastAPI)** 로 옮기고, 모델 아티팩트는 **HF Private Model Repo**에서 기동 시 1회 다운로드하도록 바꿔 **완전 무료 스택**으로 돌아가게 만들었다.

---

## 🌐 Live Demo

| | |
|---|---|
| 🔗 **Dashboard** | [sepsis-clinsight-*.streamlit.app/?patient_id=DEMO_001](https://sepsis-clinsight-jox4mrvrheepe8aojo2djm.streamlit.app/?patient_id=DEMO_001) |
| 🧪 **샘플 환자** | `DEMO_001` ~ `DEMO_010` (합성 더미 10명, 시나리오별 위험도 스펙트럼) |
| ⚡ **콜드 스타트** | 첫 요청 시 HF Space가 잠시 웜업 (~30초) |

URL의 `?patient_id=` 쿼리를 `DEMO_002`, `DEMO_003`, ...으로 바꿔가며 시나리오 비교 가능하다.

---

## 🏗 Architecture

### Before — 팀플 시절

```
Streamlit Cloud ──HTTPS──▶ AWS EC2 (FastAPI + 4 models)
                                  ▲
                             AWS S3 (patients + models)

🪦 EC2 인스턴스 요금 부담으로 팀플 종료 후 백엔드 shutdown
```

### After — 개인 재설계 (현재)

```
┌──────────────────────┐   HTTPS   ┌──────────────────────────────┐
│  Streamlit Cloud     │ ────────▶ │  HF Spaces (Docker + FastAPI) │
│  dashboard/app.py    │           │  pipeline/api.py              │
│  frontend            │ ◀──────── │  4-model inference            │
└──────────────────────┘   JSON    └──────────────┬───────────────┘
                                                  │ startup pull
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │  HF Private Model Repo       │
                                   │  hgkim1598/                  │
                                   │  sepsis-clinsight-models     │
                                   └──────────────────────────────┘
```

| Layer | Service | 역할 | Cost |
|---|---|---|---|
| Frontend | Streamlit Community Cloud | 대시보드 UI (환자 선택, 시각화) | Free |
| Backend | HF Spaces (Docker SDK) | FastAPI 추론 서버 | Free |
| Model store | HF Private Model Repo | 학습된 모델 9개 | Free |
| Demo data | Repo-embedded synthetic | `demo/patients/DEMO_001~010/` | — |

---

## 🧠 Models

4개 모델이 한 요청에서 병렬 호출된다. 각 모델은 시계열(BiLSTM / GRU) + 정적 피처(XGBoost)의 스태킹 구조가 기본이다.

| 모델 | 예측 대상 | 아키텍처 |
|---|---|---|
| **Mortality** | ICU 사망률 | BiLSTM + XGBoost Stacking (meta: Logistic Regression) |
| **ARDS** | 급성호흡곤란증후군 | XGBoost (Platt calibrated) |
| **AKI** | 급성신손상 | GRU + XGBoost Stacking |
| **SIC** | 패혈증 유발 응고병증 | BiLSTM + XGBoost Stacking (meta: Logistic Regression) |

응답에는 **확률 · 예측 레이블 · 임상 핵심 지표 · SHAP Top-3 피처**가 포함된다.

---

## 📊 Data

| 구분 | 내용 |
|---|---|
| 🎓 **학습 데이터** | MIMIC-IV (PhysioNet) — 이 레포에는 포함하지 않음 |
| 🧪 **데모 환자** | 합성 더미 10명 (DEMO_001 ~ DEMO_010). MIMIC-IV의 컬럼 구조·수치 범위만 참조하여 `scripts/generate_demo_patients.py`로 생성 |
| 📜 **License** | MIMIC-IV 사용 시 **PhysioNet Credentialed Health Data License 1.5.0** 조건 준수 필요 (PhysioNet 인증 · CITI 수료 · DUA 서명) |

> ⚠️ 본 대시보드는 **데모·포트폴리오 시연용**이며, 임상적 의사결정에 사용해서는 안 된다.

---

## 🚀 Local Development

### Backend (FastAPI)

```bash
# deps
pip install -r pipeline/requirements.txt

# env
export HF_TOKEN=hf_xxx                          # private model repo 접근 토큰
export LOCAL_MODEL_PATH=./models                # 모델 캐시 경로
export PATIENTS_DIR=./demo/patients             # 데모 환자 루트

# run (기동 시 모델 9개가 HF 리포에서 LOCAL_MODEL_PATH로 1회 다운로드됨)
uvicorn pipeline.api:app --port 7860
```

사용 가능한 엔드포인트:
- `GET  /patients` — 환자 ID 목록
- `GET  /patients/{id}/data` — raw 시계열/랩/메타
- `POST /predict/{id}` — 4개 모델 병렬 추론

### Frontend (Streamlit)

```bash
pip install -r dashboard/requirements.txt

export API_BASE_URL=http://localhost:7860
streamlit run dashboard/app.py
```

### Docker (HF Spaces와 동일 이미지)

```bash
docker build -t sepsis-clinsight-api .
docker run --rm -p 7860:7860 -e HF_TOKEN=hf_xxx sepsis-clinsight-api
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Backend** | FastAPI · Uvicorn · Pydantic |
| **Frontend** | Streamlit · Plotly |
| **ML / DL** | PyTorch 2.11 · TensorFlow 2.21 · Keras 3.14 · scikit-learn 1.8 · XGBoost 3.2 · SHAP |
| **Data** | pandas · pyarrow · NumPy |
| **MLOps** | Hugging Face Hub (model registry) · Docker |
| **Hosting** | HF Spaces (backend) · Streamlit Community Cloud (frontend) |

---

## 📁 Repository Layout

```
sepsis-clinsight/
├── README.md                    # (이 문서 — HF Spaces frontmatter 포함)
├── Dockerfile                   # HF Spaces Docker 빌드 정의
│
├── pipeline/                    # FastAPI backend + 4 models
│   ├── api.py                   # 엔드포인트
│   ├── hf_model_loader.py       # 기동 시 HF repo → /app/models 다운로드
│   ├── mortality/  ARDS/  AKI/  SIC/
│   └── requirements.txt
│
├── dashboard/                   # Streamlit frontend
│   ├── app.py
│   ├── api_client.py
│   └── requirements.txt
│
├── demo/patients/               # 합성 환자 10명
│   └── DEMO_00X/{patient_meta.json, vital_ts.parquet, lab_df.parquet}
│
├── scripts/
│   └── generate_demo_patients.py
│
└── docs/
    ├── environment_analysis.md  # 모델 아티팩트 버전 역추적 노트
    └── repo_analysis.md
```

---

## 📜 License

MIT © 2026 hgkim

> MIMIC-IV 기반 모델을 재사용·재학습할 경우 **PhysioNet Credentialed Health Data License 1.5.0** 를 별도로 준수해야 한다.

---

<div align="center">

**Made with 🩺 for learning & portfolio**

</div>
