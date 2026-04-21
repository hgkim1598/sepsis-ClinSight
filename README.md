---
title: sepsis-clinsight-api
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Sepsis-ClinSight

패혈증 ICU 환자의 **사망률 / ARDS / SIC / AKI** 위험도를 예측하는 임상 의사결정 지원 대시보드.

> ⚠️ 이 레포는 **개인 포트폴리오용 데모 버전**이다. 원본은 팀 프로젝트로 진행됐으며, 이 레포는 데모 배포 파이프라인(HF Spaces + Streamlit Cloud)과 합성 환자 데이터로 공개 시연이 가능하도록 재구성한 개인 버전이다.

---

## 아키텍처

```
┌──────────────────────┐      HTTPS       ┌──────────────────────────┐
│  Streamlit Cloud     │ ───────────────► │  HF Spaces (Docker+API)  │
│  dashboard/app.py    │                  │  pipeline/api.py         │
│  (프론트엔드)          │ ◄─────────────── │  FastAPI + 4 모델 추론      │
└──────────────────────┘      JSON        └──────────┬───────────────┘
                                                     │ startup pull
                                                     ▼
                                          ┌──────────────────────────┐
                                          │  HF Private Model Repo   │
                                          │  hgkim1598/               │
                                          │  sepsis-clinsight-models │
                                          └──────────────────────────┘
```

| 컴포넌트 | 역할 | 호스팅 |
|---|---|---|
| 프론트엔드 | Streamlit 대시보드 (환자 선택 · 추론 결과 시각화) | Streamlit Community Cloud |
| 백엔드 | FastAPI 추론 서버 (`/patients`, `/predict/{id}`) | HF Spaces (Docker SDK) |
| 모델 저장소 | 학습된 모델 가중치 9개 | HF Private Model Repo |
| 환자 데이터 | MIMIC-IV 구조를 모방한 **합성 데모 환자 10명** (DEMO_001 ~ DEMO_010) | 레포에 포함 (`demo/patients/`) |

## 모델

4개 모델이 한 요청으로 병렬 호출된다. 각 모델은 시계열(BiLSTM/GRU) + 정적 피처(XGBoost)의 스태킹 구조를 기본으로 한다.

| 모델 | 대상 | 구조 |
|---|---|---|
| Mortality | ICU 사망률 | BiLSTM + XGBoost 스태킹 (meta: Logistic Regression) |
| ARDS | 급성호흡곤란증후군 | XGBoost (Platt calibrated) |
| SIC | 패혈증 유발 응고병증 | BiLSTM + XGBoost 스태킹 (meta: Logistic Regression) |
| AKI | 급성신손상 | GRU + XGBoost 앙상블 |

응답에는 확률·예측 레이블·임상 지표·SHAP 상위 피처(Top-3)가 포함된다.

## 로컬 실행

### 1) 백엔드 (FastAPI)

```bash
# 의존성 설치
pip install -r pipeline/requirements.txt
pip install tensorflow==2.13.0      # AKI(.h5) 호환

# 환경변수
export HF_TOKEN=hf_xxx                            # 모델 리포 읽기 권한
export LOCAL_MODEL_PATH=./models                  # 모델 캐시 경로 (기본: /app/models)
export PATIENTS_DIR=./demo/patients               # 데모 환자 루트 (기본: /app/demo/patients)

# 실행 (시작 시 모델 9개가 HF 리포에서 LOCAL_MODEL_PATH로 1회 다운로드됨)
uvicorn pipeline.api:app --port 7860
```

기동 후:
- `GET http://localhost:7860/patients`
- `GET http://localhost:7860/patients/DEMO_001/data`
- `POST http://localhost:7860/predict/DEMO_001`

### 2) 프론트엔드 (Streamlit)

```bash
pip install -r dashboard/requirements.txt

export API_BASE_URL=http://localhost:7860
streamlit run dashboard/app.py
```

기본적으로 `http://localhost:8501`에서 접속 가능하다.

### 3) Docker로 백엔드 통째로 실행

```bash
docker build -t sepsis-clinsight-api .
docker run --rm -p 7860:7860 -e HF_TOKEN=hf_xxx sepsis-clinsight-api
```

## HF Spaces 배포 설정

이 레포의 루트(`Dockerfile` + 이 README의 YAML frontmatter)를 HF Spaces에 그대로 밀어넣으면 Docker SDK로 기동한다.

### Secrets (Settings → Repository secrets)

| 이름 | 설명 |
|------|------|
| `HF_TOKEN` | 모델 Private Repo 읽기 권한이 있는 토큰. **코드에 하드코딩하지 말 것.** Spaces가 컨테이너에 주입한다. |

### 환경 변수 (Dockerfile 기본값 기준, 필요 시 Spaces Variables에서 오버라이드)

| 이름 | 기본값 | 용도 |
|------|--------|------|
| `LOCAL_MODEL_PATH` | `/app/models` | 모델 캐시 경로 |
| `PATIENTS_DIR` | `/app/demo/patients` | 데모 환자 파일 루트 |
| `HF_REPO_ID` | `hgkim1598/sepsis-clinsight-models` | 모델 리포 ID |

### 의존성 정책
- **torch**: CPU 빌드로 고정 (`torch==2.0.1+cpu`). HF 무료 Spaces는 GPU가 없고 CPU 휠이 이미지 용량도 훨씬 작다.
- **tensorflow**: `2.13.0`으로 고정 (AKI GRU `.h5` 호환). `pipeline/requirements.txt`에는 넣지 않고 Dockerfile에서만 설치해 로컬 개발 충돌을 피한다.
- 그 외 런타임 의존성은 `pipeline/requirements.txt` 참조.

## 디렉토리 구조

```
sepsis-clinsight/
├── README.md                    # 이 문서 (HF Spaces frontmatter 포함)
├── Dockerfile                   # HF Spaces용 컨테이너 정의
├── requirements.txt             # 개발 편의용 통합 deps
│
├── pipeline/                    # FastAPI 백엔드 + 추론 모델
│   ├── api.py                   # 엔드포인트
│   ├── hf_model_loader.py       # 시작 시 HF 리포에서 모델 다운로드
│   ├── mortality/  ARDS/  SIC/  AKI/
│   └── requirements.txt
│
├── dashboard/                   # Streamlit 프론트엔드
│   ├── app.py
│   └── requirements.txt
│
├── demo/patients/               # 합성 환자 10명 (DEMO_001 ~ DEMO_010)
│   └── DEMO_00X/{patient_meta.json, vital_ts.parquet, lab_df.parquet}
│
└── scripts/
    └── generate_demo_patients.py
```

## 데이터 및 라이선스

- **학습 데이터**: MIMIC-IV (PhysioNet)로 학습됐다. 이 레포에는 **학습 데이터·모델 아티팩트 모두 포함하지 않는다**. 모델은 Private HF Repo에 저장되고 런타임에만 로드된다.
- **데모 환자 데이터**: `demo/patients/` 는 전부 **합성 데이터**다 — MIMIC-IV의 컬럼 구조와 수치 범위만 참조해 시나리오별로 스크립트(`scripts/generate_demo_patients.py`)로 생성했으며, 실제 환자 데이터는 한 건도 포함되지 않는다.
- **라이선스 준수**: MIMIC-IV는 **PhysioNet Credentialed Health Data License 1.5.0** 조건을 따른다. 본 레포를 fork하거나 확장해 MIMIC-IV 원본을 사용할 경우, PhysioNet 인증 · CITI Data or Specimens Only Research 수료 · DUA 서명 요건을 직접 충족해야 한다.
- **데모 배포 공개 범위**: 이 레포와 HF Space에 배포된 결과물은 모델 예측 인터페이스 시연만을 목적으로 하며, 임상적 의사결정에 사용해서는 안 된다.
