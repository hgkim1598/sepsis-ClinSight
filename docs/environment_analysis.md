# Environment analysis — pipeline/ and models/

로컬에 있는 바이트코드와 모델 아티팩트에 남은 메타데이터로부터 **학습/저장 당시의 환경**과 **현재 런타임**을 역추적한 결과.

조사 방법:
- `.pyc` 파일명의 `cpython-XYZ` 태그 → Python ABI 버전
- `.pkl` / `.joblib` → `joblib.load` 후 `__getstate__()['_sklearn_version']` 추출
- `.pt` → `torch.load`로 state_dict 로드 후 키/shape 확인 (PyTorch는 버전을 파일에 박지 않음)
- `.h5` → `h5py`로 root attrs 직접 읽어 `keras_version` / `backend` 확인
- `xgboost .json` → 최상위 `version` 필드 및 `learner.attributes` 확인

---

## 1. pipeline/ 바이트코드로 본 Python 버전

| 디렉토리 | 발견된 cpython 태그 | 파일 수 | 추정 |
|---|---|---|---|
| `pipeline/` | cpython-311, cpython-312 | 3 (api/ 중복) | 3.11, 3.12 두 가지 해석기로 실행됨 |
| `pipeline/mortality/` | cpython-311, cpython-312 | 13 | 두 해석기 모두 사용 흔적 |
| `pipeline/ARDS/` | cpython-312 | 4 | 3.12만 |
| `pipeline/SIC/` | cpython-312 | 5 | 3.12만 |
| `pipeline/AKI/` | cpython-312 | 4 | 3.12만 |

**함의**
- 현재 Dockerfile은 `python:3.11-slim`이지만, 로컬에서는 **3.12**가 훨씬 많이 쓰였다 (ARDS/SIC/AKI는 3.12에서만 실행된 흔적).
- `pipeline/mortality/`는 과거 3.11에서도 실행된 적이 있음 (config/history/loader/model/preprocess cpython-311 존재).
- `pipeline/api.py`는 3.11·3.12 둘 다 실행된 기록 — 배포 이전 로컬 스모크 테스트가 양쪽에서 돌아갔다는 뜻.

---

## 2. models/ 아티팩트별 내장 버전 메타데이터

| 파일 | 확인된 버전 정보 | 추정 학습/저장 환경 |
|---|---|---|
| `mortality_bilstm.pt` | PyTorch state_dict (OrderedDict). 버전 미내장. shape: `lstm.weight_ih_l0 = (256, 18)` → input_dim=18, hidden=64, 2-layer BiLSTM | PyTorch 2.x 계열로 추정. 아키텍처가 단순 LSTM이라 2.0~2.11 호환 예상 |
| `mortality_xgb.json` | `version: [3, 2, 0]`, gbtree, `best_iteration=456` | **XGBoost 3.2.0** |
| `mortality_stacking_lr.pkl` | `LogisticRegression._sklearn_version = 1.8.0` (mtime 2026-04-22 01:49) | **sklearn 1.8.0**. 오늘 재저장됨 |
| `ards_XGB.joblib` | dict{`base_model`, `calibrator`, `features`, `threshold_from_val`, ...}. `base_model` = XGBoost 객체(runtime 3.2.0 보고), `calibrator._sklearn_version = 1.8.0` | **sklearn 1.8.0 + XGBoost 3.x**. base_model의 save 버전은 내부 XGB.json을 통해 3.x로 추정 |
| `sic_bilstm.pt` | PyTorch state_dict. shape: `lstm.weight_ih_l0 = (512, 15)` → input_dim=15 (config `INPUT_DIM=15`와 일치), hidden=128 | PyTorch 2.x |
| `sic_xgb.json` | `version: [3, 1, 3]`, gbtree, `best_iteration=147` | **XGBoost 3.1.3** — mortality/aki와 다른 버전 |
| `sic_stacking_lr.pkl` | `LogisticRegression._sklearn_version = 1.8.0` (mtime 2026-04-22 01:49) | **sklearn 1.8.0**. 오늘 재저장됨 |
| `aki_gru_final.h5` | root attrs: `backend=tensorflow`, **`keras_version=3.12.1`**. 모델: Sequential(GRU 128→BN→Dropout→GRU 64→Dropout→Dense 32→Dense 1) | **Keras 3.12.1** (standalone Keras 3.x)로 저장 |
| `aki_xgb_final.pkl` | `XGBClassifier` (runtime XGBoost 3.2.0 보고) | **XGBoost 3.2.0** |

---

## 3. 조사 세션의 호스트 런타임 (참고)

현재 분석 환경(`.venv/Scripts/python.exe`)에서 직접 확인:
- **Python**: 3.12.x (`.venv` 기본)
- **scikit-learn**: 1.8.0
- **xgboost**: 3.2.0
- **torch**: 2.11.0+cpu

---

## 4. Dockerfile 핀과의 충돌 지점

현재 `Dockerfile` 핀:
- `python:3.11-slim`
- `torch==2.0.1+cpu`
- `tensorflow==2.13.0`
- `scikit-learn==1.8.0`
- (requirements.txt) `xgboost` 버전 미핀

| 항목 | 저장 당시 | Dockerfile | 충돌 여부 |
|---|---|---|---|
| Python | 3.11 / 3.12 혼용 (주로 3.12) | 3.11 | 로컬 3.12에서 만들어진 pyc가 런타임에서 재생성되지만, 순수 모델 로딩엔 무관. 문제 없음 |
| sklearn | 1.8.0 | 1.8.0 | ✅ 일치 |
| XGBoost (mortality, aki) | 3.2.0 | (unpinned, 최신) | ✅ 최신이면 3.2.0 이상이라 상위호환 |
| XGBoost (sic) | 3.1.3 | (unpinned) | ⚠️ 3.1.3로 저장된 json을 3.2+에서 로드 — 일반적으로 상위호환이지만 **xgboost>=3.2.0** 명시 추천 |
| Keras / TF (AKI) | **Keras 3.12.1** | **tensorflow==2.13.0** (Keras 2.13 번들) | ❌ **충돌**. Keras 2.x는 Keras 3 포맷 `.h5`를 로드 못한다 |
| PyTorch | 2.x (shape 기준으로만 확인) | 2.0.1+cpu | ⚠️ 저장 환경이 2.11일 가능성이 높아 보이나, state_dict는 호환될 가능성 큼 |

### 권장 조치 (별도 작업으로 분리, 본 문서는 현상만 기록)
1. **Keras 3 호환 TF로 올리거나, 모델을 Keras 2로 재저장.**
   - 옵션 A: Dockerfile에서 `tensorflow==2.13.0` 제거 → `tensorflow==2.18.0` + `keras==3.12.1` (TF 2.16+부터 Keras 3 표준)
   - 옵션 B: 로컬에서 `aki_gru_final.h5`를 Keras 2 포맷으로 재저장 후 재업로드
2. **XGBoost 버전 핀 추가**: `pipeline/requirements.txt`에 `xgboost>=3.2.0`.
3. **Python 3.12로 통일**하면 로컬 재현성이 높아진다 (`python:3.12-slim`).

> 이 문서는 상태 스냅샷이다. 다음 학습 재저장 사이클 이후 다시 재생성 필요.
