"""
ARDS 모델 아티팩트 로더.
로컬(LOCAL_MODEL_PATH)에서 joblib 아티팩트(dict: base_model / calibrator / features / threshold)를
한 번만 로드해 모듈 전역에 캐시한다.
"""

import sys, os
ARDS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARDS_DIR)

import joblib

from ards_config import LOCAL_MODEL_PATH, ARTIFACT_FILENAME


_artifact = None


def _load_artifact():
    path = os.path.join(LOCAL_MODEL_PATH, ARTIFACT_FILENAME)
    return joblib.load(path)


def _get_artifact():
    global _artifact
    if _artifact is None:
        print("[ARDS] 모델 로드 중...")
        _artifact = _load_artifact()
        print(f"[ARDS] 모델 로드 완료 (피처 {len(_artifact['features'])}개)")
    return _artifact
