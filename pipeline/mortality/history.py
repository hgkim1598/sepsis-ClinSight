"""
S3 기반 이력(latest.json) 저장/로드 기능은 제거되었다.
predict.py가 이 모듈의 심볼을 호환용으로 import할 수 있도록
no-op 스텁만 남겨둔다.
"""
from __future__ import annotations


def load_latest(patient_id: str) -> dict | None:
    return None


def save_result(patient_id: str, result: dict) -> None:
    return None


def compute_changes(current_feats: dict, previous_result: dict | None) -> dict:
    return {}
