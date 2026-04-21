"""
HF Private Repo(hgkim1598/sepsis-clinsight-models)에서 모델 파일을
앱 시작 시 LOCAL_MODEL_PATH(/app/models)로 1회 다운로드한다.

- HF_TOKEN은 환경변수에서 읽음
- 이미 존재하는 파일은 스킵
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_REPO_ID = os.getenv("HF_REPO_ID", "hgkim1598/sepsis-clinsight-models")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models")

MODEL_FILES = [
    "mortality_bilstm.pt",
    "mortality_xgb.json",
    "mortality_stacking_lr.pkl",
    "ards_XGB.joblib",
    "sic_bilstm.pt",
    "sic_xgb.json",
    "sic_stacking_lr.pkl",
    "aki_gru_final.h5",
    "aki_xgb_final.pkl",
]


def download_models(
    repo_id: str = HF_REPO_ID,
    local_dir: str = LOCAL_MODEL_PATH,
    files: list[str] | None = None,
) -> list[str]:
    """HF 리포에서 모델 파일들을 local_dir로 다운로드.

    이미 local_dir에 존재하는 파일은 스킵한다.
    Returns: 실제로 (스킵 포함) 사용 가능해진 파일의 절대 경로 리스트.
    """
    token = os.getenv("HF_TOKEN")
    target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_list = files if files is not None else MODEL_FILES
    resolved: list[str] = []

    for fname in file_list:
        print(f"[hf_model_loader] downloading {fname} from {repo_id} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="model",
            local_dir=str(target_dir),
            token=token,
        )
        resolved.append(path)
        print(f"[hf_model_loader] done: {fname}")

    return resolved


if __name__ == "__main__":
    download_models()
