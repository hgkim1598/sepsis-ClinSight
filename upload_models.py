import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
api = HfApi()

model_files = [
    "models/mortality_bilstm.pt",
    "models/mortality_xgb.json",
    "models/mortality_stacking_lr.pkl",
    "models/ards_XGB.joblib",
    "models/sic_bilstm.pt",
    "models/sic_xgb.json",
    "models/sic_stacking_lr.pkl",
    "models/aki_gru_final.h5",
    "models/aki_xgb_final.pkl",
]

repo_id = "hgkim1598/sepsis-clinsight-models"

for file_path in model_files:
    filename = os.path.basename(file_path)
    print(f"업로드 중: {filename} ...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"완료: {filename}")

print("\n전체 업로드 완료!")