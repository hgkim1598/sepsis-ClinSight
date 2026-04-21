import os
import joblib
import torch
import xgboost as xgb

from config import LOCAL_MODEL_PATH
from model import BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_bilstm, _clf_xgb, _lr = None, None, None


def _load_models():
    bilstm_path = os.path.join(LOCAL_MODEL_PATH, 'mortality_bilstm.pt')
    xgb_path    = os.path.join(LOCAL_MODEL_PATH, 'mortality_xgb.json')
    lr_path     = os.path.join(LOCAL_MODEL_PATH, 'mortality_stacking_lr.pkl')

    state = torch.load(bilstm_path, map_location=device)
    bilstm = BiLSTM().to(device)
    bilstm.load_state_dict(state)
    bilstm.eval()

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(xgb_path)

    lr = joblib.load(lr_path)

    return bilstm, clf_xgb, lr


def get_models():
    global _bilstm, _clf_xgb, _lr
    if _bilstm is None:
        print("모델 로드 중...")
        _bilstm, _clf_xgb, _lr = _load_models()
        print("모델 로드 완료")
    return _bilstm, _clf_xgb, _lr
