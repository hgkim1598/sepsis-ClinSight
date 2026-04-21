import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import torch
import xgboost as xgb

from sic_config import LOCAL_MODEL_PATH, INPUT_DIM
from sic_model import BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_bilstm, _clf_xgb, _lr = None, None, None


def _load_models():
    bilstm_path = os.path.join(LOCAL_MODEL_PATH, 'sic_bilstm.pt')
    xgb_path    = os.path.join(LOCAL_MODEL_PATH, 'sic_xgb.json')
    lr_path     = os.path.join(LOCAL_MODEL_PATH, 'sic_stacking_lr.pkl')

    state = torch.load(bilstm_path, map_location=device)
    bilstm = BiLSTM(input_dim=INPUT_DIM).to(device)
    bilstm.load_state_dict(state)
    bilstm.eval()

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(xgb_path)

    lr = joblib.load(lr_path)

    return bilstm, clf_xgb, lr


def get_models():
    global _bilstm, _clf_xgb, _lr
    if _bilstm is None:
        print("SIC 모델 로드 중...")
        _bilstm, _clf_xgb, _lr = _load_models()
        print("SIC 모델 로드 완료")
    return _bilstm, _clf_xgb, _lr
