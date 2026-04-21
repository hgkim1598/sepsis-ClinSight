import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import tensorflow as tf

from aki_config import LOCAL_MODEL_PATH

_gru_model = None
_xgb_model = None


def _load_models():
    gru_path = os.path.join(LOCAL_MODEL_PATH, 'aki_gru_final.h5')
    xgb_path = os.path.join(LOCAL_MODEL_PATH, 'aki_xgb_final.pkl')

    gru_model = tf.keras.models.load_model(gru_path)
    xgb_model = joblib.load(xgb_path)

    return gru_model, xgb_model


def get_models():
    global _gru_model, _xgb_model
    if _gru_model is None:
        print("AKI 모델 로드 중...")
        _gru_model, _xgb_model = _load_models()
        print("AKI 모델 로드 완료")
    return _gru_model, _xgb_model
