"""Hàm cần cho pipeline câu d. Nằm trong module riêng để joblib nạp lại được (FunctionTransformer cần hàm import được)."""
from __future__ import annotations

import numpy as np


def scale_by(X, w):
    """Nhân các cột one-hot với trọng số w (không chuẩn hóa theo phương sai để mức hiếm không bị khuếch đại)."""
    return np.asarray(X, float) * w
