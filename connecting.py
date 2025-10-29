import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

# scaler = joblib.load("scaler.pkl")
# thresholds = joblib.load("thresholds.pkl")
# clf_svm = joblib.load("model.pkl")
# clf_rf = joblib.load("model_rf.pkl")
# clf_gb = joblib.load("model_gb.pkl")

# app = FastAPI()

# # 입력 데이터 정의
# class EmotionInput(BaseModel):
#     mean_hr: float
#     std_hr: float
#     range_hr: float
#     mbti_tf: str  # "t" or "f"

# # 예측 함수
# def predict_3class(p_ang, mbti_tf):
#     group = "T" if mbti_tf.lower() == "t" else "F"
#     if thresholds.get("use", "TF") == "ALL":
#         group = "ALL"
#     th_ang = thresholds[group]["th_ang"]
#     th_sad = thresholds[group]["th_sad"]

#     if p_ang >= th_ang:
#         return 2  # angry
#     elif (1 - p_ang) >= th_sad:
#         return 1  # sad
#     else:
#         return 0  # normal

# # API
# @app.post("/predict")
# def predict(data: EmotionInput):
#     x = np.array([[data.mean_hr, data.std_hr, data.range_hr]])
#     x_scaled = scaler.transform(x)
#     p_ang = float(clf_gb.predict_proba(x_scaled)[:, 1])   # 일단 gradient boosting으로 선택
#     result = predict_3class(p_ang, data.mbti_tf)
#     label_map = {0: "normal", 1: "sad", 2: "angry"}

#     return {
#         "prob_angry": round(p_ang, 4),
#         "prob_sad": round(1 - p_ang, 4),
#         "class_id": result,
#         "class_name": label_map[result]
#     }


# json 형식
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, json

# ===== 모델 & 스케일러 로드 =====
scaler = joblib.load("scaler.pkl")
clf_gb = joblib.load("model_gb.pkl")

# thresholds.json 로드
with open("thresholds.json", "r", encoding="utf-8") as f:
    thresholds = json.load(f)

app = FastAPI()

# ===== 입력 데이터 정의 =====
class EmotionInput(BaseModel):
    mean_hr: float
    std_hr: float
    range_hr: float
    mbti_tf: str  # "t" or "f"

# ===== 예측 함수 =====
def predict_3class(p_ang, mbti_tf):
    group = "T" if mbti_tf.lower() == "t" else "F"
    if thresholds.get("use", "TF") == "ALL":
        group = "ALL"

    th_ang = thresholds[group]["th_ang"]
    th_sad = thresholds[group]["th_sad"]

    if p_ang >= th_ang:
        return 2  # angry
    elif (1 - p_ang) >= th_sad:
        return 1  # sad
    else:
        return 0  # normal

# ===== API =====
@app.post("/predict")
def predict(data: EmotionInput):
    x = np.array([[data.mean_hr, data.std_hr, data.range_hr]])
    x_scaled = scaler.transform(x)
    p_ang = float(clf_gb.predict_proba(x_scaled)[:, 1])  # 일단 gradient boosting
    result = predict_3class(p_ang, data.mbti_tf)

    label_map = {0: "normal", 1: "sad", 2: "angry"}
    return {
        "prob_angry": round(p_ang, 4),
        "prob_sad": round(1 - p_ang, 4),
        "class_id": result,
        "class_name": label_map[result],
    }
