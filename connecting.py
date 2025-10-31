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
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, json

class EmotionInput(BaseModel):
    mean_hr: float
    std_hr: float
    range_hr: float
    mbti_tf: str                   # "t" or "f"
    debug_force_prob: Optional[float] = None  # ← 선택(0~1), 있으면 이 값으로 판정


# ===== 모델 & 스케일러 로드 =====
scaler = joblib.load("scaler.pkl")
clf_gb = joblib.load("model_gb.pkl")

# thresholds.json 로드
with open("thresholds.json", "r", encoding="utf-8") as f:
    thresholds = json.load(f)

app = FastAPI()

# ===== CORS 설정 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (프로덕션에서는 특정 도메인으로 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# # ===== 입력 데이터 정의 =====
# class EmotionInput(BaseModel):
#     mean_hr: float
#     std_hr: float
#     range_hr: float
#     mbti_tf: str  # "t" or "f"

# ===== 예측 함수 =====
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

# 임계갑말고 직접 받게
def predict_3class(p_ang, th_ang, th_sad):
    if p_ang >= th_ang:
        return 2  # angry
    elif (1 - p_ang) >= th_sad:
        return 1  # sad
    else:
        return 0  # normal

# ===== API =====
# @app.post("/predict")

# def predict(data: EmotionInput):
#     group = "T" if data.mbti_tf.strip().lower()=="t" else "F"
#     if thresholds.get("use","TF") == "ALL":
#         group = "ALL"
#     th = thresholds[group]
#     th_ang, th_sad = float(th["th_ang"]), float(th["th_sad"])
#     print(f"[CHK] group={group} th_ang={th_ang} th_sad={th_sad} gap={th_ang - (1 - th_sad):.3f}")
#     # gap > 0 이어야 NORMAL 구간 존재

#     x = np.array([[data.mean_hr, data.std_hr, data.range_hr]])
#     x_scaled = scaler.transform(x)
#     p_ang = float(clf_gb.predict_proba(x_scaled)[:, 1])  # 일단 gradient boosting
#     result = predict_3class(p_ang, data.mbti_tf)

#     label_map = {0: "normal", 1: "sad", 2: "angry"}
#     color_map = {0: "#4CAF50", 1: "#2196F3", 2: "#F44336"}  # Green, Blue, Red
#     emoji_map = {0: "😊", 1: "😢", 2: "😠"}
    
#     return {
#         "prob_angry": round(p_ang, 4),
#         "prob_sad": round(1 - p_ang, 4),
#         "class_id": result,
#         "class_name": label_map[result],
#         "color_hex": color_map[result],  # For Flutter UI
#         "emoji": emoji_map[result],      # For display
#         "mbti_group": data.mbti_tf.upper(),
#         "threshold_used": {
#             "th_angry": thresholds[data.mbti_tf.upper()]["th_ang"],
#             "th_sad": thresholds[data.mbti_tf.upper()]["th_sad"]
#         }
#     }

# 한 번 선택한 임계값을 로그, 판정, 응답 모두에 동일 적용
# @app.post("/predict")
# def predict(data: EmotionInput):
#     # --- 그룹/임계값 선택 (한 번만) ---
#     group = "T" if data.mbti_tf.strip().lower()=="t" else "F"
#     if thresholds.get("use","TF") == "ALL":
#         group = "ALL"

#     th = thresholds[group]
#     th_ang, th_sad = float(th["th_ang"]), float(th["th_sad"])
#     gap = th_ang - (1 - th_sad)
#     print(f"[CHK] group={group} th_ang={th_ang} th_sad={th_sad} gap={gap:.3f}")

#     # (선택) NORMAL 밴드 강제 확보
#     if gap <= 0:
#         bump = 0.02 - gap  # 최소 0.02 확보
#         th_ang = min(0.99, th_ang + bump/2)
#         th_sad = min(0.99, th_sad + bump/2)
#         print(f"[FIX] adjusted th_ang={th_ang:.3f} th_sad={th_sad:.3f} (gap>0 enforced)")

#     # --- 확률 산출 ---
#     x = np.array([[data.mean_hr, data.std_hr, data.range_hr]])
#     x_scaled = scaler.transform(x)
#     p_ang = float(clf_gb.predict_proba(x_scaled)[0, 1])

#     # --- 3클래스 판정 (같은 임계값 사용) ---
#     result = predict_3class(p_ang, th_ang, th_sad)

#     label_map = {0: "normal", 1: "sad", 2: "angry"}
#     color_map = {0: "#4CAF50", 1: "#2196F3", 2: "#F44336"}
#     emoji_map = {0: "😊", 1: "😢", 2: "😠"}

#     return {
#         "prob_angry": round(p_ang, 4),
#         "prob_sad": round(1 - p_ang, 4),
#         "class_id": result,
#         "class_name": label_map[result],
#         "color_hex": color_map[result],
#         "emoji": emoji_map[result],
#         "mbti_group": data.mbti_tf.upper(),
#         "threshold_used": {          
#             "group": group,
#             "th_angry": th_ang,
#             "th_sad": th_sad
#         }
#     }

@app.post("/predict")
def predict(data: EmotionInput):
    # --- 입력 정규화 & 그룹 결정 ---
    g = (data.mbti_tf or "").strip().lower()
    if g not in {"t", "f"}:
        g = "f"
    group = "T" if g == "t" else "F"

    use = (thresholds.get("use", "TF") or "TF").upper()
    if use == "ALL":
        group = "ALL"

    # --- 임계값 로드(방어 포함) ---
    th = thresholds.get(group)
    if not isinstance(th, dict) or "th_ang" not in th or "th_sad" not in th:
        th = thresholds.get("ALL") or thresholds.get("T") or thresholds.get("F")

    th_ang = float(th["th_ang"])
    th_sad = float(th["th_sad"])

    # --- NORMAL 폭 보장 ---
    target_gap = 0.30  # 서비스용 추천   0.20  0.25   0.30
    gap = th_ang - (1 - th_sad)
    if gap < target_gap:
        bump = target_gap - gap
        th_ang = min(0.99, th_ang + bump/2)
        th_sad = min(0.99, th_sad + bump/2)

    print(f"[CHK] group={group} use={use} th_ang={th_ang:.3f} th_sad={th_sad:.3f} gap={th_ang-(1-th_sad):.3f}")

    # --- 확률 ---
    # x = np.array([[data.mean_hr, data.std_hr, data.range_hr]], dtype=float)
    # x_scaled = scaler.transform(x)
    # p_ang = float(clf_gb.predict_proba(x_scaled)[0, 1])
    # --- 확률 계산 ---
    if data.debug_force_prob is not None:
        p_ang = float(data.debug_force_prob)        # ← 디버그: 모델 건너뛰고 이 확률 사용
    else:
        x = np.array([[data.mean_hr, data.std_hr, data.range_hr]], dtype=float)
        x_scaled = scaler.transform(x)
        p_ang = float(clf_gb.predict_proba(x_scaled)[0, 1])


    # --- 3클래스 판정 ---
    result = predict_3class(p_ang, th_ang, th_sad)

    label_map = {0: "normal", 1: "sad", 2: "angry"}
    color_map = {0: "#4CAF50", 1: "#2196F3", 2: "#F44336"}
    emoji_map = {0: "😊", 1: "😢", 2: "😠"}


    lower = 1.0 - th_sad
    upper = th_ang
    result = predict_3class(p_ang, th_ang, th_sad)

    return {
        "prob_angry": round(p_ang, 4),
        "prob_sad": round(1 - p_ang, 4),
        "class_id": result,
        "class_name": label_map[result],
        "color_hex": color_map[result],
        "emoji": emoji_map[result],
        "mbti_group": g.upper(),
        "threshold_used": { 
            "group": group, 
            "th_angry": th_ang, 
            "th_sad": th_sad 
        },
        "debug": {                           # ← 추가
        "band_lower": round(lower, 4),
        "band_upper": round(upper, 4),
        "gap": round(upper - lower, 4),
        "used_mode": use
    }
        
    }
