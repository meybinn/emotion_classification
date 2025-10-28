import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,classification_report
from sklearn.svm import SVC

# 1. 데이터
f_df = pd.read_csv('preprocessed_emotion_hr_features_by_video.csv')
s_df = pd.read_csv('features_for_FT_classification.csv')#[['patient_id','mbti_tf']]

df = f_df.merge(s_df,on='patient_id',how='left')
print(df.columns)   

# 2. 라벨 및 특징 정의
df['label'] = df["emotion"].map({"sad":0, "angry": 1})
features = [
    "mean_hr","std_hr","min_hr","max_hr","range_hr",
    "delta_mean","delta_std","delta_range",
    # "mean_hr_sad","mean_hr_angry"
]

x = df[features].values    # 모델 입력
y = df["label"].values     # 정답 (sad=0 angry=1)
tf = df["mbti_tf"].values  # tf 구분

# 3. 데이터 분할 -> train=모델 학습 val=임계점 test=성능 테스트
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE

gss = GroupShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
idx_train, idx_temp = next(gss.split(df,groups=df["patient_id"]))
train = df.iloc[idx_train]; temp=df.iloc[idx_temp]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
idx_val, idx_test = next(gss2.split(temp, groups=temp["patient_id"]))
val = temp.iloc[idx_val]; test = temp.iloc[idx_test]

x_train_bin_raw = train[features].values
y_train_bin = train["label"].values

# 스케일링
scaler = StandardScaler().fit(train[features])
x_train_bin_scaled = scaler.transform(x_train_bin_raw)
x_val = scaler.transform(val[features])
x_test = scaler.transform(test[features])

# SMOTE 적용
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_bin_scaled, y_train_bin)

# 예측 (3클래스 분류 함수)
def predict_3class(p_ang, th_ang, th_sad):
    p_sad = 1 - p_ang
    pred = np.zeros(len(p_ang), dtype=int)  # default: normal
    pred[p_ang >= th_ang] = 2               # angry
    pred[(p_ang < th_ang) & (p_sad >= th_sad)] = 1  # sad
    return pred

def sweep_for_group(p, y, mask):
    best = {"th_ang": None, "th_sad": None, "score": -1}
    for ta in np.arange(0.3, 0.71, 0.01):
        for ts in np.arange(0.3, 0.71, 0.01):
            pred = predict_3class(p[mask], ta, ts)
            f1 = f1_score(y[mask], pred, average="macro", labels=[1, 2])
            if f1 > best["score"]:
                best = {"th_ang": round(ta, 2), "th_sad": round(ts, 2), "score": round(f1, 3)}
    return best

# SVM 모델
clf_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
clf_svm.fit(x_train_resampled, y_train_resampled)

# 임계점
p_val_svm = clf_svm.predict_proba(x_val)[:, 1]  # angry 확률
maskT = (val["mbti_tf"]=="t")
maskF = (val["mbti_tf"]=="f")

bestF_svm = sweep_for_group(p_val_svm, val["label"], maskF)
bestT_svm = sweep_for_group(p_val_svm, val["label"], maskT)

# ---- 테스트셋 3클래스 판정 (T/F별 임계값 적용) ----
# 1) mbti_tf 문자열 정규화 (공백/대소문자 안전)
val_mbti_norm  = val["mbti_tf"].astype(str).str.strip().str.lower()
test_mbti_norm = test["mbti_tf"].astype(str).str.strip().str.lower()

# 2) 테스트셋 확률(p_angry) 산출
p_test_svm = clf_svm.predict_proba(x_test)[:, 1]      # angry 확률
tf_test_str = test_mbti_norm.to_numpy()               # 't' 또는 'f'
y_true = test["label"].to_numpy().astype(int)         # 0=SAD, 1=ANGRY

# 3) T/F별 (th_ang, th_sad) 적용하여 3클래스 예측 생성
def predict_3class_with_tf(p_ang, tf_group_str, bestT, bestF):
    out = np.zeros(len(p_ang), dtype=int)  # 0=NORMAL
    for i, (p, g) in enumerate(zip(p_ang, tf_group_str)):
        isT = (g == "t")
        th_ang = bestT["th_ang"] if isT else bestF["th_ang"]
        th_sad = bestT["th_sad"] if isT else bestF["th_sad"]
        # 구간 규칙: angry 우선 → sad → 나머지 normal
        out[i] = 2 if p >= th_ang else (1 if (1 - p) >= th_sad else 0)
    return out

pred3_test = predict_3class_with_tf(p_test_svm, tf_test_str, bestT_svm, bestF_svm)

# 4) 리포트: 본질 2클래스 성능(angry vs sad) + NORMAL 비율
#    NORMAL(0)은 오답으로 취급하여 2클래스로 압축해 평가
pred2_from3 = (pred3_test == 2).astype(int)   # 2(ANGRY)만 1, 나머지(NORMAL/SAD)는 0

from sklearn.metrics import classification_report
print("\n[SVM] 2-class report (from 3-class TF-threshold rule):")
print(classification_report(y_true, pred2_from3,
                            labels=[0,1], target_names=["SAD","ANGRY"], zero_division=0))
print("NORMAL rate on test:", float((pred3_test == 0).mean()))


p_test_svm = clf_svm.predict_proba(x_test)[:, 1]

pred_test_svm = clf_svm.predict(x_test)

# 평가 결과
print(classification_report(test["label"], pred_test_svm, labels=[0, 1], target_names=["SAD", "ANGRY"], zero_division=0))

print("train size:", len(train), "val size:", len(val), "test size:", len(test))
print("train F1:", f1_score(y_train_resampled, clf_svm.predict(x_train_resampled), average="macro"))
print("val F1:",   f1_score(val["label"], clf_svm.predict(x_val), average="macro"))