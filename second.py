# 로지스틱 회귀 사용 - 이진 분류 모델 -> 임계값 -> 3클래스 분류(angry/sad/normal)

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,classification_report

# 1. 데이터
f_df = pd.read_csv('preprocessed_emotion_hr_features_by_video.csv')
s_df = pd.read_csv('features_for_FT_classification.csv')[["patient_id", "mbti_tf"]]  #보강 (현재 mbti_tf 식별자가 없음)

df = f_df.merge(s_df,on='patient_id',how='left')

# 2. 라벨 및 특징 정의
df['label'] = df["emotion"].map({"sad":0, "angry": 1})
features = ["mean_hr","std_hr","min_hr","max_hr","range_hr"]
x = df[features].values    # 모델 입력
y = df["label"].values     # 정답 (sad=0 angry=1)
tf = df["mbti_tf"].values  # tf 구분

# 3. 데이터 분할
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
idx_train, idx_temp = next(gss.split(df,groups=df["patient_id"]))
train = df.iloc[idx_train]; temp=df.iloc[idx_temp]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
idx_val, idx_test = next(gss2.split(temp, groups=temp["patient_id"]))
val = temp.iloc[idx_val]; test = temp.iloc[idx_test]

# 4. 스케일링 + 모델 학습(이진 분류)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# scaler = StandardScaler().fit(train[features])
# x_train = scaler.transform(train[features])
# x_val = scaler.transform(val[features])
# x_test = scaler.transform(test[features])
x_train = train[features]
x_val = val[features]
x_test = test[features]

clf = LogisticRegression(max_iter=1000).fit(x_train,train["label"])
clf.fit(x_train,train["label"])

# 5. 임게값 (논리공식?)
def predict_3class(p_ang, th_ang, th_sad):
    p_sad = 1 - p_ang
    pred = np.zeros(len(p_ang),dtype=int)
    pred[p_ang>=th_ang]=2
    pred[(p_ang<th_ang)&(p_sad>=th_sad)]=1
    return pred

def sweep_for_group(p,y,mask):
    best={"th_ang":None,"th_sad":None,"score":-1}
    for ta in np.arange(0.3,0.71,0.01):
        for ts in np.arange(0.3,0.71,0.01):
            pred=predict_3class(p[mask],ta,ts)
            f1=f1_score(y[mask],pred,average="macro",labels=[1,2])
            if f1>best["score"]:
                best={"th_ang":round(ta,2),"th_sad":round(ts,2),"score":round(f1,3)}
    return best

p_val = clf.predict_proba(x_val)[:,1]
maskF, maskT = (val["mbti_tf"]==0),(val["mbti_tf"]==1)
bestF, bestT = sweep_for_group(p_val,val["label"],maskF),sweep_for_group(p_val,val["label"],maskT)

# 6. 테스트셋에 적용
p_test = clf.predict_proba(x_test)[:,1]

def apply_tf_threshold(p,tf):
    preds=[]
    for prob,grp in zip(p,tf):
        if grp==0:
            ta,ts=bestT["th_ang"],bestT["th_sad"]
        else:
            ta,ts=bestF["th_ang"],bestF["th_sad"]
        preds.append(2 if prob>=ta else 1 if (1-prob)>=ts else 0)
    return np.array(preds)

pred_test = apply_tf_threshold(p_test, test["mbti_tf"].values)
print("[VAL] T:",bestT," F:",bestF)
print(classification_report(test["label"], pred_test, target_names=["NORMAL","SAD","ANGRY"]))
