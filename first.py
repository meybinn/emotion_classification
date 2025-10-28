# 이진분류 기반 random forest -> 2클래스 분류 (데이터 보강)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. 데이터
df = pd.read_csv("features_for_FT_classification.csv")

# 2. 감정라벨 재정의
df['emotion'] = df['emotion_sad'].map({'sad':0,'angry':1}) # 0: sad, 1: angry

# 3. 특징 
features = [
    'mean_hr_angry','std_hr_sad','range_hr_sad','min_hr_sad',  # sad
    'mean_hr_angry','std_hr_angry','range_hr_angry','min_hr_angry', # angry
    'delta_mean','delta_std','delta_range'  # delta values
]

x = df[features]
y = df['emotion']

# 4. train/test 분할
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 5. 모델 학습
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)

# 6. 평가
y_pred = rf_model.predict(X_test)
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("\n Classification Report: ",classification_report(y_test,y_pred))

# +. 과적합 테스트
print("Train Accuracy:", rf_model.score(X_train, y_train))
print("Test Accuracy:", rf_model.score(X_test, y_test))