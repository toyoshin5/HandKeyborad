# hand_landmark.csvをXGBoostで学習させる
#多値分類問題
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

mode = "2D" #2D or 3D

#csvを読み込み
df = pd.read_csv('hand_landmark.csv')
df = df.dropna()
df = df.reset_index(drop=True)

#学習データとテストデータに分割
from sklearn.model_selection import train_test_split
X = df.drop('target',axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#前処理
#4から各点までの距離を特徴量に追加
for i in range(5,21):
    if mode == "3D":
        X_train['distance'+str(i)] = np.sqrt((X_train['x4']-X_train['x'+str(i)])**2+(X_train['y4']-X_train['y'+str(i)])**2+(X_train['z4']-X_train['z'+str(i)])**2)
        X_test['distance'+str(i)] = np.sqrt((X_test['x4']-X_test['x'+str(i)])**2+(X_test['y4']-X_test['y'+str(i)])**2+(X_test['z4']-X_test['z'+str(i)])**2)
    elif mode == "2D":
        X_train['distance'+str(i)] = np.sqrt((X_train['x4']-X_train['x'+str(i)])**2+(X_train['y4']-X_train['y'+str(i)])**2)
        X_test['distance'+str(i)] = np.sqrt((X_test['x4']-X_test['x'+str(i)])**2+(X_test['y4']-X_test['y'+str(i)])**2)
#4から最も近い点はどれか
X_train['min_distance'] = X_train[['distance5','distance6','distance7','distance8','distance9','distance10','distance11','distance12','distance13','distance14','distance15','distance16','distance17','distance18','distance19','distance20']].min(axis=1)
X_test['min_distance'] = X_test[['distance5','distance6','distance7','distance8','distance9','distance10','distance11','distance12','distance13','distance14','distance15','distance16','distance17','distance18','distance19','distance20']].min(axis=1)
#4から各座標までの角度を特徴量に追加
for i in range(5,21):
    X_train['angle'+str(i)] = np.arctan2((X_train['y4']-X_train['y'+str(i)]),(X_train['x4']-X_train['x'+str(i)]))
    X_test['angle'+str(i)] = np.arctan2((X_test['y4']-X_test['y'+str(i)]),(X_test['x4']-X_test['x'+str(i)]))
#xn,ynを消去
for i in range(0,21):
    X_train = X_train.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
    X_test = X_test.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
model = XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=100)
#ログを100回ごとに出力

model.fit(X_train,y_train,eval_metric=["merror", "mlogloss"],eval_set=[(X_test, y_test)],verbose=100)
#評価
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('accuracy_score:',accuracy_score(y_test, y_pred))

#表を作成
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()


pickle.dump(model, open('shiin_model_'+mode+'.pkl', 'wb'))



