# hand_landmark.csvをXGBoostで学習させる
#多値分類問題
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

mode = "2D" #2D or 3D

target_dict = {0:"あ",1:"い←",2:"う↑",3:"え→",4:"お↓"}
rev_target_dict = {v:k for k,v in target_dict.items()}

#csvを読み込み
df = pd.read_csv('hand_landmark_boin.csv')
df = df.dropna()
df = df.reset_index(drop=True)

#学習データとテストデータに分割
from sklearn.model_selection import train_test_split
X = df.drop('target',axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#前処理なし
model = XGBClassifier(early_stopping_rounds=5)
#ログを100回ごとに出力

model.fit(X_train,y_train,eval_metric=["mlogloss"],eval_set=[(X_train, y_train),(X_test, y_test)],verbose=1)

#trainとtestのグラフを出力
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

#評価
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('accuracy_score:',accuracy_score(y_test, y_pred))
#表を作成

#ヒラギノフォントを使う
fontpath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
font = {'family' : 'YuGothic'}
plt.rc('font', **font)
#軸のラベルはtarget_dictのキーを使う
labels = [target_dict[i] for i in range(5)]
#混同行列を作成
cm = confusion_matrix(y_test, y_pred)
#ヒートマップを作成
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels,fmt="d")
plt.xlabel('予測')
plt.ylabel('正解')
plt.show()

pickle.dump(model, open('boin_model_'+mode+'.pkl', 'wb'))



