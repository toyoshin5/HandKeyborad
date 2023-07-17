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

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"小"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"小":10}

def rotate_coordinates(rotation_matrix, coordinates):
    rotated_coordinates = np.einsum('ijk,jk->ik', rotation_matrix, coordinates)
    return np.array(rotated_coordinates[0]), np.array(rotated_coordinates[1])

#csvを読み込み
df = pd.read_csv('hand_landmark_10000.csv')
df = df.dropna()
df = df.reset_index(drop=True)

#学習データとテストデータに分割
from sklearn.model_selection import train_test_split
X = df.drop('target',axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#前処理
#xnを全て反転
for i in range(21):
    X_train['x'+str(i)] = X_train['x'+str(i)]*-1
    X_test['x'+str(i)] = X_test['x'+str(i)]*-1
#4から各点まで変位を計算
# hand_size_train = np.sqrt((X_train['x0']-X_train['x17'])**2+(X_train['y0']-X_train['y17'])**2)
# hand_size_test = np.sqrt((X_test['x0']-X_test['x17'])**2+(X_test['y0']-X_test['y17'])**2)

#0,17のベクトルが[0,1]となるように線形変換を行うための行列を計算
#アフィン変換から座標を求めるためには通常3点ずつ座標が必要だが、今回縦横比維持の成約付きのため、2点でよい


if mode == "2D":
    #回転変換を行う
    a_train = np.array([X_train['x17']-X_train['x0'],X_train['y17']-X_train['y0']])
    a_test = np.array([X_test['x17']-X_test['x0'],X_test['y17']-X_test['y0']])
    sin_train = a_train[1]/np.sqrt(a_train[0]**2+a_train[1]**2)
    cos_train = a_train[0]/np.sqrt(a_train[0]**2+a_train[1]**2)
    sin_test = a_test[1]/np.sqrt(a_test[0]**2+a_test[1]**2)
    cos_test = a_test[0]/np.sqrt(a_test[0]**2+a_test[1]**2)
    #回転行列を計算
    rot_train = np.array([[cos_train,sin_train],[-sin_train,cos_train]])
    rot_test = np.array([[cos_test,sin_test],[-sin_test,cos_test]])

    for i in range(21):
        # 座標の回転
        X_test['x'+str(i)],X_test['y'+str(i)] = rotate_coordinates(rot_test, np.array([X_test['x'+str(i)],X_test['y'+str(i)]]))
        X_train['x'+str(i)],X_train['y'+str(i)] = rotate_coordinates(rot_train, np.array([X_train['x'+str(i)],X_train['y'+str(i)]]))
    #中点を計算
    cnt = 21
    for i in range(5,20):
        if i%4 != 0:
            X_train["x"+str(cnt)],X_train["y"+str(cnt)],X_train["z"+str(cnt)] = (X_train['x'+str(i)]+X_train['x'+str(i+1)])/2,(X_train['y'+str(i)]+X_train['y'+str(i+1)])/2,(X_train['z'+str(i)]+X_train['z'+str(i+1)])/2
            X_test["x"+str(cnt)],X_test["y"+str(cnt)],X_test["z"+str(cnt)] = (X_test['x'+str(i)]+X_test['x'+str(i+1)])/2,(X_test['y'+str(i)]+X_test['y'+str(i+1)])/2,(X_test['z'+str(i)]+X_test['z'+str(i+1)])/2
            cnt += 1
#4から各点までの変位と距離を計算
hand_size_train = np.sqrt((X_train['x17']-X_train['x0'])**2+(X_train['y17']-X_train['y0'])**2)
hand_size_test = np.sqrt((X_test['x17']-X_test['x0'])**2+(X_test['y17']-X_test['y0'])**2)
if mode == "2D":
    for i in range(21,33):
        X_train['offset_x'+str(i)] = (X_train['x4']-X_train['x'+str(i)])/hand_size_train
        X_train['offset_y'+str(i)] = (X_train['y4']-X_train['y'+str(i)])/hand_size_train
        X_test['offset_x'+str(i)] = (X_test['x4']-X_test['x'+str(i)])/hand_size_test
        X_test['offset_y'+str(i)] = (X_test['y4']-X_test['y'+str(i)])/hand_size_test
        X_train['distance'+str(i)] = np.sqrt((X_train['x4']-X_train['x'+str(i)])**2+(X_train['y4']-X_train['y'+str(i)])**2)/hand_size_train
        X_test['distance'+str(i)] = np.sqrt((X_test['x4']-X_test['x'+str(i)])**2+(X_test['y4']-X_test['y'+str(i)])**2)/hand_size_test
    #xn,ynを消去
    for i in range(0,33):
        X_train = X_train.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
        X_test = X_test.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
if mode == "3D":
    for i in range(5,21):
        X_train['offset_x'+str(i)] = (X_train['x4']-X_train['x'+str(i)])/hand_size_train
        X_train['offset_y'+str(i)] = (X_train['y4']-X_train['y'+str(i)])/hand_size_train
        X_train['offset_z'+str(i)] = (X_train['z4']-X_train['z'+str(i)])/hand_size_train
        X_test['offset_x'+str(i)] = (X_test['x4']-X_test['x'+str(i)])/hand_size_test
        X_test['offset_y'+str(i)] = (X_test['y4']-X_test['y'+str(i)])/hand_size_test
        X_test['offset_z'+str(i)] = (X_test['z4']-X_test['z'+str(i)])/hand_size_test
        X_train['distance'+str(i)] = np.sqrt((X_train['x4']-X_train['x'+str(i)])**2+(X_train['y4']-X_train['y'+str(i)])**2+(X_train['z4']-X_train['z'+str(i)])**2)
        X_test['distance'+str(i)] = np.sqrt((X_test['x4']-X_test['x'+str(i)])**2+(X_test['y4']-X_test['y'+str(i)])**2+(X_test['z4']-X_test['z'+str(i)])**2)
    #xn,ynを消去
    for i in range(0,21):
        X_train = X_train.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
        X_test = X_test.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)


model = XGBClassifier(early_stopping_rounds=10)
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
labels = [target_dict[i] for i in range(11)]
#混同行列を作成
cm = confusion_matrix(y_test, y_pred)
#ヒートマップを作成
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels,fmt="d")
plt.xlabel('予測')
plt.ylabel('正解')
plt.show()

pickle.dump(model, open('shiin_model_'+mode+'.pkl', 'wb'))



