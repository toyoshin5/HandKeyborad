import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import japanize_matplotlib

# CSVファイルの読み込み
data = pd.read_csv('../1stExp/hand_landmark_shiin_all.csv')


# 特徴量とラベルの分割
X = data.drop('target', axis=1)
# zn(2,5,8,,番目)を省く
X = X.drop(X.columns[range(2, 63, 3)], axis=1)
y = data['target']

# ランダムフォレストモデルの作成
model = RandomForestClassifier()
model.fit(X, y)

# 特徴量の重要度を取得
feature_importances = model.feature_importances_

# 特徴量の名前を取得
feature_names = X.columns

# xn, yn, znの重要度を合計
n_importances = {}
for i, name in enumerate(feature_names):
    n = name[1:] # 特徴量の番号を取得
    if n not in n_importances:
        n_importances[n] = 0
    n_importances[n] += feature_importances[i]

# 重要度と特徴量の番号をリストに変換
n_values = list(n_importances.values())
n_names = list(n_importances.keys())

# 特徴量の重要度を降順でソート
indices = np.argsort(n_values)[::-1]

# グラフに表示
plt.figure(figsize=(10, 6))
plt.bar(range(len(n_names)), np.array(n_values)[indices], align='center')
plt.xticks(range(len(n_names)), np.array(n_names)[indices], rotation=90)
plt.xlabel('MediaPipe Handsのランドマーク')
plt.ylabel('重要度')
plt.title('特徴量重要度')
plt.show()
