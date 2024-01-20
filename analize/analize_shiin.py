


from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2
import japanize_matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

LANDMARK_PATH = "../1stExp/hand_landmark_shiin_all.csv"
#データセットが左右反転しているかどうか
HAND_IS_REVERSED = False

CALC_AVERAGE_OF_HAND_AS_HOMOGRAPHY = False


class ConfidenceEllipse:
    def __init__(self, data, p=0.95):
        self.data = data
        self.p = p

        self.means = np.mean(data, axis=0)
        self.cov = np.cov(data[:,0], data[:,1])

        lambdas, vecs = np.linalg.eigh(self.cov)
        order = lambdas.argsort()[::-1]
        lambdas, vecs = lambdas[order], vecs[:,order]

        c = np.sqrt(chi2.ppf(self.p, 2))
        self.w, self.h = 2 * c * np.sqrt(lambdas)
        self.theta = np.degrees(np.arctan(
            ((lambdas[0] - lambdas[1])/self.cov[0,1])))
        
    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_patch(self, line_color="grey", face_color="none", alpha=0):
        el = Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el
    
def remove_outliers(X,Y):
    ellipse = ConfidenceEllipse(np.array([X,Y]).T,p = 0.98)
    means, w, h, theta = ellipse.get_params()
    #楕円に含まれ無い点は除外
    for i in range(len(X)-1,-1,-1):
        if (X[i]-means[0])**2/w**2 + (Y[i]-means[1])**2/h**2 > 1:
            X.pop(i)
            Y.pop(i)
    return X,Y

def in_rect(rect,target,i,j):
    #a - d
    #| e |
    #b - c
    a = (rect[0][0], rect[0][1])
    b = (rect[1][0], rect[1][1])
    c = (rect[2][0], rect[2][1])
    d = (rect[3][0], rect[3][1])
    e = (target[0], target[1])

    # 原点から点へのベクトルを求める
    vector_a = np.array(a)
    vector_b = np.array(b)
    vector_c = np.array(c)
    vector_d = np.array(d)
    vector_e = np.array(e)

    # 点から点へのベクトルを求める
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d
    vector_de = vector_e - vector_d

    # 外積を求める
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)

    #普通であれば、全てマイナスであれば、点は四角形の内側にある
    #端っこの四角形の場合は、端方向であればはみ出てもいいということにする
    return (i == 0 or vector_cross_ab_ae < 0) and (j+1 == 3 or vector_cross_bc_be < 0) and (i+1 == 3 or vector_cross_cd_ce < 0) and (j == 0 or vector_cross_da_de < 0)
def calc_regression_plane(coords):
    # 最小二乗法による平面の方程式の解を求める
    A = np.column_stack((coords[:, 0], coords[:, 1], np.ones(len(coords))))
    b = coords[:, 2]
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 平面の方程式の係数を取得
    a, b ,c = coefficients
    return a, b, c

def rotate_points_yaw_pitch(coords):
    #  指の平面を求める
    #5~20の点
    finger_coords = coords[5:21]
    # 最小二乗法による平面の方程式の解を求める
    a,b,_ = calc_regression_plane(finger_coords)
    # 平面の法線ベクトルを計算
    normal_vector = np.array([a, b, -1])
    # 法線ベクトルを正規化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    #yaw方向
    yaw_theta = np.arctan2(normal_vector[0], normal_vector[2])
    yaw_theta = -yaw_theta
    # 回転行列を計算
    rotation_matrix_yaw = np.array([[np.cos(yaw_theta),0, np.sin(yaw_theta)],
                                    [0, 1, 0],
                                    [-np.sin(yaw_theta), 0, np.cos(yaw_theta)]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_yaw, coords.T).T
    # #pitch方向
    pitch_theta = np.arctan2(normal_vector[1], normal_vector[2])
    pitch_theta = -pitch_theta
    # 回転行列を計算
    rotation_matrix_pitch = np.array([[1, 0, 0],
                                        [0, np.cos(pitch_theta), -np.sin(pitch_theta)],
                                        [0, np.sin(pitch_theta), np.cos(pitch_theta)]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_pitch, rotated_coords.T).T
    return rotated_coords

#ロール方向に回転する関数。各指が左を向くようにする
def rotate_points_roll(coords):
    #8->5, 12->9, 16->13, 20->17のベクトルを計算
    vector1 = -np.array([coords[8][0]-coords[5][0], coords[8][1]-coords[5][1]])
    vector2 = -np.array([coords[12][0]-coords[9][0], coords[12][1]-coords[9][1]])
    vector3 = -np.array([coords[16][0]-coords[13][0], coords[16][1]-coords[13][1]])
    vector4 = -np.array([coords[20][0]-coords[17][0], coords[20][1]-coords[17][1]])
    #平均をとる
    vector = (vector1 + vector2 + vector3 + vector4) / 4
    #roll方向
    roll_theta = np.arctan2(vector[1], vector[0])
    roll_theta = -roll_theta
    # 回転行列を計算
    rotation_matrix_roll = np.array([[np.cos(roll_theta), -np.sin(roll_theta), 0],
                                    [np.sin(roll_theta), np.cos(roll_theta), 0],
                                    [0, 0, 1]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_roll, coords.T).T
    return rotated_coords

def showPlotProjected(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y,c = 'r')
    #000111222
    x_coords = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    y_coords = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    ax.scatter(x_coords, y_coords, c = 'b')
    
     # スケールの調整xは-1~4, yは-1~4
    ax.set_xlim(-1, 4)
    

    #y軸は下向きに正
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

#4つの点から射影変換行列を求める関数
def find_homography(src, dst):
    
    # X = (x*h0 +y*h1 + h2) / (x*h6 + y*h7 + 1)
    # Y = (x*h3 +y*h4 + h5) / (x*h6 + y*h7 + 1)
    #
    # X = (x*h0 +y*h1 + h2) - x*h6*X - y*h7*X
    # Y = (x*h3 +y*h4 + h5) - x*h6*Y - y*h7*Y
    
    x1, y1 = src[0]
    x2, y2 = src[1]
    x3, y3 = src[2]
    x4, y4 = src[3]
    
    u1, v1 = dst[0]
    u2, v2 = dst[1]
    u3, v3 = dst[2]
    u4, v4 = dst[3]
    
    A = np.matrix([
            [ x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1, 0 ],
            [ 0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1, 0 ],
            [ x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2, 0 ],
            [ 0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2, 0 ],
            [ x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3, 0 ],
            [ 0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3, 0 ],
            [ x4, y4, 1, 0, 0, 0, -x4*u4, -y4*u4, 0 ],
            [ 0, 0, 0, x4, y4, 1, -x4*v4, -y4*v4, 0 ],
            [ 0, 0, 0,  0,  0, 0,      0,      0, 1 ],
            ])
    B = np.matrix([
            [ u1 ],
            [ v1 ],
            [ u2 ],
            [ v2 ],
            [ u3 ],
            [ v3 ],
            [ u4 ],
            [ v4 ],
            [ 1  ],
            ])
    
    X = A.I * B
    X.shape = (3, 3)
    return X.tolist()





#main
if __name__ == '__main__':
    #ファイル名を取得
    file_name = LANDMARK_PATH
    ave_x_coords = []
    ave_y_coords = []
    df = pd.read_csv(file_name)
    df = df.dropna()
    #1列目はラベルなので削除
    df = df.drop(df.columns[[0]], axis=1)
    df = df.reset_index(drop=True)
    df = df.astype(float)
    #各列の平均をとる
    if CALC_AVERAGE_OF_HAND_AS_HOMOGRAPHY:
        average_of_hand = df.mean()
        average_of_hand = average_of_hand.values.tolist()
        print("average_of_hand",end=":")
        #平均をリストに変換
        print(average_of_hand)
    else:
        average_of_hand = [0.7050574771237412, 0.7453036618739438, 4.220176619447094e-07, 0.6380715746489578, 0.5537182858377383, -0.04065226526935274, 0.5434539755573684, 0.4508825126798234, -0.03390088839189952, 0.44519127578768014, 0.4578004294824783, -0.024573237044675202, 0.37307509774826975, 0.4911852523618461, -0.012327942991832154, 0.5059959732602433, 0.4289161944964497, 0.05602689533240021, 0.4091820318843212, 0.3961846968947064, 0.0656691299043447, 0.34666060758071693, 0.3974543131953953, 0.05622074677117737, 0.2934040388063881, 0.40084353620359053, 0.04432528910008988, 0.4875388513352965, 0.5258393738025576, 0.06888396954525841, 0.3824632530048862, 0.5001935499669995, 0.08041936595391437, 0.31294162611955123, 0.49503579424764727, 0.05768020218652304, 0.254272694985392, 0.4891538744781836, 0.03621396545367038, 0.48339028725525146, 0.620041376293474, 0.07113007325131464, 0.3823291003992248, 0.598686353879096, 0.06993089157887986, 0.3179530892205949, 0.5856796870372836, 0.04052457782898027, 0.2617927516940426, 0.5772903134219355, 0.01728786701674265, 0.48810099663271433, 0.7192457505801241, 0.0684737746956558, 0.4057268740265577, 0.7043628966575376, 0.060886312493620955, 0.3526093629425231, 0.694492256150286, 0.04308659675822081, 0.30303866852684236, 0.6795648798394549, 0.027648239867118108]
    x_coords = average_of_hand[::3]
    y_coords = average_of_hand[1::3]    
    z_coords = average_of_hand[2::3]
    #z = 0の平面に変換
    #xを反転    
    if HAND_IS_REVERSED:
        x_coords = [1-x for x in x_coords] #反転している場合は必要
    res = rotate_points_yaw_pitch(np.array([x_coords, y_coords, z_coords]).T)
    res = rotate_points_roll(res)
    # #カンマ区切りでprint
    # for i in range(0,len(res)):
    #     for j in range(0,len(res[i])):
    #         print(res[i][j],end=",")
    print("\n")
    ave_x_coords, ave_y_coords, _ = res.T
        

    
    #ファイルを1行ずつ読み込む
    with open(file_name, mode='r') as f:
        lines = f.readlines()
    
    
    #各行を読み込んで、座標を取得
    X = [[] for i in range(11)]#11
    Y = [[] for i in range(11)]#11
    for k in range(0,len(lines)):
        #最初の行は飛ばす
        if lines[k][0] == 't':
            continue
        print("\r"+str(k),end="")
        lines[k] = lines[k].rstrip('\n')
        lines[k] = lines[k].split(',')
        label = int(lines[k][0])
        hand_xyz = [float(x) for x in lines[k][1:]]

        x_coords = hand_xyz[::3]
        y_coords = hand_xyz[1::3]
        z_coords = hand_xyz[2::3]

        # xを反転   
        if HAND_IS_REVERSED: 
            x_coords = [1-x for x in x_coords] #反転している場合は戻す

        #===================================================================================================
        
        #z = 0の平面に変換
        res = rotate_points_yaw_pitch(np.array([x_coords, y_coords, z_coords]).T)
        res = rotate_points_roll(res)
        x_coords, y_coords, z_coords = res.T

        for j in range(0,3):
            for i in range(0,3):
                # 8  7  6  5
                # 12 11 10 9
                # 16 15 14 13
                # 20 19 18 17
                index = 8+j*4-i
                rect = [[x_coords[index], y_coords[index]], [x_coords[index+4], y_coords[index+4]], [x_coords[index+3], y_coords[index+3]],[x_coords[index-1], y_coords[index-1]]]
                # 0 1 2 3
                # 1
                # 2
                # 3

                if in_rect(rect, [x_coords[4],y_coords[4]],i,j):
                    dst = [[ave_x_coords[index], ave_y_coords[index]], [ave_x_coords[index+4], ave_y_coords[index+4]], [ave_x_coords[index+3], ave_y_coords[index+3]],[ave_x_coords[index-1], ave_y_coords[index-1]]]
                    A = find_homography(rect, dst)
                    xc = []
                    yc = []
                    #変換後の座標を計算
                    src = np.array([x_coords[4], y_coords[4], 1])
                    dst = np.dot(A, src)
                    new_x_coords = dst[0]/dst[2]
                    new_y_coords = dst[1]/dst[2]
                    #プロット。labelの値によって色を変える
                    X[label].append(new_x_coords)
                    Y[label].append(new_y_coords)
                    #0~1の範囲に収まっているかどうか
                    # if USE_AVERAGE_OF_HAND_AS_HOMOGRAPHY and new_x_coords >= 0 and new_x_coords <= 1 and new_y_coords >= 0 and new_y_coords <= 1:
                    #     X[label].append(new_x_coords)
                    #     Y[label].append(new_y_coords)
                    # elif not USE_AVERAGE_OF_HAND_AS_HOMOGRAPHY and new_x_coords >= -1 and new_x_coords <= 4 and new_y_coords >= -1 and new_y_coords <= 4:
                    #     X[label].append(new_x_coords)
                    #     Y[label].append(new_y_coords)
                    # else:
                    #     print("\r",k+1,"行目の点が除外されました")

    #===================================================================================================
    #外れ値を除去
    for i in range(0,11):
        X[i],Y[i] = remove_outliers(X[i],Y[i])
    #===================================================================================================
    #プロット
    #凡例
    legend_str = ["あ","か","さ","た","な","は","ま","や","ら","変換","わ",]
    #指の骨の組み合わせ
    finger_bones = [(7,8),(6,7),(5,6),(11,12),(10,11),(9,10),(15,16),(14,15),(13,14),(19,20),(18,19),(17,18),(5,9),(9,13),(13,17)]
    for i,j in finger_bones:
        plt.plot([ave_x_coords[i], ave_x_coords[j]], [ave_y_coords[i], ave_y_coords[j]], c = 'b',linewidth=1)

    cm = plt.get_cmap("rainbow")
    #plot
    for i in range(0,11):
        marker = "o" if i < 6 else "^"
        plt.scatter(X[i], Y[i], c = cm(i%6/6),s = 3,marker=marker, label=legend_str[i])
    #目盛り
    plt.scatter(ave_x_coords[5:], ave_y_coords[5:], c = 'b',s=10)
    plt.xlim(0.2,0.7)
    plt.ylim(0.1,0.75)
    #目盛りを表示しない
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    #y軸は下向きに正
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    #===================================================================================================
    #楕円を描画
    #指の骨の組み合わせ
    finger_bones = [(7,8),(6,7),(5,6),(11,12),(10,11),(9,10),(15,16),(14,15),(13,14),(19,20),(18,19),(17,18),(5,9),(9,13),(13,17)]
    for i,j in finger_bones:
        plt.plot([ave_x_coords[i], ave_x_coords[j]], [ave_y_coords[i], ave_y_coords[j]], c = 'b',linewidth=1)

    #95%の信頼楕円を描画
    for i in range(0,11):
        ellipse = ConfidenceEllipse(np.array([X[i],Y[i]]).T)
        means, w, h, theta = ellipse.get_params()
        ellipse = ellipse.get_patch()
        ellipse.set_alpha(0.5)
        plt.gca().add_patch(ellipse)
    #plot
    for i in range(0,11):
        marker = "o" if i < 6 else "^"
        plt.scatter(X[i], Y[i], c = cm(i%6/6),s = 3,marker=marker, label=legend_str[i])
    #目盛り
    plt.scatter(ave_x_coords[5:], ave_y_coords[5:], c = 'b',s=10)
    plt.xlim(0.2,0.7)
    plt.ylim(0.1,0.75)
    #plt.title("参加者F")
    #目盛りを表示しない
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    #y軸は下向きに正
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

    #===================================================================================================
    #kNN
    k = 27
    X_train = []
    Y_train = []
    for i in range(0,11):
        for j in range(len(X[i])):
            X_train.append([X[i][j],Y[i][j]])
            Y_train.append(i)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, Y_train)
    #モデルを保存
    joblib.dump(clf, 'shiin_knn.pkl')

    # 結果を図示するコード
    # 0.01刻みのグリッド生成
    x_min = min([min(x) for x in X])
    x_max = max([max(x) for x in X])
    y_min = min([min(y) for y in Y])
    y_max = max([max(y) for y in Y])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))
    # 予測
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 結果を図示
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z ,20,cmap="jet",alpha=0.2)
    for i in range(0,11):
        plt.scatter(X[i], Y[i], c = cm(i/12),s = 2,label=legend_str[i])
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    #===================================================================================================
    #kNNの交差検証
    # データを訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=1)
    # 訓練データ、テストデータの精度を記録するための配列
    training_accuracy = []
    test_accuracy = []
    # n_neighborsを1から101まで試す
    neighbors_settings = range(1, 51,2)
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
        # 訓練データの精度を記録
        training_accuracy.append(clf.score(X_train, y_train))
        # テストデータの精度を記録
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()