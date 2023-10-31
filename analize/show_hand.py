

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
#指の腹の位置を推定する関数xz平面で三角形を作るイメージ
def calc_finger_surface(coords):
    #指の骨の組み合わせ
    finger_bones = [(7,8),(6,7),(5,6),(11,12),(10,11),(9,10),(15,16),(14,15),(13,14),(19,20),(18,19)]
    finger_surface = np.empty((0,3), float)
    for bone in finger_bones:
        x1, y1, z1 = coords[bone[0]]
        x2, y2, z2 = coords[bone[1]]
        #中点を求める
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        z = (z1 + z2) / 2
        xz_dist = np.sqrt((x1-x2)**2 + (z1-z2)**2)
        #中点から指の腹の位置までの距離
        #日本人男性の中指の内径は約13号で約53mm
        #半径は約53/2/2/np.sqrt(3) = 9.2mm
        #中指の背側長は25mm(とする)
        #10と11の距離
        bone_length = np.sqrt((coords[10][0]-coords[11][0])**2 + (coords[10][1]-coords[11][1])**2 + (coords[10][2]-coords[11][2])**2)
        #指の半径を求める
        radius = bone_length * 9.2 / 25
        #指の腹の位置を求める。y座標はそのままで、x,z座標は中点からの距離をradiusにする
        newx = x + radius * (z1-z2) / xz_dist
        newz = z - radius * (x1-x2) / xz_dist
        finger_surface = np.append(finger_surface, np.array([[newx, y, newz]]), axis=0)
    return finger_surface
        
#3次元座標をプロットする関数
def showPlot(x_coords, y_coords, z_coords):


    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (17, 13),(9,13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords, y_coords, z_coords)

    for bone in bones:
        x_start = x_coords[bone[0]]
        y_start = y_coords[bone[0]]
        z_start = z_coords[bone[0]]
        x_end = x_coords[bone[1]]
        y_end = y_coords[bone[1]]
        z_end = z_coords[bone[1]]
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], 'b')
        
    # #指の腹の位置を推定
    # finger_surface = calc_finger_surface(np.array([x_coords, y_coords, z_coords]).T).T
    # ax.scatter(finger_surface[0], finger_surface[1], finger_surface[2], c='r')

    # スケールの調整
    max_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords))
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    ax.set_box_aspect([1, 1, 1])  # X, Y, Zのスケールを同じにする

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    plt.show()

#二次元座標のプロット。先頭の要素は赤になる。y軸は下向きに正。
def showPlot2D(x_coords, y_coords):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x_coords[1:], y_coords[1:])
    ax.scatter(x_coords[0], y_coords[0], c='r')

    # スケールの調整
    max_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)

    #y軸は下向きに正
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

def showPlotProjected(x,y,dst_x,dst_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y,c = 'r')
    #000111222
    dst_x = dst_x[5:]
    dst_y = dst_y[5:]
    ax.scatter(dst_x, dst_y, c = 'g')

     # スケールの調整
    max_range = max(max(dst_x) - min(dst_x), max(dst_y) - min(dst_y))
    mid_x = (max(dst_x) + min(dst_x)) * 0.5
    mid_y = (max(dst_y) + min(dst_y)) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)

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

#コマンドライン引数から文字列を取得
args = sys.argv[1]
hand_xyz = args.split(',')
hand_xyz = [float(x) for x in hand_xyz]
#hand_xyz = [0.641210122715859,0.4942765495288447,-0.23567364552656675,0.5868175186311222,0.3737301356641019,-0.24718040866765723,0.5253038096767172,0.31928673104032695,-0.2196821981200324,0.4592711871353451,0.3357012442194339,-0.19526091098245565,0.40447369047143433,0.356856016416988,-0.17167317596842058,0.5626615798685981,0.2530466975737158,-0.11374038702920768,0.5058888996974564,0.23641682169536737,-0.08303260720104466,0.4529722801505953,0.25218537825736936,-0.08529380611106135,0.40821969787216295,0.27078003463179656,-0.09107028633468742,0.5473898833401906,0.3319933304730334,-0.10086231107310824,0.48783398562896724,0.31314116272587766,-0.0715815761227944,0.43426232937275855,0.3224863011242835,-0.09501912880438287,0.3863795563801131,0.3380758645680298,-0.11513945882590497,0.5383601516784311,0.4004798084544264,-0.0982852529088293,0.48382822434275413,0.3868312554041477,-0.07642901907185062,0.4367829764057085,0.38579410498853794,-0.09800306800437898,0.3974196545692311,0.40340173320582284,-0.11685502432187879,0.5294659874522152,0.4803075963310187,-0.1018767700990754,0.4820731602638567,0.4588675467849866,-0.08106228636247684,0.44543015207050296,0.4536956939976679,-0.08464828577426604,0.4100415506209963,0.4535698004265451,-0.08960989168260167]

x_coords = hand_xyz[::3]
y_coords = hand_xyz[1::3]
z_coords = hand_xyz[2::3]

#xを反転    
#x_coords = [1-x for x in x_coords] #反転している場合は必要


#===================================================================================================
# #垂線
# # 4~20
# a,b,c = calc_regression_plane(np.array([x_coords, y_coords, z_coords]).T[5:21])
# projaction_coords = []#4~20の垂線の足の座標を管理
# for x,y,z in zip(x_coords[4:21],y_coords[4:21],z_coords[4:21]):
#     # 平面の法線ベクトルを計算
#     normal_vector = np.array([a, b, -1])
#     # 平面の交点を計算
#     t = (a * x + b * y + c - z) / (a**2 + b**2 + 1)
#     intersection = np.array([x - a * t, y - b * t, z + t])
#     projaction_coords.append(intersection)
#     #垂線の足の座標を描画
#     if x == x_coords[4]:
#         ax.scatter(intersection[0], intersection[1], intersection[2], color='r', s=10)
#===================================================================================================
#プロット1
showPlot(x_coords, y_coords, z_coords)
#===================================================================================================
#z = 0の平面に変換
res = rotate_points_yaw_pitch(np.array([x_coords, y_coords, z_coords]).T)
res = rotate_points_roll(res)
x_coords, y_coords, z_coords = res.T

#===================================================================================================
#プロット2
showPlot(x_coords, y_coords, z_coords)

showPlot2D(x_coords[4:], y_coords[4:])

#===================================================================================================
dst_hand_str = "0.8694720005051488,0.6125456286468557,-0.06241354109662445,0.7516713819717142,0.43684356905920096,-0.09647392046154543,0.6381812843289673,0.35163350530403464,-0.07786856799349441,0.5449889441771603,0.3718150043428731,-0.05390180717280749,0.48356775893366916,0.41711726855396314,-0.028768114143931094,0.6175030521856252,0.32643307216302664,0.022354811123633817,0.5195923122605438,0.3104407991399291,0.04859924589689326,0.4588744558591453,0.3229839990818058,0.04933180965922528,0.40819429074797436,0.3363441011883234,0.04543231097120396,0.6221451970062815,0.42423041963979286,0.04078988409432999,0.5168174777142056,0.41436397939655967,0.07089259683755883,0.4458627887340822,0.42117340823470206,0.058198739937397444,0.38572495711912297,0.4259225581564012,0.0444607104291028,0.6374113170324149,0.5179895460837254,0.046718498255842636,0.5347247829137108,0.5127066958010575,0.06290607752078221,0.4664677298219883,0.5112997507991703,0.041919823990423065,0.40719843586017457,0.5124882530017151,0.025269857674598644,0.6613577735680873,0.6159570970963899,0.04697400475925584,0.5787647613816045,0.6156460534125838,0.05369634626504267,0.5244440539331132,0.6159224447547323,0.04369347967938833,0.47222965890275387,0.6098552226364952,0.03447723725822396"
dst_hand_arr = dst_hand_str.split(',')
dst_hand_x = [float(x) for x in dst_hand_arr[::3]]
dst_hand_y = [float(x) for x in dst_hand_arr[1::3]]
for j in range(0,3):
    for i in range(0,3):
        # 8  7  6  5
        # 12 11 10 9
        # 16 15 14 13
        # 20 19 18 17
        index = 8+j*4-i
        src = [[x_coords[index], y_coords[index]], [x_coords[index-1], y_coords[index-1]], [x_coords[index+4], y_coords[index+4]], [x_coords[index+3], y_coords[index+3]]]
        dst = [[dst_hand_x[index], dst_hand_y[index]], [dst_hand_x[index-1], dst_hand_y[index-1]], [dst_hand_x[index+4], dst_hand_y[index+4]], [dst_hand_x[index+3], dst_hand_y[index+3]]]
        X = find_homography(src, dst)
        xc = []
        yc = []
        #変換後の座標を計算
        src = np.array([x_coords[4], y_coords[4], 1])
        dst = np.dot(X, src)
        new_x_coords = dst[0]/dst[2]
        new_y_coords = dst[1]/dst[2]
        #枠内かチェック
        flg = (i == 0 or i <= new_x_coords) and (i+1 == 3 or new_x_coords <= i+1) and (j == 0 or j <= new_y_coords) and (j+1 == 3 or new_y_coords <= j+1)
        print(new_x_coords, new_y_coords,flg,i,j,index)
        if flg:
            showPlotProjected(new_x_coords, new_y_coords,dst_hand_x,dst_hand_y)
        
