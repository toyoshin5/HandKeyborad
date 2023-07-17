

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def rotate_points_yaw_pitch(coords):
    #  指の平面を求める
    #5~20の点
    finger_coords = coords[5:21]
    # 最小二乗法による平面の方程式の解を求める
    A = np.column_stack((finger_coords[:, 0], finger_coords[:, 1], np.ones(len(finger_coords))))
    b = finger_coords[:, 2]
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 平面の方程式の係数を取得
    a, b ,_= coefficients
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
def convert_to_2D_coordinates(point_cloud, reference_point):

    # 基準点の座標
    x_ref, y_ref, _ = reference_point

    # 点群を二次元座標に変換
    coordinates_2D = []
    for point in point_cloud:
        x, y, _ = point
        x_2D = x - x_ref
        y_2D = y - y_ref
        coordinates_2D.append((x_2D, y_2D))

    return coordinates_2D

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



hand_xyz = [0.6918866151413895,0.5818294198624415,-0.09155796260506703,0.5941487944784334,0.4424916579625764,-0.1267355059556998,0.4834969672563541,0.3916900057758885,-0.10825522713068293,0.39331058289084836,0.4134639421429627,-0.0891436840881185,0.3142617922355008,0.4416772431230577,-0.06939599399030888,0.4642377240985375,0.37273820809129127,0.02287373237540093,0.3826636191832992,0.35892582154378105,0.053187819802751016,0.3288543788505156,0.3690536398899355,0.05031128042630939,0.2775763197073548,0.3760753825378812,0.04244093381060858,0.4754557889038784,0.46671000885906777,0.038868294506647624,0.39740082647663844,0.45403106246328007,0.06825489384026326,0.3390711948209472,0.4618770842333937,0.046514412906728045,0.2886661822283701,0.4728573263484382,0.025145158043845758,0.49303488282977725,0.5456296174802497,0.040445230317720064,0.41662026580065625,0.5453425969001566,0.06242667691267701,0.36030573672580385,0.5457073216197145,0.04487364442673846,0.3168724682255091,0.5523278632444167,0.02666924681459905,0.5143550550406885,0.6384157083820254,0.035054348385980066,0.44686804794880264,0.6233780283741073,0.05115541243037706,0.3976198478131851,0.6219205328960595,0.048257209547341605,0.350278735820999,0.6222329706818981,0.04192526860139927]
x_coords = hand_xyz[::3]
y_coords = hand_xyz[1::3]
z_coords = hand_xyz[2::3]

#xを反転    
#x_coords = [1-x for x in x_coords] #反転している場合は必要

#===================================================================================================
# #垂線
# # 4~20
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


