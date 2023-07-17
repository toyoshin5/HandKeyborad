

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def rotate_points(coords, a, b, c):
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

hand_xyz = [0.25097590684890747,0.5334060192108154,8.100722084236622e-07,0.3147541880607605,0.4432395398616791,-0.045924510806798935,0.3902142345905304,0.41922932863235474,-0.04368078336119652,0.4540112018585205,0.4661737382411957,-0.03488130494952202,0.5065555572509766,0.5015403032302856,-0.022376637905836105,0.4029878079891205,0.31191691756248474,0.05077483132481575,0.46953722834587097,0.31213122606277466,0.06228572502732277,0.5165619254112244,0.3374391198158264,0.04978443682193756,0.553362250328064,0.3620351254940033,0.035664476454257965,0.4118056893348694,0.3891797959804535,0.07224249839782715,0.47505897283554077,0.3865611255168915,0.08119460195302963,0.5188854932785034,0.40912261605262756,0.05049758031964302,0.5566449165344238,0.4322740435600281,0.023196078836917877,0.4110196530818939,0.45749178528785706,0.08450748771429062,0.47148311138153076,0.4544259309768677,0.08428109437227249,0.5094884634017944,0.4655401110649109,0.05156936123967171,0.5367550849914551,0.48965543508529663,0.025749219581484795,0.40818238258361816,0.5333566665649414,0.09198743104934692,0.46025222539901733,0.5201414227485657,0.08783706277608871,0.4906270205974579,0.5204014778137207,0.06943982094526291,0.5180841684341431,0.524652898311615,0.053049106150865555]
x_coords = hand_xyz[::3]
y_coords = hand_xyz[1::3]
z_coords = hand_xyz[2::3]

#xを反転    
x_coords = [-x for x in x_coords]


#===================================================================================================
#  指の平面を求める
#5~20の点
finger_coords = np.array([[x_coords[i], y_coords[i], z_coords[i]] for i in range(5, 21)])
# 最小二乗法による平面の方程式の解を求める
A = np.column_stack((finger_coords[:, 0], finger_coords[:, 1], np.ones(len(finger_coords))))
b = finger_coords[:, 2]
coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
# 平面の方程式の係数を取得
a, b, c = coefficients

# 平面の方程式を表示
print(f"平面の方程式: {a:.3f}x + {b:.3f}y + {c:.3f} = z")


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
#z = 0の平面に変換
res = rotate_points(np.array([x_coords, y_coords, z_coords]).T, a,b,c)
res = rotate_points_roll(res)
x_coords, y_coords, z_coords = res.T

#===================================================================================================
#プロット
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

# 平面を描画
# x = np.linspace(min(x_coords), max(x_coords), 10)
# y = np.linspace(min(y_coords), max(y_coords), 10)
# X, Y = np.meshgrid(x, y)
# Z = a * X + b * Y + c
#ax.plot_wireframe(X, Y, Z, color='r',linewidth=0.1)

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



# # 平面に変換
# twod_coords = convert_to_2D_coordinates(projaction_coords, projaction_coords[0])
# # 二次元座標を描画
# plt.scatter([x for x, y in twod_coords], [y for x, y in twod_coords])
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()



