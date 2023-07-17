#指の向きを補正するプログラム
import sys

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

#main
if __name__ == '__main__':
    #ファイル名を取得
    args = sys.argv
    file_name = args[1]
    #ファイルを1行ずつ読み込む
    with open(file_name, mode='r') as f:
        lines = f.readlines()
    #各行を読み込んで、座標を取得
    for i,line in enumerate(lines):
        #最初の行は飛ばす
        print(i)
        if line[0] == 't':
            with open('corrected_' + file_name, mode='a') as f:
                f.write("target,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6,x7,y7,z7,x8,y8,z8,x9,y9,z9,x10,y10,z10,x11,y11,z11,x12,y12,z12,x13,y13,z13,x14,y14,z14,x15,y15,z15,x16,y16,z16,x17,y17,z17,x18,y18,z18,x19,y19,z19,x20,y20,z20\n")
            continue
        line = line.rstrip('\n')
        line = line.split(',')
        label = line[0]
        line = [float(x) for x in line[1:]]
        x_coords = line[::3]
        y_coords = line[1::3]
        z_coords = line[2::3]
        x_coords = [1-x for x in x_coords]#ログが反転している場合に必要
        coords = np.array([x_coords, y_coords, z_coords]).T
        #指の向きを補正
        coords = rotate_points_yaw_pitch(coords)
        coords = rotate_points_roll(coords)
        #補正した座標をファイルに書き込む
        with open('corrected_' + file_name, mode='a') as f:
            f.write(label + ',')
            for i in range(len(coords)):
                if i == len(coords)-1:
                    f.write(str(coords[i][0]) + ',' + str(coords[i][1]) + ',' + str(coords[i][2]))
                else:
                    f.write(str(coords[i][0]) + ',' + str(coords[i][1]) + ',' + str(coords[i][2]) + ',')
            f.write('\n')


